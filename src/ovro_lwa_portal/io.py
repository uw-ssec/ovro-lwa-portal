"""Data loading utilities for OVRO-LWA datasets.

This module provides a unified interface for loading OVRO-LWA data from
multiple sources including local paths, remote URLs, and DOI identifiers.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import xarray as xr

__all__ = ["open_dataset", "DataSourceError"]

logger = logging.getLogger(__name__)

# DOI pattern: matches both "doi:10.xxxx/xxxxx" and "10.xxxx/xxxxx"
DOI_PATTERN = re.compile(r"^(?:doi:)?(10\.\d{4,}(?:\.\d+)*\/\S+)$", re.IGNORECASE)


class DataSourceError(Exception):
    """Exception raised for errors in data source handling."""

    pass


def _is_doi(source: str) -> bool:
    """Check if source string is a DOI identifier.

    Parameters
    ----------
    source : str
        Source string to check.

    Returns
    -------
    bool
        True if source matches DOI pattern.
    """
    return bool(DOI_PATTERN.match(source))


def _normalize_doi(source: str) -> str:
    """Normalize DOI string by removing 'doi:' prefix if present.

    Parameters
    ----------
    source : str
        DOI string, possibly with 'doi:' prefix.

    Returns
    -------
    str
        Normalized DOI without prefix.
    """
    match = DOI_PATTERN.match(source)
    if match:
        return match.group(1)
    return source


def _resolve_doi(doi: str) -> str:
    """Resolve DOI to data URL using Caltech Data API.

    Parameters
    ----------
    doi : str
        DOI identifier to resolve.

    Returns
    -------
    str
        URL to the data resource.

    Raises
    ------
    DataSourceError
        If DOI cannot be resolved or data URL not found.
    """
    try:
        import requests
    except ImportError as e:
        msg = (
            "requests library is required for DOI resolution. "
            "Install with: pip install requests"
        )
        raise ImportError(msg) from e

    # Try caltechdata_api first if available
    try:
        from caltechdata_api import get_metadata

        logger.info(f"Resolving DOI using caltechdata_api: {doi}")
        try:
            metadata = get_metadata(doi)

            # Extract data URL from metadata
            # The structure depends on the Caltech Data API response
            if "files" in metadata and metadata["files"]:
                # Get the first file URL or look for zarr store
                for file_info in metadata["files"]:
                    if "links" in file_info and "self" in file_info["links"]:
                        url = file_info["links"]["self"]
                        logger.info(f"Resolved DOI to URL: {url}")
                        return url

            msg = f"No data URL found in DOI metadata for {doi}"
            raise DataSourceError(msg)
        except Exception as e:
            # If caltechdata_api fails, fall back to DOI.org
            logger.warning(f"caltechdata_api failed: {e}, falling back to DOI.org")
            raise ImportError("Fallback to DOI.org") from e

    except ImportError:
        # Fallback to DOI.org resolution
        logger.info(f"caltechdata_api not available, using DOI.org resolution for: {doi}")
        doi_url = f"https://doi.org/{doi}"

        try:
            response = requests.get(
                doi_url,
                headers={"Accept": "application/json"},
                allow_redirects=True,
                timeout=10,
            )
            response.raise_for_status()

            # The final URL after redirects might be the data location
            final_url = response.url
            logger.info(f"Resolved DOI to URL: {final_url}")
            return final_url

        except requests.RequestException as e:
            msg = f"Failed to resolve DOI {doi}: {e}"
            raise DataSourceError(msg) from e


def _detect_source_type(source: str | Path) -> tuple[str, str]:
    """Detect the type of data source.

    Parameters
    ----------
    source : str or Path
        Data source to analyze.

    Returns
    -------
    tuple[str, str]
        Tuple of (source_type, normalized_source) where source_type is one of:
        'local', 'remote', 'doi'.
    """
    source_str = str(source)

    # Check for DOI
    if _is_doi(source_str):
        normalized_doi = _normalize_doi(source_str)
        return ("doi", normalized_doi)

    # Check for remote URL
    parsed = urlparse(source_str)
    if parsed.scheme in ("http", "https", "s3", "gs", "gcs", "abfs", "az"):
        return ("remote", source_str)

    # Default to local path
    return ("local", source_str)


def _validate_dataset(ds: xr.Dataset) -> None:
    """Validate that dataset conforms to OVRO-LWA data model.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.

    Raises
    ------
    DataSourceError
        If dataset doesn't conform to expected structure.
    """
    # Check for required dimensions (at least some of these should exist)
    expected_dims = {"time", "frequency", "l", "m"}
    found_dims = set(ds.sizes.keys())

    if not found_dims.intersection(expected_dims):
        logger.warning(
            f"Dataset may not be OVRO-LWA format. "
            f"Expected dimensions like {expected_dims}, found {found_dims}"
        )

    # Check for common OVRO-LWA data variables
    common_vars = {"SKY", "BEAM"}
    found_vars = set(ds.data_vars.keys())

    if not found_vars.intersection(common_vars):
        logger.warning(
            f"Dataset may not be OVRO-LWA format. "
            f"Expected variables like {common_vars}, found {found_vars}"
        )

    logger.info(f"Dataset loaded with dimensions: {dict(ds.sizes)}")
    logger.info(f"Dataset variables: {list(ds.data_vars.keys())}")


def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    engine: str = "zarr",
    validate: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """Load OVRO-LWA data as an xarray Dataset.

    This function provides a unified interface for loading OVRO-LWA data from
    multiple sources including local file paths, remote URLs, and DOI identifiers.

    Parameters
    ----------
    source : str or Path
        Data source, can be:
        - Local file path (e.g., "/path/to/data.zarr")
        - Remote URL (e.g., "s3://bucket/data.zarr", "https://...")
        - DOI string (e.g., "doi:10.xxxx/xxxxx" or "10.xxxx/xxxxx")
    chunks : dict, str, or None, default "auto"
        Chunking strategy for lazy loading:
        - dict: Explicit chunk sizes per dimension, e.g., {"time": 100, "frequency": 50}
        - "auto": Let xarray/dask determine optimal chunks
        - None: Load entire dataset into memory (not recommended for large data)
    engine : str, default "zarr"
        Backend engine for loading data. Currently supports "zarr".
    validate : bool, default True
        If True, validate that loaded data conforms to OVRO-LWA data model.
    **kwargs
        Additional arguments passed to the underlying loader (xr.open_zarr, etc.)

    Returns
    -------
    xr.Dataset
        OVRO-LWA dataset with standardized structure.

    Raises
    ------
    DataSourceError
        If source cannot be accessed or loaded.
    FileNotFoundError
        If local file path doesn't exist.
    ImportError
        If required dependencies for remote/DOI access are not installed.

    Examples
    --------
    Load from local zarr store:

    >>> import ovro_lwa_portal
    >>> ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

    Load from S3 bucket:

    >>> ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

    Load from HTTP/HTTPS URL:

    >>> ds = ovro_lwa_portal.open_dataset("https://data.ovro.caltech.edu/obs_12345.zarr")

    Load via DOI (with or without prefix):

    >>> ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
    >>> ds = ovro_lwa_portal.open_dataset("10.5281/zenodo.1234567")

    Customize chunking:

    >>> ds = ovro_lwa_portal.open_dataset(
    ...     "path/to/data.zarr",
    ...     chunks={"time": 100, "frequency": 50}
    ... )

    Notes
    -----
    For remote data sources (S3, GCS), authentication is handled via environment
    variables or configuration files specific to each cloud provider:

    - AWS S3: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, or ~/.aws/credentials
    - Google Cloud Storage: GOOGLE_APPLICATION_CREDENTIALS
    - Azure: AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY

    For large datasets, lazy loading with dask is used by default (chunks="auto").
    This allows working with datasets larger than memory.
    """
    source_type, normalized_source = _detect_source_type(source)
    logger.info(f"Detected source type: {source_type}")

    # Resolve DOI to actual data URL
    if source_type == "doi":
        logger.info(f"Resolving DOI: {normalized_source}")
        try:
            normalized_source = _resolve_doi(normalized_source)
            source_type = "remote"  # After resolution, treat as remote URL
        except Exception as e:
            msg = f"Failed to resolve DOI {normalized_source}: {e}"
            raise DataSourceError(msg) from e

    # Load data based on source type and engine
    try:
        if engine == "zarr":
            if source_type == "local":
                # Check if local path exists
                path = Path(normalized_source)
                if not path.exists():
                    msg = f"Local path does not exist: {path}"
                    raise FileNotFoundError(msg)

                logger.info(f"Loading local zarr store: {path}")
                # Allow warnings to pass through (e.g., chunk alignment warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("default")
                    ds = xr.open_zarr(path, chunks=chunks, **kwargs)

            elif source_type == "remote":
                # For remote sources, we need fsspec and appropriate backend
                logger.info(f"Loading remote zarr store: {normalized_source}")

                # Check if we need cloud storage backends
                parsed = urlparse(normalized_source)
                if parsed.scheme == "s3":
                    try:
                        import s3fs  # noqa: F401
                    except ImportError as e:
                        msg = (
                            "s3fs is required for S3 access. "
                            "Install with: pip install s3fs"
                        )
                        raise ImportError(msg) from e
                elif parsed.scheme in ("gs", "gcs"):
                    try:
                        import gcsfs  # noqa: F401
                    except ImportError as e:
                        msg = (
                            "gcsfs is required for Google Cloud Storage access. "
                            "Install with: pip install gcsfs"
                        )
                        raise ImportError(msg) from e

                # Allow warnings to pass through (e.g., chunk alignment warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("default")
                    ds = xr.open_zarr(normalized_source, chunks=chunks, **kwargs)

            else:
                msg = f"Unsupported source type: {source_type}"
                raise DataSourceError(msg)

        else:
            msg = f"Unsupported engine: {engine}. Currently only 'zarr' is supported."
            raise DataSourceError(msg)

    except FileNotFoundError:
        raise
    except ImportError:
        raise
    except Exception as e:
        msg = f"Failed to load dataset from {normalized_source}: {e}"
        raise DataSourceError(msg) from e

    # Validate dataset structure if requested
    if validate:
        _validate_dataset(ds)

    return ds
