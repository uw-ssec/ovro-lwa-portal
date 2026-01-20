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
    """Resolve DOI to zarr URL using caltechdata_api.

    Parameters
    ----------
    doi : str
        DOI identifier to resolve.

    Returns
    -------
    str
        URL to the zarr store.

    Raises
    ------
    DataSourceError
        If DOI cannot be resolved or zarr URL not found.
    ImportError
        If caltechdata_api is not installed.
    """
    try:
        from caltechdata_api import download_url
    except ImportError as e:
        msg = (
            "caltechdata_api is required for DOI resolution. "
            "Install with: pip install 'ovro_lwa_portal[remote]'"
        )
        raise ImportError(msg) from e

    logger.info(f"Resolving DOI via caltechdata_api: {doi}")

    try:
        # Try to get zarr media type first
        result = download_url(doi, media_type="application/zarr")
        if result:
            # Handle case where caltechdata_api returns dict instead of string
            url = result.get("url") if isinstance(result, dict) else result
            if url:
                logger.info(f"Resolved DOI to zarr URL: {url}")
                return url

        # Fallback to first available URL if no zarr media type
        url = download_url(doi)
        if url:
            logger.warning(f"No zarr media type found, using first URL: {url}")
            return url

        msg = f"No download URL found for DOI: {doi}"
        raise DataSourceError(msg)
    except Exception as e:
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

    # Load data based on engine
    try:
        if engine == "zarr":
            # Use fsspec's universal pathlib for unified handling of local and remote paths
            try:
                from upath import UPath
            except ImportError as e:
                msg = (
                    "universal-pathlib is required for path handling. "
                    "Install with: pip install universal-pathlib"
                )
                raise ImportError(msg) from e

            # Create UPath object - works for both local and remote paths
            store_path = UPath(normalized_source)
            logger.info(f"Loading zarr store from: {store_path}")

            # Explicit local existence check
            if store_path.protocol in ("", "file") and not store_path.exists():
                raise FileNotFoundError(f"Local path does not exist: {store_path}")

            # Build a Zarr store (fsspec mapper) from the UPath
            fs = store_path.fs
            store = fs.get_mapper(store_path.path)

            # Check if we need cloud storage backends for remote paths
            if source_type == "remote":
                parsed = urlparse(normalized_source)
                if parsed.scheme == "s3":
                    try:
                        import s3fs  # noqa: F401
                    except ImportError as e:
                        msg = (
                            "s3fs is required for S3 access. "
                            "Install with: pip install 'ovro_lwa_portal[remote]'"
                        )
                        raise ImportError(msg) from e
                elif parsed.scheme in ("gs", "gcs"):
                    try:
                        import gcsfs  # noqa: F401
                    except ImportError as e:
                        msg = (
                            "gcsfs is required for Google Cloud Storage access. "
                            "Install with: pip install 'ovro_lwa_portal[remote]'"
                        )
                        raise ImportError(msg) from e

            # Open the zarr store using the fsspec mapper
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                ds = xr.open_zarr(store, chunks=chunks, **kwargs)

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
