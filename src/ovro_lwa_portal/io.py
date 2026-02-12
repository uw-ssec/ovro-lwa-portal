"""Data loading utilities for OVRO-LWA datasets.

This module provides a unified interface for loading OVRO-LWA data from
multiple sources including local paths, remote URLs, and DOI identifiers.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import xarray as xr

__all__ = ["open_dataset", "DataSourceError"]

# DOI pattern: matches both "doi:10.xxxx/xxxxx" and "10.xxxx/xxxxx"
DOI_PATTERN = re.compile(r"^(?:doi:)?(10\.\S+)$", re.IGNORECASE)

# DataCite API base URLs
DATACITE_API_PRODUCTION = "https://api.datacite.org/dois"
DATACITE_API_TEST = "https://api.test.datacite.org/dois"


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


def _resolve_doi(doi: str, production: bool = True) -> str:
    """Resolve DOI to zarr URL using DataCite Media API.

    Parameters
    ----------
    doi : str
        DOI identifier (e.g., "10.33569/9wsys-h7b71").
    production : bool, default True
        If True, use production DataCite API (api.datacite.org).
        If False, use test DataCite API (api.test.datacite.org).

    Returns
    -------
    str
        Resolved zarr URL.

    Raises
    ------
    DataSourceError
        If DOI resolution fails.
    """
    try:
        import requests
    except ImportError as e:
        msg = "requests is required for DOI resolution. Install with: pip install requests"
        raise ImportError(msg) from e

    api_base = DATACITE_API_PRODUCTION if production else DATACITE_API_TEST
    media_url = f"{api_base}/{doi}/media"

    try:
        r = requests.get(media_url, timeout=30)
        r.raise_for_status()
        data = r.json()["data"]

        # Prefer zarr media type, fallback to first URL
        for media in data:
            if media["attributes"]["mediaType"] == "application/zarr":
                return media["attributes"]["url"]

        if data:
            return data[0]["attributes"]["url"]
    except requests.exceptions.HTTPError as e:
        # If media endpoint returns 404, fall back to DOI metadata
        if e.response.status_code == 404:
            return _resolve_doi_from_metadata(doi, api_base)
        msg = f"Failed to resolve DOI {doi}: {e}"
        raise DataSourceError(msg) from e
    except Exception as e:
        msg = f"Failed to resolve DOI {doi}: {e}"
        raise DataSourceError(msg) from e

    # No media entries found, fall back to DOI metadata
    return _resolve_doi_from_metadata(doi, api_base)


def _resolve_doi_from_metadata(doi: str, api_base: str) -> str:
    """Resolve DOI to URL from DOI metadata attributes.

    This is a fallback when the DataCite Media API endpoint returns no results
    or a 404. It queries the DOI metadata directly for a URL.

    Parameters
    ----------
    doi : str
        DOI identifier.
    api_base : str
        DataCite API base URL.

    Returns
    -------
    str
        Resolved URL.

    Raises
    ------
    DataSourceError
        If resolution fails.
    """
    import requests

    doi_url = f"{api_base}/{doi}"

    try:
        r = requests.get(doi_url, timeout=30)
        r.raise_for_status()
        doi_data = r.json()

        url = doi_data.get("data", {}).get("attributes", {}).get("url")
        if url:
            return url

        msg = f"No download URL found in metadata for DOI: {doi}"
        raise DataSourceError(msg)
    except requests.exceptions.RequestException as e:
        msg = f"Failed to resolve DOI {doi} from metadata: {e}"
        raise DataSourceError(msg) from e


def _convert_osn_https_to_s3(url: str, storage_options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert OSN HTTPS URL to S3 URL when credentials are provided.

    OSN (Open Storage Network) provides both HTTPS and S3 access to the same data.
    When S3 credentials are provided, we need to use the S3 endpoint.

    Parameters
    ----------
    url : str
        HTTPS URL (e.g., https://caltech1.osn.mghpcc.org/path/to/data.zarr)
    storage_options : dict
        Storage options containing S3 credentials

    Returns
    -------
    tuple[str, dict]
        (s3_url, updated_storage_options) - S3 URL and storage options with endpoint
    """
    parsed = urlparse(url)
    if parsed.scheme == "https" and ".osn.mghpcc.org" in parsed.netloc:
        # Convert https://bucket.osn.mghpcc.org/path -> s3://bucket/path
        bucket = parsed.netloc.split(".")[0]
        path = parsed.path.lstrip("/")
        s3_url = f"s3://{bucket}/{path}"

        # Add OSN S3 endpoint to storage_options
        # Use virtual-host style endpoint (bucket in subdomain)
        updated_options = storage_options.copy()
        if "client_kwargs" not in updated_options:
            updated_options["client_kwargs"] = {}
        updated_options["client_kwargs"]["endpoint_url"] = f"https://{bucket}.osn.mghpcc.org"

        return (s3_url, updated_options)

    return (url, storage_options)


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
        warnings.warn(
            f"Dataset may not be OVRO-LWA format. "
            f"Expected dimensions like {expected_dims}, found {found_dims}",
            UserWarning,
            stacklevel=2,
        )

    # Check for common OVRO-LWA data variables
    common_vars = {"SKY", "BEAM"}
    found_vars = set(ds.data_vars.keys())

    if not found_vars.intersection(common_vars):
        warnings.warn(
            f"Dataset may not be OVRO-LWA format. "
            f"Expected variables like {common_vars}, found {found_vars}",
            UserWarning,
            stacklevel=2,
        )


def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    production: bool = True,
    storage_options: dict[str, Any] | None = None,
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
    production : bool, default True
        Which DataCite API to use when resolving DOI identifiers:
        - True: production API (api.datacite.org)
        - False: test API (api.test.datacite.org)
    storage_options : dict, optional
        Options passed to the filesystem backend (e.g., S3 credentials).
        Example: storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
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

    Load from test DataCite API with S3 credentials:

    >>> ds = ovro_lwa_portal.open_dataset(
    ...     "10.33569/4q7nb-ahq31",
    ...     production=False,
    ...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
    ... )

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

    # Resolve DOI to actual data URL
    if source_type == "doi":
        try:
            normalized_source = _resolve_doi(normalized_source, production=production)
            source_type = "remote"  # After resolution, treat as remote URL
        except Exception as e:
            msg = f"Failed to resolve DOI {normalized_source}: {e}"
            raise DataSourceError(msg) from e

    # Convert OSN HTTPS URLs to S3 when credentials are provided
    if source_type == "remote" and storage_options:
        normalized_source, storage_options = _convert_osn_https_to_s3(
            normalized_source, storage_options
        )

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

            # Create filesystem and mapper based on whether storage_options are provided
            if storage_options:
                # When storage options are provided, create filesystem directly
                # This ensures credentials are passed correctly to the backend
                try:
                    import fsspec
                except ImportError as e:
                    msg = (
                        "fsspec is required for remote storage access. "
                        "Install with: pip install fsspec"
                    )
                    raise ImportError(msg) from e

                # Parse the URL to get protocol and path
                parsed = urlparse(normalized_source)
                protocol = parsed.scheme if parsed.scheme else "file"

                # Create filesystem with storage options
                fs = fsspec.filesystem(protocol, **storage_options)

                # Get the path without the protocol
                if protocol in ("s3", "gs", "gcs"):
                    # For cloud storage: s3://bucket/path -> bucket/path
                    path = f"{parsed.netloc}/{parsed.path.lstrip('/')}"
                else:
                    path = normalized_source

                # Check local existence
                if protocol in ("", "file") and not Path(path).exists():
                    raise FileNotFoundError(f"Local path does not exist: {path}")

                # Create the mapper
                store = fs.get_mapper(path)
            else:
                # Without storage options, use UPath as before
                store_path = UPath(normalized_source)

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

            # Open the zarr store using the UPath
            # xr.open_zarr can handle fsspec mappers directly”
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
