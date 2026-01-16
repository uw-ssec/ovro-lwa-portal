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

import requests
import xarray as xr

__all__ = ["open_dataset", "DataSourceError"]

# DOI pattern: matches both "doi:10.xxxx/xxxxx" and "10.xxxx/xxxxx"
DOI_PATTERN = re.compile(r"^(?:doi:)?(10\.\S+)$", re.IGNORECASE)

# DataCite API URL pattern: matches both production and test API URLs
# Examples:
#   https://api.datacite.org/dois/10.33569/9wsys-h7b71
#   https://api.test.datacite.org/dois/10.33569/9wsys-h7b71
DATACITE_API_PATTERN = re.compile(
    r"^https?://api\.(?:test\.)?datacite\.org/dois/(10\.\S+)$", re.IGNORECASE
)

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


def _is_datacite_api_url(source: str) -> bool:
    """Check if source string is a DataCite API URL.

    Parameters
    ----------
    source : str
        Source string to check.

    Returns
    -------
    bool
        True if source matches DataCite API URL pattern.
    """
    return bool(DATACITE_API_PATTERN.match(source))


def _is_datacite_test_url(source: str) -> bool:
    """Check if source string is a DataCite TEST API URL.

    Parameters
    ----------
    source : str
        Source string to check.

    Returns
    -------
    bool
        True if source is a DataCite test API URL.
    """
    return "api.test.datacite.org" in source.lower()


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


def _extract_doi_from_datacite_url(url: str) -> str:
    """Extract DOI from a DataCite API URL.

    Parameters
    ----------
    url : str
        DataCite API URL.

    Returns
    -------
    str
        Extracted DOI.

    Raises
    ------
    ValueError
        If URL doesn't match DataCite API pattern.
    """
    match = DATACITE_API_PATTERN.match(url)
    if match:
        return match.group(1)
    raise ValueError(f"Not a valid DataCite API URL: {url}")


def _resolve_doi(doi: str, production: bool = True) -> str:
    """Resolve DOI to zarr URL using DataCite Media API.

    Parameters
    ----------
    doi : str
        DOI identifier (e.g., "10.33569/9wsys-h7b71").
    production : bool, default True
        If True, use production DataCite API.
        If False, use test DataCite API.

    Returns
    -------
    str
        Resolved zarr URL.

    Raises
    ------
    DataSourceError
        If DOI resolution fails.
    """
    # Select API base URL
    api_base = DATACITE_API_PRODUCTION if production else DATACITE_API_TEST

    # First, try to get media URLs from the DOI
    media_url = f"{api_base}/{doi}/media"

    try:
        response = requests.get(media_url, timeout=30)
        response.raise_for_status()
        media_data = response.json()

        # Look for zarr media type first
        media_list = media_data.get("data", [])
        for media in media_list:
            attrs = media.get("attributes", {})
            media_type = attrs.get("mediaType", "")
            url = attrs.get("url", "")
            if "zarr" in media_type.lower() and url:
                return url

        # If no zarr media type, try the first available URL
        for media in media_list:
            attrs = media.get("attributes", {})
            url = attrs.get("url", "")
            if url:
                return url

    except requests.exceptions.HTTPError as e:
        # If media endpoint fails, try getting URL from DOI metadata
        if e.response.status_code == 404:
            return _resolve_doi_from_metadata(doi, api_base)
        raise DataSourceError(f"Failed to resolve DOI {doi}: {e}") from e
    except requests.exceptions.RequestException as e:
        raise DataSourceError(f"Failed to resolve DOI {doi}: {e}") from e

    # Fallback: try getting URL from DOI metadata
    return _resolve_doi_from_metadata(doi, api_base)


def _resolve_doi_from_metadata(doi: str, api_base: str) -> str:
    """Resolve DOI to URL from DOI metadata attributes.

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
    doi_url = f"{api_base}/{doi}"

    try:
        response = requests.get(doi_url, timeout=30)
        response.raise_for_status()
        doi_data = response.json()

        # Try to get URL from attributes
        attrs = doi_data.get("data", {}).get("attributes", {})
        url = attrs.get("url")
        if url:
            return url

        msg = f"No download URL found for DOI: {doi}"
        raise DataSourceError(msg)

    except requests.exceptions.RequestException as e:
        raise DataSourceError(f"Failed to resolve DOI {doi}: {e}") from e


def _detect_source_type(source: str | Path) -> tuple[str, str, bool]:
    """Detect the type of data source.

    Parameters
    ----------
    source : str or Path
        Data source to analyze.

    Returns
    -------
    tuple[str, str, bool]
        Tuple of (source_type, normalized_source, use_test_api) where:
        - source_type is one of: 'local', 'remote', 'doi', 'datacite_api'
        - normalized_source is the DOI or path
        - use_test_api is True if DataCite test API should be used
    """
    source_str = str(source)

    # Check for DataCite API URL first (before general URL check)
    if _is_datacite_api_url(source_str):
        doi = _extract_doi_from_datacite_url(source_str)
        use_test = _is_datacite_test_url(source_str)
        return ("datacite_api", doi, use_test)

    # Check for DOI
    if _is_doi(source_str):
        normalized_doi = _normalize_doi(source_str)
        return ("doi", normalized_doi, True)  # Default to production

    # Check for remote URL
    parsed = urlparse(source_str)
    if parsed.scheme in ("http", "https", "s3", "gs", "gcs", "abfs", "az"):
        return ("remote", source_str, False)

    # Default to local path
    return ("local", source_str, False)


# Known S3-compatible endpoints that require special handling
# Maps hostname to endpoint URL for S3 protocol conversion
_S3_COMPATIBLE_ENDPOINTS = {
    "uma1.osn.mghpcc.org": "https://uma1.osn.mghpcc.org",
}


def _convert_https_to_s3(url: str) -> tuple[str, str] | None:
    """Convert HTTPS URL to S3 protocol if it's a known S3-compatible endpoint.

    Parameters
    ----------
    url : str
        HTTPS URL to check.

    Returns
    -------
    tuple[str, str] or None
        Tuple of (s3_url, endpoint_url) if conversion is possible, None otherwise.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None

    hostname = parsed.hostname
    if hostname in _S3_COMPATIBLE_ENDPOINTS:
        # Convert path like /bucket/path to s3://bucket/path
        path = parsed.path.lstrip("/")
        s3_url = f"s3://{path}"
        endpoint_url = _S3_COMPATIBLE_ENDPOINTS[hostname]
        return (s3_url, endpoint_url)

    return None


def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    production: bool = True,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load OVRO-LWA zarr data as an xarray Dataset.

    Parameters
    ----------
    source : str or Path
        Data source, which can be one of:

        - **Local path**: Path to a local zarr store (e.g., ``"/path/to/data.zarr"``)
        - **Remote URL**: S3, HTTPS, or other remote URLs (e.g., ``"s3://bucket/data.zarr"``)
        - **DOI identifier**: A DOI string (e.g., ``"10.33569/9wsys-h7b71"`` or
          ``"doi:10.33569/9wsys-h7b71"``). The DOI will be resolved via DataCite API.
        - **DataCite API URL**: Full DataCite API URL (e.g.,
          ``"https://api.test.datacite.org/dois/10.33569/9wsys-h7b71"``).
          The API (production or test) is auto-detected from the URL.

    chunks : dict, str, or None, default "auto"
        Chunking for lazy loading. Use dict for explicit chunks (e.g., ``{"time": 100}``),
        ``"auto"`` for automatic chunking, or ``None`` to load into memory.
    production : bool, default True
        Which DataCite API to use when resolving plain DOI identifiers:

        - ``True``: Use production API (``api.datacite.org``)
        - ``False``: Use test API (``api.test.datacite.org``)

        This parameter is ignored when a full DataCite API URL is provided,
        as the API is auto-detected from the URL.
    storage_options : dict, optional
        Additional options passed to the filesystem backend (e.g., S3 credentials).
        For S3-compatible storage like OSN, you can pass::

            storage_options={
                "key": "YOUR_ACCESS_KEY",
                "secret": "YOUR_SECRET_KEY",
            }

        If not provided, credentials will be read from environment variables
        (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``) or AWS config files.
    **kwargs
        Additional arguments passed to :func:`xarray.open_zarr`.

    Returns
    -------
    xr.Dataset
        Loaded OVRO-LWA dataset with the ``radport`` accessor available.

    Raises
    ------
    DataSourceError
        If DOI resolution fails or the dataset cannot be loaded.
    FileNotFoundError
        If a local path does not exist.
    ImportError
        If required dependencies are not installed.

    Examples
    --------
    Load from a local path:

    >>> ds = ovro_lwa_portal.open_dataset("/path/to/data.zarr")

    Load from S3:

    >>> ds = ovro_lwa_portal.open_dataset("s3://bucket/data.zarr")

    Load from DOI (production DataCite):

    >>> ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")

    Load from DOI using test DataCite API:

    >>> ds = ovro_lwa_portal.open_dataset("10.33569/9wsys-h7b71", production=False)

    Load from DOI with S3 credentials:

    >>> ds = ovro_lwa_portal.open_dataset(
    ...     "10.33569/9wsys-h7b71",
    ...     production=False,
    ...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"},
    ... )

    Load from full DataCite API URL (auto-detects test vs production):

    >>> ds = ovro_lwa_portal.open_dataset(
    ...     "https://api.test.datacite.org/dois/10.33569/9wsys-h7b71"
    ... )
    """
    source_type, normalized_source, use_test_from_url = _detect_source_type(source)

    # Resolve DOI to actual data URL
    if source_type == "doi":
        try:
            normalized_source = _resolve_doi(normalized_source, production=production)
        except Exception as e:
            msg = f"Failed to resolve DOI {normalized_source}: {e}"
            raise DataSourceError(msg) from e
    elif source_type == "datacite_api":
        # For DataCite API URLs, use the API detected from the URL
        try:
            use_production = not use_test_from_url
            normalized_source = _resolve_doi(normalized_source, production=use_production)
        except Exception as e:
            msg = f"Failed to resolve DOI {normalized_source}: {e}"
            raise DataSourceError(msg) from e

    # Check if HTTPS URL needs to be converted to S3 protocol
    s3_conversion = _convert_https_to_s3(normalized_source)
    if s3_conversion:
        s3_url, endpoint_url = s3_conversion
        normalized_source = s3_url
        # Add endpoint_url to storage_options if not already set
        if storage_options is None:
            storage_options = {}
        if "endpoint_url" not in storage_options:
            storage_options["endpoint_url"] = endpoint_url
        # Disable anonymous access by default for S3-compatible endpoints
        if "anon" not in storage_options:
            storage_options["anon"] = False

    # Load zarr data
    try:
        from upath import UPath
    except ImportError as e:
        msg = (
            "universal-pathlib is required for path handling. "
            "Install with: pip install universal-pathlib"
        )
        raise ImportError(msg) from e

    try:
        # Create UPath object - works for both local and remote paths
        # Pass storage_options for remote filesystems
        if storage_options:
            store_path = UPath(normalized_source, **storage_options)
        else:
            store_path = UPath(normalized_source)

        # Explicit local existence check
        if store_path.protocol in ("", "file") and not store_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {store_path}")

        # Build a Zarr store (fsspec mapper) from the UPath
        fs = store_path.fs
        store = fs.get_mapper(store_path.path)

        # Open the zarr store
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            ds = xr.open_zarr(store, chunks=chunks, **kwargs)

    except FileNotFoundError:
        raise
    except ImportError:
        raise
    except Exception as e:
        msg = f"Failed to load dataset from {normalized_source}: {e}"
        raise DataSourceError(msg) from e

    return ds
