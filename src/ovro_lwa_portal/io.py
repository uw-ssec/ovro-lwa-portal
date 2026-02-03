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

        - **Local path**: e.g., ``"/path/to/data.zarr"``
        - **Remote URL**: e.g., ``"s3://bucket/data.zarr"``, ``"https://..."``
        - **DOI identifier**: e.g., ``"10.33569/9wsys-h7b71"`` or
          ``"doi:10.33569/9wsys-h7b71"``
    chunks : dict, str, or None, default "auto"
        Chunking for lazy loading. Use dict for explicit chunks (e.g., {"time": 100}),
        "auto" for automatic chunking, or None to load into memory.
    production : bool, default True
        Which DataCite API to use when resolving DOI identifiers:

        - ``True``: production API (``api.datacite.org``)
        - ``False``: test API (``api.test.datacite.org``)
    storage_options : dict, optional
        Options passed to the filesystem backend (e.g., S3 credentials)::

            storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
    **kwargs
        Additional arguments passed to :func:`xarray.open_zarr`.

    Returns
    -------
    xr.Dataset
        Loaded OVRO-LWA dataset.

    Examples
    --------
    >>> ds = ovro_lwa_portal.open_dataset("/path/to/data.zarr")
    >>> ds = ovro_lwa_portal.open_dataset("s3://bucket/data.zarr")
    >>> ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
    >>> ds = ovro_lwa_portal.open_dataset(
    ...     "10.33569/9wsys-h7b71",
    ...     production=False,
    ...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"},
    ... )
    """
    source_type, normalized_source = _detect_source_type(source)

    # Resolve DOI to actual data URL
    if source_type == "doi":
        try:
            normalized_source = _resolve_doi(normalized_source, production=production)
        except Exception as e:
            msg = f"Failed to resolve DOI {normalized_source}: {e}"
            raise DataSourceError(msg) from e

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
        # Pass storage_options so credentials reach the filesystem backend
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
