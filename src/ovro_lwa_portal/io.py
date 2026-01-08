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
    """Resolve DOI to data URL using caltechdata_api."""
    try:
        import caltechdata_api.download_file as cda_download
    except ImportError as e:
        msg = (
            "caltechdata_api is required for DOI resolution. "
            "Install with: pip install ovro-lwa-portal[remote]"
        )
        raise ImportError(msg) from e

    try:
        url = cda_download.get_download_url(doi)
    except Exception as e:
        msg = f"Failed to resolve DOI {doi}: {e}"
        raise DataSourceError(msg) from e

    if not url:
        raise DataSourceError(f"Failed to resolve DOI {doi}: no URL returned")

    return url

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
    **kwargs: Any,
) -> xr.Dataset:
    """Load OVRO-LWA zarr data as an xarray Dataset.

    Parameters
    ----------
    source : str or Path
        Local path, remote URL (s3://, https://, etc.), or DOI identifier.
    chunks : dict, str, or None, default "auto"
        Chunking for lazy loading. Use dict for explicit chunks (e.g., {"time": 100}),
        "auto" for automatic chunking, or None to load into memory.
    **kwargs
        Additional arguments passed to xr.open_zarr.

    Returns
    -------
    xr.Dataset
        Loaded OVRO-LWA dataset.

    Examples
    --------
    >>> ds = ovro_lwa_portal.open_dataset("/path/to/data.zarr")
    >>> ds = ovro_lwa_portal.open_dataset("s3://bucket/data.zarr")
    >>> ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
    """
    source_type, normalized_source = _detect_source_type(source)

    # Resolve DOI to actual data URL
    if source_type == "doi":
        try:
            normalized_source = _resolve_doi(normalized_source)
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
