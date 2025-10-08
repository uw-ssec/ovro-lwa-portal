"""FITS to Zarr conversion utilities for OVRO-LWA data.

This module provides functionality to convert FITS files from OVRO-LWA
observations into Zarr format for efficient storage and access.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    import xarray as xr

logger = logging.getLogger(__name__)


def discover_fits_files(pattern: str) -> list[str]:
    """Discover FITS files matching the specified pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern for finding FITS files

    Returns
    -------
    list[str]
        Sorted list of FITS file paths

    Raises
    ------
    FileNotFoundError
        If no FITS files are found
    """
    fits_files = sorted(glob.glob(pattern))

    if not fits_files:
        msg = f"No FITS files found matching pattern: {pattern}"
        raise FileNotFoundError(msg)

    logger.info(f"Discovered {len(fits_files)} FITS files")
    return fits_files


def parse_filename_metadata(filename: str, timestamp_parts: int = 2) -> tuple[str, str]:
    """Extract timestamp and frequency from FITS filename.

    Parameters
    ----------
    filename : str
        Path to FITS file
    timestamp_parts : int, optional
        Number of parts to use for timestamp, by default 2

    Returns
    -------
    tuple[str, str]
        Tuple of (timestamp, frequency) strings

    Raises
    ------
    ValueError
        If filename doesn't match expected format
    """
    try:
        parts = Path(filename).stem.split("_")

        if len(parts) < 3:  # Need at least timestamp parts and frequency
            msg = f"Insufficient parts in filename: {filename}"
            raise ValueError(msg)

        # Extract timestamp (first two parts: YYYYMMDD_HHMMSS)
        timestamp = "_".join(parts[:timestamp_parts])

        # Extract frequency (third part)
        frequency = parts[2]

        return timestamp, frequency

    except Exception as e:
        msg = f"Failed to parse filename {filename}: {e}"
        raise ValueError(msg) from e


def load_zarr_dataset(zarr_path: str) -> xr.Dataset:  # type: ignore[name-defined]
    """Load a Zarr dataset from disk.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr store

    Returns
    -------
    xarray.Dataset
        The loaded dataset
    """
    import xarray as xr

    return xr.open_zarr(zarr_path)


__all__ = ["discover_fits_files", "parse_filename_metadata", "load_zarr_dataset"]
