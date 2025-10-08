"""Tests for FITS to Zarr conversion utilities."""

from __future__ import annotations

import pytest

from ovro_lwa_portal import fits_to_zarr


def test_parse_filename_metadata():
    """Test parsing metadata from FITS filenames."""
    filename = "20240101_120000_74MHz_I.fits"
    timestamp, frequency = fits_to_zarr.parse_filename_metadata(filename)

    assert timestamp == "20240101_120000"
    assert frequency == "74MHz"


def test_parse_filename_metadata_with_path():
    """Test parsing metadata from FITS filenames with full paths."""
    filename = "/path/to/data/20240101_120000_74MHz_I.fits"
    timestamp, frequency = fits_to_zarr.parse_filename_metadata(filename)

    assert timestamp == "20240101_120000"
    assert frequency == "74MHz"


def test_parse_filename_metadata_invalid():
    """Test parsing metadata from invalid filenames."""
    with pytest.raises(ValueError, match="Insufficient parts"):
        fits_to_zarr.parse_filename_metadata("invalid.fits")


def test_discover_fits_files_not_found():
    """Test discovering FITS files when none exist."""
    with pytest.raises(FileNotFoundError, match="No FITS files found"):
        fits_to_zarr.discover_fits_files("/nonexistent/path/*.fits")
