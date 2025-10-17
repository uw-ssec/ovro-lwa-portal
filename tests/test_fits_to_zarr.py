"""Tests for FITS to Zarr conversion utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def test_parse_filename_pattern():
    """Test parsing metadata from FITS filenames using PAT regex."""
    # Import the pattern directly to avoid loading xradio dependencies
    import sys
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fits_to_zarr_xradio",
        Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    )
    if spec is None or spec.loader is None:
        pytest.skip("Cannot load fits_to_zarr_xradio module")

    # Read the pattern from the file directly
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()

    # Extract the PAT regex pattern
    pat_match = re.search(r'PAT = re\.compile\(\s*r"([^"]+)"\s*\)', content)
    assert pat_match is not None, "PAT pattern not found in module"

    PAT = re.compile(pat_match.group(1))

    filename = "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    match = PAT.match(filename)

    assert match is not None
    assert match.group("date") == "20240524"
    assert match.group("hms") == "050009"
    assert match.group("sb") == "41"


def test_parse_filename_pattern_with_fixed():
    """Test parsing metadata from fixed FITS filenames."""
    # Extract pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    pat_match = re.search(r'PAT = re\.compile\(\s*r"([^"]+)"\s*\)', content)
    assert pat_match is not None
    PAT = re.compile(pat_match.group(1))

    filename = "20240524_050009_41MHz_averaged_20000_iterations-I-image_fixed.fits"
    match = PAT.match(filename)

    assert match is not None
    assert match.group("date") == "20240524"
    assert match.group("hms") == "050009"
    assert match.group("sb") == "41"


def test_parse_filename_pattern_invalid():
    """Test parsing metadata from invalid filenames."""
    # Extract pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    pat_match = re.search(r'PAT = re\.compile\(\s*r"([^"]+)"\s*\)', content)
    assert pat_match is not None
    PAT = re.compile(pat_match.group(1))

    filename = "invalid.fits"
    match = PAT.match(filename)

    assert match is None


def test_mhz_from_name():
    """Test extracting MHz from filename."""
    # Extract MHZ_RE pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    mhz_match = re.search(r'MHZ_RE = re\.compile\(r"([^"]+)"\)', content)
    assert mhz_match is not None
    MHZ_RE = re.compile(mhz_match.group(1))

    # Replicate _mhz_from_name logic
    path = Path("20240524_050009_41MHz_averaged_20000_iterations-I-image.fits")
    m = MHZ_RE.search(path.name)
    mhz = int(m.group(1)) if m else 10**9

    assert mhz == 41


def test_mhz_from_name_multiple_digits():
    """Test extracting MHz from filename with larger frequency."""
    # Extract MHZ_RE pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    mhz_match = re.search(r'MHZ_RE = re\.compile\(r"([^"]+)"\)', content)
    assert mhz_match is not None
    MHZ_RE = re.compile(mhz_match.group(1))

    path = Path("20240524_050009_82MHz_averaged_20000_iterations-I-image.fits")
    m = MHZ_RE.search(path.name)
    mhz = int(m.group(1)) if m else 10**9

    assert mhz == 82


def test_mhz_from_name_no_match():
    """Test extracting MHz from filename without frequency."""
    # Extract MHZ_RE pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    mhz_match = re.search(r'MHZ_RE = re\.compile\(r"([^"]+)"\)', content)
    assert mhz_match is not None
    MHZ_RE = re.compile(mhz_match.group(1))

    path = Path("invalid.fits")
    m = MHZ_RE.search(path.name)
    mhz = int(m.group(1)) if m else 10**9

    # Should return sentinel value
    assert mhz == 10**9


def test_module_can_be_imported():
    """Test that the fits_to_zarr_xradio module can be imported."""
    try:
        from ovro_lwa_portal import fits_to_zarr_xradio
        assert hasattr(fits_to_zarr_xradio, "convert_fits_dir_to_zarr")
        assert hasattr(fits_to_zarr_xradio, "PAT")
        assert hasattr(fits_to_zarr_xradio, "MHZ_RE")
    except ImportError as e:
        pytest.skip(f"xradio dependencies not available: {e}")
