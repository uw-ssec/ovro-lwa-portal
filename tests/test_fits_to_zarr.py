"""Tests for FITS to Zarr conversion utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from astropy.io import fits


def _import_module():
    """Import module under test if optional dependencies are available."""
    try:
        from ovro_lwa_portal import fits_to_zarr_xradio
    except ImportError as e:
        pytest.skip(f"xradio dependencies not available: {e}")
    return fits_to_zarr_xradio


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
    fits_to_zarr_xradio = _import_module()
    assert hasattr(fits_to_zarr_xradio, "convert_fits_dir_to_zarr")
    assert hasattr(fits_to_zarr_xradio, "PAT")
    assert hasattr(fits_to_zarr_xradio, "MHZ_RE")


def test_extract_group_metadata_from_header(tmp_path: Path):
    """Header metadata should provide both time and frequency."""
    mod = _import_module()
    fpath = tmp_path / "arbitrary_name.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key == "20240524_050009"
    assert frequency_hz == pytest.approx(4.1e7)
    assert notes == []


def test_extract_group_metadata_fallback_to_filename(tmp_path: Path):
    """Filename fallback should be used when headers are missing metadata."""
    mod = _import_module()
    fpath = tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key == "20240524_050009"
    assert frequency_hz == pytest.approx(4.1e7)
    assert "time-from-filename" in notes
    assert "frequency-from-filename" in notes


def test_discover_groups_duplicate_without_resolver_raises(tmp_path: Path):
    """Duplicate time/frequency files should raise without a resolver."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(tmp_path / "first_name.fits")
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(tmp_path / "second_name.fits")

    with pytest.raises(RuntimeError, match="Duplicate FITS files detected"):
        mod._discover_groups(tmp_path)


def test_discover_groups_duplicate_with_resolver_selects_one(tmp_path: Path):
    """Resolver should choose one candidate for duplicate time/frequency groups."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    f1 = tmp_path / "candidate_a.fits"
    f2 = tmp_path / "candidate_b.fits"
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f1)
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f2)

    def choose_second(_time_key: str, _freq_hz: float, candidates: list[Path]) -> Path:
        return candidates[-1]

    groups = mod._discover_groups(tmp_path, duplicate_resolver=choose_second)

    assert "20240524_050009" in groups
    assert groups["20240524_050009"] == [f2]
