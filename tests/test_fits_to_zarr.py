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


def test_select_reference_shape_index_deterministic():
    """Largest LM shape should be selected deterministically."""
    try:
        from ovro_lwa_portal import fits_to_zarr_xradio
    except ImportError as e:
        pytest.skip(f"xradio dependencies not available: {e}")

    shapes = [(3122, 3122), (4096, 4096), (4096, 3000), (4096, 4096)]
    idx = fits_to_zarr_xradio._select_reference_shape_index(shapes)

    # Tie on (4096, 4096) should pick first occurrence.
    assert idx == 1


def test_regrid_to_reference_lm_mixed_shapes():
    """Smaller (m,l) grids interpolate onto the reference LM grid."""
    try:
        import numpy as np
        import xarray as xr

        from ovro_lwa_portal import fits_to_zarr_xradio
    except ImportError as e:
        pytest.skip(f"xradio dependencies not available: {e}")

    l_ref = np.linspace(-1.0, 1.0, 6)
    m_ref = np.linspace(-1.0, 1.0, 5)
    rng = np.random.default_rng(0)
    sky_ref = rng.standard_normal((5, 6))
    hdr_ref = "SIMPLE  =                   T\nNAXIS   =                    2"
    xds_ref = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_ref)},
        coords={
            "l": ("l", l_ref),
            "m": ("m", m_ref),
            "right_ascension": (("m", "l"), np.full((5, 6), 180.0)),
            "declination": (("m", "l"), np.full((5, 6), 45.0)),
        },
        attrs={"fits_wcs_header": hdr_ref},
    )
    xds_ref = xds_ref.assign(wcs_header_str=((), np.bytes_(hdr_ref.encode("utf-8"))))

    l_sm = np.linspace(-0.5, 0.5, 4)
    m_sm = np.linspace(-0.5, 0.5, 3)
    sky_sm = rng.standard_normal((3, 4))
    hdr_sm = "SIMPLE  =                   T\nNAXIS   =                    2\nSMALL=T"
    xds_sm = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_sm)},
        coords={
            "l": ("l", l_sm),
            "m": ("m", m_sm),
            "right_ascension": (("m", "l"), np.zeros((3, 4))),
            "declination": (("m", "l"), np.zeros((3, 4))),
        },
        attrs={"fits_wcs_header": hdr_sm},
    )
    xds_sm = xds_sm.assign(wcs_header_str=((), np.bytes_(hdr_sm.encode("utf-8"))))

    out = fits_to_zarr_xradio._regrid_to_reference_lm(xds_sm, xds_ref)

    assert out.sizes["m"] == 5
    assert out.sizes["l"] == 6
    np.testing.assert_allclose(out["l"].values, l_ref)
    np.testing.assert_allclose(out["m"].values, m_ref)
    assert out.attrs["fits_wcs_header"] == hdr_ref
    assert out["SKY"].attrs["fits_wcs_header"] == hdr_ref
    np.testing.assert_allclose(out["right_ascension"].values, xds_ref["right_ascension"].values)
    np.testing.assert_allclose(out["declination"].values, xds_ref["declination"].values)
    assert bytes(out["wcs_header_str"].values.item()) == hdr_ref.encode("utf-8")
