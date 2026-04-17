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


def test_select_reference_shape_index_deterministic():
    """Largest LM shape should be selected deterministically."""
    fits_to_zarr_xradio = _import_module()

    shapes = [(3122, 3122), (4096, 4096), (4096, 3000), (4096, 4096)]
    idx = fits_to_zarr_xradio._select_reference_shape_index(shapes)

    # Tie on (4096, 4096) should pick first occurrence.
    assert idx == 1


def test_regrid_to_reference_lm_mixed_shapes():
    """Smaller (m,l) grids interpolate onto the reference LM grid."""
    import numpy as np
    import xarray as xr

    fits_to_zarr_xradio = _import_module()

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


def test_assert_same_lm_clear_error_on_length_mismatch():
    """Length mismatch must raise RuntimeError, not a NumPy broadcast error."""
    import numpy as np

    fits_to_zarr_xradio = _import_module()

    ref = (np.linspace(-1, 1, 4096), np.linspace(-1, 1, 4096))
    cur = (np.linspace(-1, 1, 3122), np.linspace(-1, 1, 3122))
    with pytest.raises(RuntimeError, match="length mismatch"):
        fits_to_zarr_xradio._assert_same_lm(ref, cur)


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


def test_discover_groups_triple_duplicate_resolver_sees_fresh_candidates(tmp_path: Path):
    """After resolving two-way duplicate, third file must not reuse stale candidate list."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    paths = [tmp_path / f"candidate_{i}.fits" for i in range(3)]
    for p in paths:
        fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(p)

    resolver_calls: list[list[Path]] = []

    def record_first(_time_key: str, _freq_hz: float, candidates: list[Path]) -> Path:
        resolver_calls.append(list(candidates))
        return candidates[0]

    groups = mod._discover_groups(tmp_path, duplicate_resolver=record_first)

    assert groups["20240524_050009"] == [paths[0]]
    assert len(resolver_calls) == 2
    assert resolver_calls[0] == paths[:2]
    assert resolver_calls[1] == [paths[0], paths[2]]


def test_discover_groups_header_based_frequency_sorting(tmp_path: Path):
    """Groups should be deterministically sorted by header frequency."""
    mod = _import_module()
    # Write intentionally out-of-order names with opposite order frequencies.
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 8.2e7}),
    ).writeto(tmp_path / "aaa_name.fits")
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7}),
    ).writeto(tmp_path / "zzz_name.fits")

    groups = mod._discover_groups(tmp_path)
    group_files = groups["20240524_050009"]

    freqs = [mod._extract_group_metadata(p)[1] for p in group_files]
    assert freqs == [pytest.approx(4.1e7), pytest.approx(8.2e7)]


def test_discover_groups_skips_file_without_time_or_frequency_metadata(tmp_path: Path):
    """Files with no usable header or filename metadata should be skipped."""
    mod = _import_module()
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "unparseable_name.fits"
    )

    groups = mod._discover_groups(tmp_path)

    assert groups == {}


def test_discover_groups_filename_fallback_compatibility(tmp_path: Path):
    """Legacy OVRO-LWA filename pattern should still group when headers are incomplete."""
    mod = _import_module()
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    )
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "20240524_050009_82MHz_averaged_20000_iterations-I-image.fits"
    )

    groups = mod._discover_groups(tmp_path)

    assert "20240524_050009" in groups
    freqs = [mod._extract_group_metadata(p)[1] for p in groups["20240524_050009"]]
    assert freqs == [pytest.approx(4.1e7), pytest.approx(8.2e7)]
