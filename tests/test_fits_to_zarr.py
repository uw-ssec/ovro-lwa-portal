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


def test_peek_lm_shape_reads_dimensions_from_header(tmp_path: Path):
    """Peeked LM shape should match NAXIS2/NAXIS1 from FITS header."""
    import numpy as np

    mod = _import_module()
    fpath = tmp_path / "lm_shape_from_header.fits"
    fits.PrimaryHDU(data=np.zeros((5, 7), dtype=np.float32)).writeto(fpath)

    assert mod._peek_lm_shape(fpath) == (5, 7)


def test_peek_lm_shape_errors_when_naxis_less_than_two(tmp_path: Path):
    """Peeking LM shape should fail clearly for non-image FITS data."""
    import numpy as np

    mod = _import_module()
    fpath = tmp_path / "not_2d.fits"
    fits.PrimaryHDU(data=np.zeros(8, dtype=np.float32)).writeto(fpath)

    with pytest.raises(RuntimeError, match="NAXIS=1"):
        mod._peek_lm_shape(fpath)


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


def test_regrid_to_reference_lm_requires_l_m_coords():
    """Regridding must fail clearly when ``l``/``m`` coordinates are absent."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    l_ref = np.linspace(-1.0, 1.0, 4)
    m_ref = np.linspace(-1.0, 1.0, 4)
    sky_ref = np.zeros((4, 4))
    xds_ref = xr.Dataset(
        {"SKY": (("m", "l"), sky_ref)},
        coords={
            "l": ("l", l_ref),
            "m": ("m", m_ref),
            "right_ascension": (("m", "l"), np.zeros((4, 4))),
            "declination": (("m", "l"), np.zeros((4, 4))),
        },
        attrs={"fits_wcs_header": "SIMPLE  =                   T"},
    )
    xds_no_coords = xr.Dataset({"SKY": (("m", "l"), np.zeros((2, 3)))})

    with pytest.raises(RuntimeError, match="missing"):
        mod._regrid_to_reference_lm(xds_no_coords, xds_ref)


def test_regrid_to_reference_lm_error_includes_source_label():
    """Interpolation failures should name the source file when provided."""
    from unittest.mock import patch

    import numpy as np
    import xarray as xr

    mod = _import_module()
    rng = np.random.default_rng(2)
    l_ref = np.linspace(-1.0, 1.0, 5)
    m_ref = np.linspace(-1.0, 1.0, 5)
    sky_ref = rng.standard_normal((5, 5))
    hdr_ref = "SIMPLE  =                   T\nNAXIS   =                    2"
    xds_ref = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_ref)},
        coords={
            "l": ("l", l_ref),
            "m": ("m", m_ref),
            "right_ascension": (("m", "l"), np.full((5, 5), 180.0)),
            "declination": (("m", "l"), np.full((5, 5), 45.0)),
        },
        attrs={"fits_wcs_header": hdr_ref},
    )
    l_sm = np.linspace(-0.5, 0.5, 4)
    m_sm = np.linspace(-0.5, 0.5, 3)
    sky_sm = rng.standard_normal((3, 4))
    xds_sm = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_sm)},
        coords={
            "l": ("l", l_sm),
            "m": ("m", m_sm),
            "right_ascension": (("m", "l"), np.zeros((3, 4))),
            "declination": (("m", "l"), np.zeros((3, 4))),
        },
        attrs={"fits_wcs_header": hdr_ref},
    )

    with (
        patch.object(xr.Dataset, "interp", side_effect=ValueError("simulated interp failure")),
        pytest.raises(RuntimeError, match="bad_file.fits"),
    ):
        mod._regrid_to_reference_lm(xds_sm, xds_ref, source_label="bad_file.fits")


def test_load_global_lm_reference_selects_largest_shape(monkeypatch, tmp_path: Path):
    """Global reference must load the FITS whose LM shape wins the max-shape rule."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    f_small = tmp_path / "small.fits"
    f_large = tmp_path / "large.fits"
    f_small.touch()
    f_large.touch()
    by_time = {"20240101_120000": [f_small, f_large]}

    monkeypatch.setattr(mod, "fix_fits_headers", lambda files, fd, skip_existing=True: list(files))
    monkeypatch.setattr(
        mod,
        "_peek_lm_shape",
        lambda fp: (32, 32) if fp.name.startswith("small") else (64, 64),
    )

    loaded: list[Path] = []

    def fake_load(fp: Path, chunk_lm: int = 1024) -> xr.Dataset:
        loaded.append(fp)
        n = 64
        l_ = np.linspace(-1.0, 1.0, n)
        m_ = np.linspace(-1.0, 1.0, n)
        sky = np.zeros((n, n))
        hdr = "SIMPLE  =                   T\nNAXIS   =                    2"
        return (
            xr.Dataset(
                {"SKY": (("m", "l"), sky)},
                coords={
                    "l": ("l", l_),
                    "m": ("m", m_),
                    "frequency": ("frequency", np.array([1.4e8])),
                    "right_ascension": (("m", "l"), np.full((n, n), 180.0)),
                    "declination": (("m", "l"), np.full((n, n), 45.0)),
                },
                attrs={"fits_wcs_header": hdr},
            )
            .assign(wcs_header_str=((), np.bytes_(hdr.encode("utf-8"))))
        )

    monkeypatch.setattr(mod, "_load_for_combine", fake_load)

    out = mod._load_global_lm_reference_dataset(
        by_time,
        tmp_path / "fixed",
        chunk_lm=0,
        fix_headers_on_demand=True,
    )

    assert loaded == [f_large]
    assert int(out.sizes["m"]) == 64
    assert int(out.sizes["l"]) == 64


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


def test_discover_groups_duplicate_without_resolver_keeps_first(tmp_path: Path):
    """Same time + same 10 kHz bin: keep the first file and warn; do not stack duplicates."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    f1 = tmp_path / "first_name.fits"
    f2 = tmp_path / "second_name.fits"
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f1)
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f2)

    groups = mod._discover_groups(tmp_path)
    assert groups["20240524_050009"] == [f1]


def test_discover_groups_header_frequency_jitter_single_plane(tmp_path: Path):
    """RESTFREQ differing by <<10 kHz should map to one binned subband (one FITS kept)."""
    mod = _import_module()
    t = "2024-05-24T05:00:09.0"
    f1 = tmp_path / "a.fits"
    f2 = tmp_path / "b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": t, "RESTFREQ": 4.1e7}),
    ).writeto(f1)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": t, "RESTFREQ": 4.1e7 + 100.0}),
    ).writeto(f2)
    groups = mod._discover_groups(tmp_path)
    assert groups["20240524_050009"] == [f1]


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


def test_rechunk_lm_for_zarr_uniform_spatial_chunks():
    """Irregular dask chunks along l/m must become uniform for Zarr compatibility."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    arr = np.random.default_rng(0).random((512, 512))
    data = da.from_array(arr, chunks=((256, 256), (256, 128, 128)))
    l = np.linspace(-1.0, 1.0, 512)
    m = np.linspace(-1.0, 1.0, 512)
    ds = xr.Dataset(
        {"SKY": (("m", "l"), data)},
        coords={"l": ("l", l), "m": ("m", m)},
    )
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=256)
    chunks = out["SKY"].data.chunks
    assert chunks[0] == (256, 256)
    assert chunks[1] == (256, 256)


def test_rechunk_lm_for_zarr_chunk_lm_zero_single_spatial_chunk():
    """chunk_lm=0 should use one chunk per spatial axis (still uniform)."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    arr = np.random.default_rng(1).random((100, 100))
    data = da.from_array(arr, chunks=((50, 50), (50, 50)))
    l = np.linspace(-1.0, 1.0, 100)
    m = np.linspace(-1.0, 1.0, 100)
    ds = xr.Dataset({"SKY": (("m", "l"), data)}, coords={"l": ("l", l), "m": ("m", m)})
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=0)
    assert out["SKY"].data.chunks == ((100,), (100,))


def test_rechunk_lm_for_zarr_fixes_irregular_wcs_header_str_chunks(tmp_path):
    """Non-uniform dask chunks on aux vars (e.g. wcs along frequency) must be Zarr-safe."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    n = 4
    l = np.linspace(-1.0, 1.0, 32)
    m = np.linspace(-1.0, 1.0, 32)
    sky = da.random.random((32, 32), chunks=(16, 16))
    hdr = np.array([np.bytes_(b"x" * 20) for _ in range(n)], dtype=np.bytes_)
    w = da.from_array(hdr, chunks=((2, 1, 1),))
    ds = xr.Dataset(
        {"SKY": (("m", "l"), sky), "wcs_header_str": (("frequency",), w)},
        coords={
            "l": ("l", l),
            "m": ("m", m),
            "frequency": np.arange(n, dtype=np.float64),
        },
    )
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=8)
    assert out["wcs_header_str"].data.chunks == ((4,),)
    out.to_zarr(tmp_path / "t.zarr", mode="w", consolidated=False)


def test_rechunk_lm_for_zarr_strips_coord_encoding_conflicts(tmp_path):
    """Stale ``encoding['chunks']`` on coords must not break ``to_zarr`` (Dask vs Zarr grid)."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nf, ny, nx = 2, 32, 32
    sky = da.random.random((nf, ny, nx), chunks=(1, 16, 16))
    ra = da.random.random((nf, ny, nx), chunks=(1, 16, 16))
    dec = da.random.random((nf, ny, nx), chunks=(1, 16, 16))
    ds = xr.Dataset(
        {"SKY": (("frequency", "m", "l"), sky)},
        coords={
            "frequency": np.arange(nf),
            "l": np.linspace(-1, 1, nx),
            "m": np.linspace(-1, 1, ny),
            "right_ascension": (("frequency", "m", "l"), ra),
            "declination": (("frequency", "m", "l"), dec),
        },
    )
    ds["right_ascension"].encoding = {"chunks": (2, 128, 128)}
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=8)
    assert out["right_ascension"].encoding == {}
    assert out["declination"].encoding == {}
    out.to_zarr(tmp_path / "coord.zarr", mode="w", consolidated=False)


def test_rechunk_lm_for_zarr_fixes_nonuniform_coord_time_chunks(tmp_path):
    """Coords with time chunks like (2,1,1) should be rechunked to Zarr-safe layout."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nt, ny, nx = 4, 32, 32
    sky = da.random.random((nt, ny, nx), chunks=(1, 16, 16))
    # Deliberately non-uniform time chunks to match runtime failure.
    ra_np = np.random.default_rng(2).random((nt, ny, nx))
    dec_np = np.random.default_rng(3).random((nt, ny, nx))
    ra = da.from_array(ra_np, chunks=((2, 1, 1), (16, 16), (16, 16)))
    dec = da.from_array(dec_np, chunks=((2, 1, 1), (16, 16), (16, 16)))

    ds = xr.Dataset(
        {"SKY": (("time", "m", "l"), sky)},
        coords={
            "time": np.arange(nt),
            "l": np.linspace(-1, 1, nx),
            "m": np.linspace(-1, 1, ny),
            "right_ascension": (("time", "m", "l"), ra),
            "declination": (("time", "m", "l"), dec),
        },
    )
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=8)
    assert hasattr(out["right_ascension"].data, "chunks")
    assert out["right_ascension"].data.chunks[0] == (4,)
    out.to_zarr(tmp_path / "coord_nonuniform_time.zarr", mode="w", consolidated=False)


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


def test_fix_headers_adds_stokes_axis_when_missing(tmp_path: Path):
    """Header fixing should add a singleton STOKES axis for 3D FREQ-only cubes."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "input.fits"
    out_path = tmp_path / "output_fixed.fits"

    data = np.zeros((1, 4, 4), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 3,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "NAXIS3": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CRVAL3": 4.1e7,
            "CRPIX3": 1.0,
            "CDELT3": 1.0,
            "CUNIT3": "Hz",
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header
        out_data = hdul[0].data

    assert hdr["NAXIS"] == 4
    assert hdr["CTYPE4"] == "STOKES"
    assert hdr["CRVAL4"] == pytest.approx(1.0)
    assert hdr["CRPIX4"] == pytest.approx(1.0)
    assert hdr["CDELT4"] == pytest.approx(1.0)
    assert out_data is not None
    assert out_data.shape == (1, 1, 4, 4)


def test_normalize_time_key_from_datetime64():
    """Datetime64 values should normalize to discovery-style time keys."""
    mod = _import_module()
    value = mod.np.datetime64("2024-12-18T06:33:36.987654321")

    out = mod._normalize_time_key(value)

    assert out == "20241218_063336"


def test_normalize_time_key_from_mjd_float():
    """Numeric MJD time coordinates should normalize to discovery-style keys."""
    mod = _import_module()
    # 2024-12-20T03:00:00 UTC in MJD.
    out = mod._normalize_time_key(60664.125)

    assert out == "20241220_030000"


def test_existing_time_keys_from_zarr(tmp_path: Path):
    """Existing Zarr time coordinates should map to a set of normalized keys."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_zarr = tmp_path / "existing.zarr"
    ds = xr.Dataset(
        {"SKY": (("time", "m", "l"), np.zeros((2, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2024-12-18T06:33:36", "2024-12-18T06:33:37"], dtype="datetime64[ns]"),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
        },
    )
    ds.to_zarr(out_zarr, mode="w", consolidated=False)

    keys = mod._existing_time_keys_from_zarr(out_zarr)

    assert keys == {"20241218_063336", "20241218_063337"}


def test_existing_time_keys_from_zarr_missing_time_raises(tmp_path: Path):
    """Resume helper should fail clearly when existing Zarr has no time coordinate."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_zarr = tmp_path / "no_time.zarr"
    ds = xr.Dataset(
        {"SKY": (("m", "l"), np.zeros((2, 2), dtype=np.float32))},
        coords={"m": np.array([0.0, 1.0]), "l": np.array([0.0, 1.0])},
    )
    ds.to_zarr(out_zarr, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match="has no 'time' coordinate"):
        mod._existing_time_keys_from_zarr(out_zarr)


def test_reindex_time_step_to_expected_frequencies_fills_missing_with_nan():
    """Per-time datasets should be expanded to the expected subband axis."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    xds_t = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "m", "l"),
                np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
            )
        },
        coords={
            "time": np.array(["2024-12-18T06:33:36"], dtype="datetime64[s]"),
            "frequency": np.array([41_000_000.0, 55_000_000.0]),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
        },
    )

    out = mod._reindex_time_step_to_expected_frequencies(
        xds_t,
        [41_000_000.0, 48_000_000.0, 55_000_000.0],
    )

    assert out.sizes["frequency"] == 3
    assert np.allclose(out["frequency"].values, [41_000_000.0, 48_000_000.0, 55_000_000.0])
    # Added frequency plane is all NaN in data variables.
    assert np.isnan(out["SKY"].isel(frequency=1).values).all()


def test_convert_resume_skips_already_ingested_times(monkeypatch, tmp_path: Path):
    """Resume mode should only process discovered timesteps missing from output Zarr."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_zarr = out_dir / "ovro_lwa_full_lm_only.zarr"
    out_zarr.mkdir()

    f1 = tmp_path / "a.fits"
    f2 = tmp_path / "b.fits"
    f1.touch()
    f2.touch()

    by_time = {"20241218_063336": [f1], "20241218_063337": [f2]}
    monkeypatch.setattr(mod, "_discover_groups", lambda *_args, **_kwargs: by_time)
    monkeypatch.setattr(mod, "_existing_time_keys_from_zarr", lambda _p: {"20241218_063336"})

    ref = xr.Dataset(coords={"l": ("l", np.array([0.0, 1.0])), "m": ("m", np.array([0.0, 1.0]))})
    monkeypatch.setattr(mod, "_load_global_lm_reference_dataset", lambda *_args, **_kwargs: ref)

    xds_t = xr.Dataset(
        {"SKY": (("time", "m", "l"), np.zeros((1, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2024-12-18T06:33:37"], dtype="datetime64[s]"),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
            "frequency": np.array([4.1e7]),
        },
    )
    monkeypatch.setattr(mod, "_combine_time_step", lambda *_args, **_kwargs: (xds_t, [4.1e7], []))

    write_calls: list[bool] = []
    monkeypatch.setattr(
        mod,
        "_write_or_append_zarr",
        lambda _xds, _out, *, first_write, chunk_lm: write_calls.append(first_write),
    )

    result = mod.convert_fits_dir_to_zarr(
        input_dir=tmp_path,
        out_dir=out_dir,
        resume=True,
        rebuild=False,
    )

    assert result == out_zarr
    assert len(write_calls) == 1
    assert write_calls == [False]


def test_convert_resume_returns_early_when_no_pending(monkeypatch, tmp_path: Path):
    """Resume mode should exit without combine/write when all times already exist."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_zarr = out_dir / "ovro_lwa_full_lm_only.zarr"
    out_zarr.mkdir()

    f1 = tmp_path / "a.fits"
    f1.touch()

    by_time = {"20241218_063336": [f1]}
    monkeypatch.setattr(mod, "_discover_groups", lambda *_args, **_kwargs: by_time)
    monkeypatch.setattr(mod, "_existing_time_keys_from_zarr", lambda _p: {"20241218_063336"})

    ref = xr.Dataset(coords={"l": ("l", np.array([0.0, 1.0])), "m": ("m", np.array([0.0, 1.0]))})
    monkeypatch.setattr(mod, "_load_global_lm_reference_dataset", lambda *_args, **_kwargs: ref)

    combine_calls: list[bool] = []
    monkeypatch.setattr(
        mod,
        "_combine_time_step",
        lambda *_args, **_kwargs: combine_calls.append(True),  # pragma: no cover
    )
    write_calls: list[bool] = []
    monkeypatch.setattr(
        mod,
        "_write_or_append_zarr",
        lambda *_args, **_kwargs: write_calls.append(True),  # pragma: no cover
    )

    result = mod.convert_fits_dir_to_zarr(
        input_dir=tmp_path,
        out_dir=out_dir,
        resume=True,
        rebuild=False,
    )

    assert result == out_zarr
    assert combine_calls == []
    assert write_calls == []
