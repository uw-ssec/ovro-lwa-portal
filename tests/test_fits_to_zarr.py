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


def test_time_does_not_fall_back_to_filename(tmp_path: Path):
    """Without ``-image-YYYYMMDD_HHMMSS`` or DATE-OBS, time key stays unknown."""
    mod = _import_module()
    fpath = tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(fpath)
    time_key, _, notes = mod._extract_group_metadata(fpath)
    assert time_key is None
    assert "time-from-filename" not in notes


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


def test_extract_group_metadata_filename_time_overrides_header(tmp_path: Path) -> None:
    """Basename ``-image-`` stamp is the default time key and overrides DATE-OBS."""
    mod = _import_module()
    fpath = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_x.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T00:00:00", "RESTFREQ": 18e6}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key == "20241221_102109"
    assert frequency_hz == pytest.approx(18e6)
    assert notes == []

    time_header, _, _ = mod._extract_group_metadata(fpath, time_key_source="header")
    assert time_header == "20241221_000000"


def test_discover_groups_filename_time_merges_same_image_id(tmp_path: Path) -> None:
    """Same ``-image-YYYYMMDD_HHMMSS`` basename groups together even if DATE-OBS differs."""
    mod = _import_module()
    a = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits"
    b = tmp_path / "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T01:00:00", "RESTFREQ": 18e6}),
    ).writeto(a)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-22T23:00:00", "RESTFREQ": 73e6}),
    ).writeto(b)

    by_header = mod._discover_groups(tmp_path, time_key_source="header")
    by_name = mod._discover_groups(tmp_path)

    assert len(by_header) == 2
    assert len(by_name) == 1
    assert list(by_name.keys()) == ["20241221_102109"]
    assert {p.name for p in by_name["20241221_102109"]} == {a.name, b.name}


def test_extract_group_metadata_requires_date_obs(tmp_path: Path):
    """OVRO-style name without ``-image-`` stamp needs DATE-OBS for the time key."""
    mod = _import_module()
    fpath = tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key is None
    assert frequency_hz == pytest.approx(4.1e7)
    assert "frequency-from-filename" in notes


def test_extract_group_metadata_filename_time_overrides_header(tmp_path: Path) -> None:
    """``time_key_source='filename'`` prefers ``-image-`` basename over DATE-OBS."""
    mod = _import_module()
    fpath = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_x.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T00:00:00", "RESTFREQ": 18e6}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath, time_key_source="filename")

    assert time_key == "20241221_102109"
    assert frequency_hz == pytest.approx(18e6)
    assert "time-from-filename" in notes


def test_discover_groups_filename_time_merges_same_image_id(tmp_path: Path) -> None:
    """Same ``-image-YYYYMMDD_HHMMSS`` basename groups together even if DATE-OBS differs."""
    mod = _import_module()
    a = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits"
    b = tmp_path / "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T01:00:00", "RESTFREQ": 18e6}),
    ).writeto(a)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-22T23:00:00", "RESTFREQ": 73e6}),
    ).writeto(b)

    by_header = mod._discover_groups(tmp_path, time_key_source="header")
    by_name = mod._discover_groups(tmp_path, time_key_source="filename")

    assert len(by_header) == 2
    assert len(by_name) == 1
    assert list(by_name.keys()) == ["20241221_102109"]
    assert {p.name for p in by_name["20241221_102109"]} == {a.name, b.name}


def test_discover_groups_duplicate_without_resolver_keeps_first(tmp_path: Path):
    """Same time + same discovery frequency bin: keep the first file and warn; do not stack duplicates."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    f1 = tmp_path / "first_name.fits"
    f2 = tmp_path / "second_name.fits"
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f1)
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f2)

    groups = mod._discover_groups(tmp_path)
    assert groups["20240524_050009"] == [f1]


def test_discover_groups_header_frequency_jitter_single_plane(tmp_path: Path):
    """RESTFREQ differing by <<23 kHz should map to one binned subband (one FITS kept)."""
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


def test_discover_groups_freq_bin_hz_controls_merge_window(tmp_path: Path):
    """Narrow bin (10 kHz) splits ~15 kHz RESTFREQ offset; 23 kHz bin merges as one subband."""
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
        header=fits.Header({"DATE-OBS": t, "RESTFREQ": 4.1e7 + 15_000.0}),
    ).writeto(f2)

    merged = mod._discover_groups(tmp_path, freq_bin_hz=23_000.0)
    assert merged["20240524_050009"] == [f1]

    split = mod._discover_groups(tmp_path, freq_bin_hz=10_000.0)
    assert len(split["20240524_050009"]) == 2


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
    """Files missing DATE-OBS and the ``-image-`` time stamp should be skipped."""
    mod = _import_module()
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    )
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "20240524_050009_82MHz_averaged_20000_iterations-I-image.fits"
    )

    groups = mod._discover_groups(tmp_path)
    assert groups == {}


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


def test_fix_headers_sets_crval_to_fk5_zenith_from_filename_image_stamp(tmp_path: Path):
    """_fix_headers sets CRVAL1/2 from FK5 zenith at ``-image-YYYYMMDD_HHMMSS`` in the basename."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241218_030201-test.fits"
    out_path = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241218_030201-test_fixed.fits"

    data = np.zeros((8, 8), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": 4.5,
            "CRPIX2": 4.5,
            "CDELT1": -0.03,
            "CDELT2": 0.03,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "DATE-OBS": "2024-12-18T03:00:01.4",
            "TIMESYS": "UTC",
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header

    # 20241218_030201 → 2024-12-18 03:02:01 UTC (not DATE-OBS).
    assert hdr["CRVAL1"] == pytest.approx(14.0996845, rel=0, abs=1e-4)
    assert hdr["CRVAL2"] == pytest.approx(37.0948037, rel=0, abs=1e-4)
    assert hdr["RADESYS"] == "FK5"


def test_fix_headers_leaves_crval_without_image_timestamp_in_name(tmp_path: Path):
    """If the basename has no ``-image-YYYYMMDD_HHMMSS``, CRVAL1/2 are not overwritten."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "no_stamp.fits"
    out_path = tmp_path / "no_stamp_fixed.fits"

    data = np.zeros((8, 8), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 1.25,
            "CRVAL2": 2.5,
            "CRPIX1": 4.5,
            "CRPIX2": 4.5,
            "CDELT1": -0.03,
            "CDELT2": 0.03,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header

    assert hdr["CRVAL1"] == pytest.approx(1.25)
    assert hdr["CRVAL2"] == pytest.approx(2.5)


def test_harmonize_celestial_coords_collapses_frequency_dim():
    """After combine, RA/Dec should be (m, l) only when slices share one WCS."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nm, nl, nf = 5, 6, 2
    ra0 = np.broadcast_to(np.linspace(100.0, 110.0, nl), (nm, nl)).copy()
    ra = np.stack([ra0, ra0], axis=0)
    dec = np.stack(
        [np.full((nm, nl), 40.0, dtype=np.float64), np.full((nm, nl), 40.0, dtype=np.float64)],
        axis=0,
    )
    hdr = "NAXIS = 2\nCRVAL1 = 105"
    ds = xr.Dataset(
        {"SKY": (("frequency", "m", "l"), np.ones((nf, nm, nl)))},
        coords={
            "frequency": np.array([45e6, 55e6], dtype=float),
            "l": np.linspace(-0.1, 0.1, nl),
            "m": np.linspace(-0.1, 0.1, nm),
            "right_ascension": (("frequency", "m", "l"), ra),
            "declination": (("frequency", "m", "l"), dec),
        },
    )
    ds["right_ascension"].attrs["fits_wcs_header"] = hdr
    ds["declination"].attrs["fits_wcs_header"] = hdr

    out = mod._harmonize_celestial_coords_independent_of_frequency(ds)
    assert "frequency" not in out.right_ascension.dims
    assert out.right_ascension.shape == (nm, nl)
    np.testing.assert_allclose(out.right_ascension.values, ra0)
    assert out["SKY"].attrs.get("fits_wcs_header") == hdr


def test_harmonize_celestial_coords_warns_on_large_wcs_drift(caplog):
    """Large per-channel RA/Dec drift vs reference should emit one warning."""
    import logging

    import numpy as np
    import xarray as xr

    mod = _import_module()
    nm, nl, nf = 5, 6, 2
    ra0 = np.broadcast_to(np.linspace(100.0, 110.0, nl), (nm, nl)).copy()
    ra1 = ra0 + 2.0
    ra = np.stack([ra0, ra1], axis=0)
    dec = np.stack(
        [np.full((nm, nl), 40.0, dtype=np.float64), np.full((nm, nl), 40.0, dtype=np.float64)],
        axis=0,
    )
    ds = xr.Dataset(
        {"SKY": (("frequency", "m", "l"), np.ones((nf, nm, nl)))},
        coords={
            "frequency": np.array([45e6, 55e6], dtype=float),
            "l": np.linspace(-0.1, 0.1, nl),
            "m": np.linspace(-0.1, 0.1, nm),
            "right_ascension": (("frequency", "m", "l"), ra),
            "declination": (("frequency", "m", "l"), dec),
        },
    )
    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    out = mod._harmonize_celestial_coords_independent_of_frequency(ds, warn_max_sep_arcsec=60.0)
    assert "Celestial coordinate grids differ" in caplog.text
    assert "frequency" not in out.right_ascension.dims


def test_harmonize_celestial_coords_samples_dask_backed_coords(monkeypatch):
    """Dask-backed celestial coords should be sampled before drift computation."""
    import numpy as np
    import xarray as xr

    da = pytest.importorskip("dask.array")
    mod = _import_module()
    nm, nl, nf = 300, 300, 2
    ra0 = np.broadcast_to(np.linspace(100.0, 110.0, nl), (nm, nl)).copy()
    ra = np.stack([ra0, ra0 + 0.01], axis=0)
    dec = np.stack(
        [np.full((nm, nl), 40.0, dtype=np.float64), np.full((nm, nl), 40.0, dtype=np.float64)],
        axis=0,
    )
    ds = xr.Dataset(
        {"SKY": (("frequency", "m", "l"), np.ones((nf, nm, nl)))},
        coords={
            "frequency": np.array([45e6, 55e6], dtype=float),
            "l": np.linspace(-0.1, 0.1, nl),
            "m": np.linspace(-0.1, 0.1, nm),
            "right_ascension": (
                ("frequency", "m", "l"),
                da.from_array(ra, chunks=(1, 75, 75)),
            ),
            "declination": (
                ("frequency", "m", "l"),
                da.from_array(dec, chunks=(1, 75, 75)),
            ),
        },
    )

    captured = {}
    original = mod._sky_sep_max_vs_ref_arcsec

    def _capture(
        ra_arr,
        dec_arr,
        *,
        ref_idx,
        max_points=mod._CELESTIAL_DRIFT_SAMPLE_MAX_POINTS,
    ):
        captured["shape"] = tuple(ra_arr.shape)
        return original(ra_arr, dec_arr, ref_idx=ref_idx, max_points=max_points)

    monkeypatch.setattr(mod, "_sky_sep_max_vs_ref_arcsec", _capture)
    out = mod._harmonize_celestial_coords_independent_of_frequency(ds)
    assert captured["shape"] == (nf, 1, mod._CELESTIAL_DRIFT_SAMPLE_MAX_POINTS)
    assert "frequency" not in out.right_ascension.dims
