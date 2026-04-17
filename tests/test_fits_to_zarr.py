"""Tests for FITS to Zarr conversion utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"


def _extract_datetime_re() -> re.Pattern[str]:
    content = MODULE_PATH.read_text()
    m = re.search(r'_DATETIME_RE = re\.compile\(r"([^"]+)"\)', content)
    assert m is not None, "_DATETIME_RE not found in module"
    return re.compile(m.group(1))


def _extract_mhz_re() -> re.Pattern[str]:
    content = MODULE_PATH.read_text()
    m = re.search(r'MHZ_RE = re\.compile\(r"([^"]+)"\)', content)
    assert m is not None, "MHZ_RE not found in module"
    return re.compile(m.group(1))


# ---- Datetime extraction ----


class TestDatetimeExtraction:
    """Verify YYYYMMDD_HHMMSS is extracted from various naming conventions."""

    def test_standard_filename(self):
        pat = _extract_datetime_re()
        m = pat.search("20240524_050009_41MHz_averaged_20000_iterations-I-image.fits")
        assert m is not None
        assert m.group("date") == "20240524"
        assert m.group("hms") == "050009"

    def test_pilot_filename(self):
        pat = _extract_datetime_re()
        m = pat.search("59MHz-Pilot-Snapshot-20241218_030338-I-image.fits")
        assert m is not None
        assert m.group("date") == "20241218"
        assert m.group("hms") == "030338"

    def test_fixed_suffix(self):
        pat = _extract_datetime_re()
        m = pat.search("20240524_050009_41MHz_averaged_20000_iterations-I-image_fixed.fits")
        assert m is not None
        assert m.group("date") == "20240524"

    def test_no_datetime(self):
        pat = _extract_datetime_re()
        m = pat.search("invalid.fits")
        assert m is None


# ---- MHz extraction ----


class TestMhzExtraction:
    """Verify <freq>MHz is extracted from various naming conventions."""

    def test_standard_filename(self):
        pat = _extract_mhz_re()
        m = pat.search("20240524_050009_41MHz_averaged_20000_iterations-I-image.fits")
        assert m is not None
        assert int(m.group(1)) == 41

    def test_pilot_filename(self):
        pat = _extract_mhz_re()
        m = pat.search("59MHz-Pilot-Snapshot-20241218_030338-I-image.fits")
        assert m is not None
        assert int(m.group(1)) == 59

    def test_larger_frequency(self):
        pat = _extract_mhz_re()
        m = pat.search("82MHz-Pilot-Snapshot-20241218_030338-I-image.fits")
        assert m is not None
        assert int(m.group(1)) == 82

    def test_no_mhz(self):
        pat = _extract_mhz_re()
        m = pat.search("invalid.fits")
        assert m is None


# ---- Module import ----


def test_module_can_be_imported():
    """Test that the fits_to_zarr_xradio module can be imported."""
    try:
        from ovro_lwa_portal import fits_to_zarr_xradio

        assert hasattr(fits_to_zarr_xradio, "convert_fits_dir_to_zarr")
        assert hasattr(fits_to_zarr_xradio, "MHZ_RE")
    except ImportError as e:
        pytest.skip(f"xradio dependencies not available: {e}")
