"""Tests for the ipyaladin-based sky viewer."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pn = pytest.importorskip("panel")
pytest.importorskip("ipyaladin")

from ovro_lwa_portal.viz.sky_viewer import SkyViewer, _build_fits_hdu


@pytest.fixture
def wcs_dataset() -> xr.Dataset:
    """Create a dataset with WCS header for sky viewer tests."""
    wcs_header = """NAXIS   =                    2
NAXIS1  =                   50
NAXIS2  =                   50
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CRPIX1  =                 25.0
CRPIX2  =                 25.0
CRVAL1  =                180.0
CRVAL2  =                 45.0
CDELT1  =                 -1.0
CDELT2  =                  1.0
CUNIT1  = 'deg'
CUNIT2  = 'deg'
RADESYS = 'FK5'
EQUINOX =               2000.0"""

    np.random.seed(42)
    ds = xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 3, 2, 50, 50) * 10,
            ),
        },
        coords={
            "time": [60000.0, 60000.1],
            "frequency": [46e6, 50e6, 54e6],
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )
    ds["SKY"].attrs["fits_wcs_header"] = wcs_header
    return ds


class TestBuildFitsHdu:
    """Tests for _build_fits_hdu."""

    def test_returns_hdulist(self, wcs_dataset):
        from astropy.io.fits import HDUList

        hdul = _build_fits_hdu(wcs_dataset)
        assert isinstance(hdul, HDUList)

    def test_data_shape_is_transposed(self, wcs_dataset):
        """FITS data should be (NAXIS2, NAXIS1) = (n_m, n_l)."""
        hdul = _build_fits_hdu(wcs_dataset)
        assert hdul[0].data.shape == (50, 50)

    def test_wcs_header_preserved(self, wcs_dataset):
        hdul = _build_fits_hdu(wcs_dataset)
        header = hdul[0].header
        assert header["CTYPE1"] == "RA---SIN"
        assert header["CTYPE2"] == "DEC--SIN"
        assert header["CRVAL1"] == 180.0
        assert header["CRVAL2"] == 45.0

    def test_float32_output(self, wcs_dataset):
        hdul = _build_fits_hdu(wcs_dataset)
        assert hdul[0].data.dtype == np.float32
        assert hdul[0].header["BITPIX"] == -32

    def test_robust_clipping(self, wcs_dataset):
        hdul = _build_fits_hdu(wcs_dataset, robust=True)
        data = hdul[0].data
        assert not np.any(np.isnan(data))

    def test_no_wcs_raises(self, viz_dataset):
        """Should raise ValueError when dataset has no WCS."""
        with pytest.raises(ValueError, match="No WCS header"):
            _build_fits_hdu(viz_dataset)

    def test_different_time_freq(self, wcs_dataset):
        hdul = _build_fits_hdu(wcs_dataset, time_idx=1, freq_idx=2, pol=1)
        assert hdul[0].data.shape == (50, 50)


class TestSkyViewer:
    """Tests for SkyViewer."""

    def test_creates_with_wcs_dataset(self, wcs_dataset):
        viewer = SkyViewer(wcs_dataset)
        assert viewer._phase_center_ra == 180.0
        assert viewer._phase_center_dec == 45.0

    def test_bounds_from_dataset(self, wcs_dataset):
        viewer = SkyViewer(wcs_dataset)
        assert viewer.param.time_idx.bounds == (0, 1)
        assert viewer.param.freq_idx.bounds == (0, 2)

    def test_panel_returns_viewable(self, wcs_dataset):
        viewer = SkyViewer(wcs_dataset)
        layout = viewer.panel()
        assert isinstance(layout, pn.viewable.Viewable)

    def test_aladin_widget_exists(self, wcs_dataset):
        from ipyaladin import Aladin

        viewer = SkyViewer(wcs_dataset)
        assert isinstance(viewer._aladin, Aladin)

    def test_survey_presets(self, wcs_dataset):
        viewer = SkyViewer(wcs_dataset)
        assert "DSS Color" in viewer.param.survey.objects

    def test_update_overlay(self, wcs_dataset):
        """Overlay update should not raise."""
        viewer = SkyViewer(wcs_dataset)
        viewer._update_overlay()
        assert viewer._current_overlay_name is not None


class TestAccessorSkyIntegration:
    """Tests for ds.radport.explore_sky()."""

    def test_explore_sky_returns_viewable(self, wcs_dataset):
        import ovro_lwa_portal  # noqa: F401

        layout = wcs_dataset.radport.explore_sky()
        assert isinstance(layout, pn.viewable.Viewable)

    def test_dashboard_includes_sky_tab(self, wcs_dataset):
        import ovro_lwa_portal  # noqa: F401

        dashboard = wcs_dataset.radport.explore()
        assert "Sky Viewer" in dashboard._names
