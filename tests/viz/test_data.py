"""Tests for the data bridge (_data.py)."""

from __future__ import annotations

import pytest

hv = pytest.importorskip("holoviews")

from ovro_lwa_portal.viz._data import (
    PreloadedCube,
    cutout_image_element,
    dynamic_spectrum_element,
    light_curve_element,
    sky_image_element,
    spectrum_element,
)


@pytest.fixture
def cube(viz_dataset):
    """Create a PreloadedCube from the test dataset."""
    return PreloadedCube(viz_dataset, var="SKY", pol=0)


class TestPreloadedCube:
    """Tests for PreloadedCube."""

    def test_creates_from_dataset(self, viz_dataset):
        cube = PreloadedCube(viz_dataset)
        assert cube.n_times == 2
        assert cube.n_freqs == 3

    def test_image_returns_2d(self, cube):
        img = cube.image(0, 0)
        assert img.ndim == 2

    def test_dynamic_spectrum_shape(self, cube):
        l_idx, m_idx = cube.nearest_lm_idx(0.0, 0.0)
        ds = cube.dynamic_spectrum(l_idx, m_idx)
        assert ds.shape == (2, 3)

    def test_nearest_lm_idx(self, cube):
        l_idx, m_idx = cube.nearest_lm_idx(0.0, 0.0)
        assert isinstance(l_idx, int)
        assert isinstance(m_idx, int)


class TestSkyImageElement:
    """Tests for sky_image_element."""

    def test_returns_hv_image(self, cube):
        img = sky_image_element(cube)
        assert isinstance(img, hv.Image)

    def test_different_time_freq(self, cube):
        img = sky_image_element(cube, time_idx=1, freq_idx=2)
        assert isinstance(img, hv.Image)

    def test_robust_scaling(self, cube):
        img = sky_image_element(cube, robust=True)
        assert isinstance(img, hv.Image)

    def test_no_robust_scaling(self, cube):
        img = sky_image_element(cube, robust=False)
        assert isinstance(img, hv.Image)


class TestDynamicSpectrumElement:
    """Tests for dynamic_spectrum_element."""

    def test_returns_hv_image(self, cube):
        img = dynamic_spectrum_element(cube, l=0.0, m=0.0)
        assert isinstance(img, hv.Image)


class TestLightCurveElement:
    """Tests for light_curve_element."""

    def test_returns_hv_curve(self, cube):
        curve = light_curve_element(cube, l=0.0, m=0.0, freq_idx=0)
        assert isinstance(curve, hv.Curve)


class TestSpectrumElement:
    """Tests for spectrum_element."""

    def test_returns_hv_curve(self, cube):
        curve = spectrum_element(cube, l=0.0, m=0.0, time_idx=0)
        assert isinstance(curve, hv.Curve)


class TestCutoutImageElement:
    """Tests for cutout_image_element."""

    def test_returns_hv_image(self, cube):
        img = cutout_image_element(cube, l_center=0.0, m_center=0.0, dl=0.5, dm=0.5)
        assert isinstance(img, hv.Image)

    def test_cutout_is_smaller_than_full(self, cube):
        full = sky_image_element(cube)
        cutout = cutout_image_element(cube, l_center=0.0, m_center=0.0, dl=0.3, dm=0.3)
        assert cutout.data.size < full.data.size
