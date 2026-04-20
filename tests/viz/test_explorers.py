"""Tests for explorer classes (state management and panel creation)."""

from __future__ import annotations

import pytest

pn = pytest.importorskip("panel")
hv = pytest.importorskip("holoviews")

from ovro_lwa_portal.viz.explorers import (
    CutoutExplorer,
    DynamicSpectrumExplorer,
    ImageExplorer,
)


class TestImageExplorer:
    """Tests for ImageExplorer."""

    def test_creates_with_defaults(self, viz_dataset):
        explorer = ImageExplorer(viz_dataset)
        assert explorer.time_idx == 0
        assert explorer.freq_idx == 0
        assert explorer.var == "SKY"

    def test_bounds_set_from_dataset(self, viz_dataset):
        explorer = ImageExplorer(viz_dataset)
        assert explorer.param.time_idx.bounds == (0, 1)
        assert explorer.param.freq_idx.bounds == (0, 2)

    def test_available_vars_detected(self, viz_dataset_with_beam):
        explorer = ImageExplorer(viz_dataset_with_beam)
        assert "SKY" in explorer.param.var.objects
        assert "BEAM" in explorer.param.var.objects

    def test_panel_returns_viewable(self, viz_dataset):
        explorer = ImageExplorer(viz_dataset)
        layout = explorer.panel()
        assert isinstance(layout, pn.viewable.Viewable)

    def test_image_view_returns_hv_image(self, viz_dataset):
        explorer = ImageExplorer(viz_dataset)
        img = explorer._image_view()
        assert isinstance(img, hv.Image)

    def test_changing_time_idx(self, viz_dataset):
        explorer = ImageExplorer(viz_dataset)
        explorer.time_idx = 1
        img = explorer._image_view()
        assert isinstance(img, hv.Image)

    def test_changing_cmap(self, viz_dataset):
        explorer = ImageExplorer(viz_dataset)
        explorer.cmap = "viridis"
        img = explorer._image_view()
        assert isinstance(img, hv.Image)


class TestDynamicSpectrumExplorer:
    """Tests for DynamicSpectrumExplorer."""

    def test_creates_with_lm_defaults(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        assert explorer.l_val == 0.0
        assert explorer.m_val == 0.0

    def test_panel_returns_viewable(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        layout = explorer.panel()
        assert isinstance(layout, pn.viewable.Viewable)

    def test_dynspec_view_returns_image(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        img = explorer._dynspec_view()
        assert isinstance(img, hv.Image)

    def test_linked_spectrum_empty_before_click(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        curve = explorer._linked_spectrum()
        assert isinstance(curve, hv.Curve)
        assert len(curve) == 0

    def test_linked_light_curve_empty_before_click(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        curve = explorer._linked_light_curve()
        assert isinstance(curve, hv.Curve)
        assert len(curve) == 0

    def test_linked_spectrum_with_coordinates(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        # Simulate tap on the dynamic spectrum image
        explorer._on_tap(x=60000.0, y=50.0)
        curve = explorer._linked_spectrum()
        assert isinstance(curve, hv.Curve)
        assert len(curve) > 0

    def test_linked_light_curve_with_coordinates(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        explorer._on_tap(x=60000.0, y=50.0)
        curve = explorer._linked_light_curve()
        assert isinstance(curve, hv.Curve)
        assert len(curve) > 0

    def test_tap_source_set_to_dynamicmap(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        explorer.panel()
        assert isinstance(explorer._tap.source, hv.DynamicMap)

    def test_on_tap_updates_click_params(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        assert explorer._click_time is None
        assert explorer._click_freq_mhz is None
        explorer._on_tap(x=60000.05, y=50.0)
        assert explorer._click_time == 60000.05
        assert explorer._click_freq_mhz == 50.0

    def test_on_tap_ignores_none(self, viz_dataset):
        explorer = DynamicSpectrumExplorer(viz_dataset, l=0.0, m=0.0)
        explorer._on_tap(x=None, y=None)
        assert explorer._click_time is None


class TestCutoutExplorer:
    """Tests for CutoutExplorer."""

    def test_creates_with_defaults(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        assert explorer.l_center == 0.0
        assert explorer.dl == 0.1

    def test_panel_returns_viewable(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        layout = explorer.panel()
        assert isinstance(layout, pn.viewable.Viewable)

    def test_cutout_view_returns_image(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        img = explorer._cutout_view()
        assert isinstance(img, hv.Image)

    def test_linked_spectrum_empty_before_click(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        curve = explorer._linked_spectrum()
        assert isinstance(curve, hv.Curve)
        assert len(curve) == 0

    def test_linked_light_curve_empty_before_click(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        curve = explorer._linked_light_curve()
        assert isinstance(curve, hv.Curve)
        assert len(curve) == 0

    def test_linked_spectrum_with_coordinates(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        # Simulate tap on the cutout image
        explorer._on_tap(x=0.0, y=0.0)
        curve = explorer._linked_spectrum()
        assert isinstance(curve, hv.Curve)
        assert len(curve) > 0

    def test_linked_light_curve_with_coordinates(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        explorer._on_tap(x=0.0, y=0.0)
        curve = explorer._linked_light_curve()
        assert isinstance(curve, hv.Curve)
        assert len(curve) > 0

    def test_tap_source_set_to_dynamicmap(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        explorer.panel()
        assert isinstance(explorer._tap.source, hv.DynamicMap)

    def test_on_tap_updates_click_params(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        assert explorer._click_l is None
        assert explorer._click_m is None
        explorer._on_tap(x=0.05, y=-0.03)
        assert explorer._click_l == 0.05
        assert explorer._click_m == -0.03

    def test_on_tap_ignores_none(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        explorer._on_tap(x=None, y=None)
        assert explorer._click_l is None

    def test_spectrum_updates_on_time_change(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        explorer._on_tap(x=0.0, y=0.0)
        curve1 = explorer._linked_spectrum()
        explorer.time_idx = 1
        curve2 = explorer._linked_spectrum()
        # Both should have data, but from different time steps
        assert len(curve1) > 0
        assert len(curve2) > 0

    def test_light_curve_updates_on_freq_change(self, viz_dataset):
        explorer = CutoutExplorer(viz_dataset)
        explorer._on_tap(x=0.0, y=0.0)
        curve1 = explorer._linked_light_curve()
        explorer.freq_idx = 1
        curve2 = explorer._linked_light_curve()
        assert len(curve1) > 0
        assert len(curve2) > 0
