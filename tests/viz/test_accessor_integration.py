"""Tests for accessor integration (explore methods)."""

from __future__ import annotations

import pytest

pn = pytest.importorskip("panel")

import ovro_lwa_portal  # noqa: F401 — registers accessor


class TestAccessorExplore:
    """Tests for ds.radport.explore*() methods."""

    def test_explore_returns_tabs(self, viz_dataset):
        result = viz_dataset.radport.explore()
        assert isinstance(result, pn.Tabs)

    def test_explore_image_returns_viewable(self, viz_dataset):
        result = viz_dataset.radport.explore_image()
        assert isinstance(result, pn.viewable.Viewable)

    def test_explore_dynamic_spectrum_returns_viewable(self, viz_dataset):
        result = viz_dataset.radport.explore_dynamic_spectrum(l=0.0, m=0.0)
        assert isinstance(result, pn.viewable.Viewable)

    def test_explore_sky_requires_wcs(self, viz_dataset):
        if not viz_dataset.radport.has_wcs:
            with pytest.raises(ValueError, match="WCS header"):
                viz_dataset.radport.explore_sky()
        else:
            pytest.skip("Dataset has WCS — cannot test missing-WCS guard")

    def test_explore_sky_returns_viewable(self, viz_dataset):
        if not viz_dataset.radport.has_wcs:
            pytest.skip("Dataset does not have WCS header")
        result = viz_dataset.radport.explore_sky()
        assert isinstance(result, pn.viewable.Viewable)
