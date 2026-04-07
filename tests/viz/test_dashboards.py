"""Tests for dashboard composition."""

from __future__ import annotations

import pytest

pn = pytest.importorskip("panel")

from ovro_lwa_portal.viz.dashboards import create_exploration_dashboard


class TestCreateExplorationDashboard:
    """Tests for create_exploration_dashboard."""

    def test_returns_tabs(self, viz_dataset):
        dashboard = create_exploration_dashboard(viz_dataset)
        assert isinstance(dashboard, pn.Tabs)

    def test_has_three_tabs(self, viz_dataset):
        dashboard = create_exploration_dashboard(viz_dataset)
        assert len(dashboard) == 3

    def test_tab_names(self, viz_dataset):
        dashboard = create_exploration_dashboard(viz_dataset)
        # Panel Tabs does not expose tab names via a public API —
        # _names is the only way to retrieve them.
        names = dashboard._names
        assert "Image" in names
        assert "Dynamic Spectrum" in names
        assert "Cutout" in names
