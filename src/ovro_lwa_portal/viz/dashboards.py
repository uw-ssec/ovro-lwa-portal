"""Pre-built dashboard layouts combining multiple explorer components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import panel as pn

from ovro_lwa_portal.viz.explorers import (
    CutoutExplorer,
    DynamicSpectrumExplorer,
    ImageExplorer,
)

if TYPE_CHECKING:
    import xarray as xr


def create_exploration_dashboard(ds: xr.Dataset, **kwargs: Any) -> pn.Tabs:
    """Create a comprehensive exploration dashboard.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.

    Returns
    -------
    panel.Tabs
        Tabbed layout with all explorers.
    """
    image_explorer = ImageExplorer(ds)
    dynspec_explorer = DynamicSpectrumExplorer(ds)
    cutout_explorer = CutoutExplorer(ds)

    tabs: list[tuple[str, Any]] = [
        ("Image", image_explorer.panel()),
        ("Dynamic Spectrum", dynspec_explorer.panel()),
        ("Cutout", cutout_explorer.panel()),
    ]

    if ds.radport.has_wcs:
        from ovro_lwa_portal.viz.sky_viewer import SkyViewer

        sky_viewer = SkyViewer(ds)
        tabs.append(("Sky Viewer", sky_viewer.panel()))

    return pn.Tabs(*tabs, sizing_mode="stretch_width")
