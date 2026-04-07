"""Interactive visualization framework for OVRO-LWA datasets.

This module provides Panel/HoloViews-based interactive explorers for
radio astronomy data. All dependencies are optional — install with:

    pip install 'ovro_lwa_portal[visualization]'

Examples
--------
>>> ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")
>>> ds.radport.explore_image()  # Launch interactive image explorer
>>> ds.radport.explore()        # Launch full exploration dashboard
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ovro_lwa_portal.viz._imports import check_viz_deps

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    "ImageExplorer",
    "DynamicSpectrumExplorer",
    "CutoutExplorer",
    "SkyViewer",
    "create_exploration_dashboard",
]


def ImageExplorer(ds: xr.Dataset, **kwargs: Any) -> Any:  # noqa: N802
    """Create an interactive image explorer.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.
    **kwargs
        Passed to :class:`~ovro_lwa_portal.viz.explorers.ImageExplorer`.

    Returns
    -------
    ovro_lwa_portal.viz.explorers.ImageExplorer
        Explorer instance. Call ``.panel()`` to get a
        ``panel.viewable.Viewable`` layout.
    """
    check_viz_deps()
    from ovro_lwa_portal.viz.explorers import ImageExplorer as _IE

    return _IE(ds, **kwargs)


def DynamicSpectrumExplorer(ds: xr.Dataset, **kwargs: Any) -> Any:  # noqa: N802
    """Create an interactive dynamic spectrum explorer.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.
    **kwargs
        Passed to :class:`~ovro_lwa_portal.viz.explorers.DynamicSpectrumExplorer`.

    Returns
    -------
    ovro_lwa_portal.viz.explorers.DynamicSpectrumExplorer
        Explorer instance. Call ``.panel()`` to get a
        ``panel.viewable.Viewable`` layout.
    """
    check_viz_deps()
    from ovro_lwa_portal.viz.explorers import DynamicSpectrumExplorer as _DSE

    return _DSE(ds, **kwargs)


def CutoutExplorer(ds: xr.Dataset, **kwargs: Any) -> Any:  # noqa: N802
    """Create an interactive cutout explorer.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.
    **kwargs
        Passed to :class:`~ovro_lwa_portal.viz.explorers.CutoutExplorer`.

    Returns
    -------
    ovro_lwa_portal.viz.explorers.CutoutExplorer
        Explorer instance. Call ``.panel()`` to get a
        ``panel.viewable.Viewable`` layout.
    """
    check_viz_deps()
    from ovro_lwa_portal.viz.explorers import CutoutExplorer as _CE

    return _CE(ds, **kwargs)


def SkyViewer(ds: xr.Dataset, **kwargs: Any) -> Any:  # noqa: N802
    """Create an interactive sky viewer with Aladin Lite.

    Overlays OVRO-LWA data on astronomical survey backgrounds (DSS,
    WISE, Planck, etc.) with real-time panning, zooming, and
    coordinate exploration. Requires WCS header in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset with WCS header to explore.
    **kwargs
        Passed to :class:`~ovro_lwa_portal.viz.sky_viewer.SkyViewer`.

    Returns
    -------
    ovro_lwa_portal.viz.sky_viewer.SkyViewer
        Sky viewer instance. Call ``.panel()`` to get a
        ``panel.viewable.Viewable`` layout.
    """
    check_viz_deps()
    from ovro_lwa_portal.viz.sky_viewer import SkyViewer as _SV

    return _SV(ds, **kwargs)


def create_exploration_dashboard(ds: xr.Dataset) -> Any:
    """Create a comprehensive exploration dashboard.

    Combines image, dynamic spectrum, and cutout explorers into a
    tabbed dashboard layout.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.

    Returns
    -------
    panel.viewable.Viewable
        Panel tabbed layout with all explorers.
    """
    check_viz_deps()
    from ovro_lwa_portal.viz.dashboards import create_exploration_dashboard as _ced

    return _ced(ds)
