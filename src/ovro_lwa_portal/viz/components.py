"""Reusable styled HoloViews components for OVRO-LWA visualization.

Provides consistent styling defaults and common plot configurations
used across the explorer classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import holoviews as hv


# Standard colormaps suitable for radio astronomy data
COLORMAPS = ["inferno", "viridis", "plasma", "magma", "cividis", "gray"]

# Default plot dimensions (pixels).
# Fixed sizes avoid the DynamicMap + responsive rendering bug where
# plots render at zero size and need a manual resize to appear.
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 600
CURVE_WIDTH = 500
CURVE_HEIGHT = 280


def style_sky_image(img: hv.Image, *, cmap: str = "inferno") -> hv.Image:
    """Apply standard styling to a sky image element.

    Parameters
    ----------
    img : hv.Image
        HoloViews image element.
    cmap : str
        Colormap name.

    Returns
    -------
    hv.Image
        Styled image.
    """
    return img.opts(
        cmap=cmap,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        tools=["hover", "tap", "crosshair"],
        active_tools=["tap"],
    )


def style_spectrum_image(img: hv.Image, *, cmap: str = "inferno") -> hv.Image:
    """Apply standard styling to a dynamic spectrum image.

    Parameters
    ----------
    img : hv.Image
        HoloViews image element.
    cmap : str
        Colormap name.

    Returns
    -------
    hv.Image
        Styled image.
    """
    return img.opts(
        cmap=cmap,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        tools=["hover", "tap", "crosshair"],
        active_tools=["tap"],
    )


def style_curve(curve: hv.Curve) -> hv.Curve:
    """Apply standard styling to a curve element.

    Parameters
    ----------
    curve : hv.Curve
        HoloViews curve element.

    Returns
    -------
    hv.Curve
        Styled curve.
    """
    return curve.opts(
        width=CURVE_WIDTH,
        height=CURVE_HEIGHT,
        tools=["hover"],
        line_width=1.5,
    )
