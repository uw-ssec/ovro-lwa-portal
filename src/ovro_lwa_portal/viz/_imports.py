"""Centralized dependency checking for the visualization module."""

from __future__ import annotations

from typing import NoReturn

_VIZ_DEPS_CHECKED: bool | None = None


def check_viz_deps() -> None:
    """Verify that visualization dependencies are installed.

    Raises
    ------
    ImportError
        If panel, holoviews, or bokeh are not available, with install instructions.
    """
    global _VIZ_DEPS_CHECKED  # noqa: PLW0603
    if _VIZ_DEPS_CHECKED is True:
        return
    if _VIZ_DEPS_CHECKED is False:
        _raise_missing()

    missing: list[str] = []
    for pkg in ("panel", "holoviews", "bokeh", "param"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        _VIZ_DEPS_CHECKED = False
        _raise_missing(missing)

    _VIZ_DEPS_CHECKED = True


def _raise_missing(missing: list[str] | None = None) -> NoReturn:
    msg = (
        "Interactive visualization requires additional dependencies"
        + (f" ({', '.join(missing)})" if missing else "")
        + ". Install with: pip install 'ovro_lwa_portal[visualization]'"
    )
    raise ImportError(msg)
