"""OVRO-LWA Portal: Radio astronomy data processing and visualization library.

This package provides tools for processing radio astronomy data from the
Owens Valley Radio Observatory - Long Wavelength Array (OVRO-LWA),
including FITS to Zarr conversion and visualization components.
"""

from __future__ import annotations

try:
    from .version import version as __version__  # type: ignore[import-not-found]
except ImportError:
    __version__ = "0.0.0+unknown"

from . import fits_to_zarr_xradio

__all__ = ["__version__", "fits_to_zarr_xradio"]
