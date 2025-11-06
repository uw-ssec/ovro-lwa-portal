"""FITS to Zarr ingest package for OVRO-LWA data processing.

This package provides tools for converting OVRO-LWA FITS image files to optimized
Zarr stores with support for incremental processing, WCS coordinate preservation,
and robust error handling.

Usage
-----
Command-line interface::

    ovro-ingest convert /path/to/fits /path/to/output --rebuild

Python API::

    from ovro_lwa_portal.ingest import FITSToZarrConverter
    from ovro_lwa_portal.ingest.core import ConversionConfig

    config = ConversionConfig(
        input_dir="/path/to/fits",
        output_dir="/path/to/output",
        rebuild=False,
    )
    converter = FITSToZarrConverter(config)
    result = converter.convert()

Optional Prefect integration::

    from ovro_lwa_portal.ingest.prefect_workflow import fits_to_zarr_flow

    result = fits_to_zarr_flow(
        input_dir="/path/to/fits",
        output_dir="/path/to/output",
    )
"""

from __future__ import annotations

from ovro_lwa_portal.ingest.core import (
    ConversionConfig,
    FITSToZarrConverter,
    ProgressCallback,
)

__all__ = ["FITSToZarrConverter", "ConversionConfig", "ProgressCallback"]
