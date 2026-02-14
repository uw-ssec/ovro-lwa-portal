# OVRO-LWA Portal

A Python library for radio astronomy data processing and visualization for the Owens Valley Radio Observatory - Long Wavelength Array (OVRO-LWA).

## Features

- **Unified Data Loading**: Load OVRO-LWA data from local paths, remote URLs (S3, HTTPS), or DOI identifiers with a single `open_dataset()` function
- **FITS to Zarr Conversion**: Convert OVRO-LWA FITS image files to cloud-optimized Zarr format
- **Command-Line Interface**: User-friendly `ovro-ingest` CLI with progress tracking
- **WCS Coordinate Preservation**: Maintain celestial coordinates (RA/Dec) for FITS-free analysis
- **Incremental Processing**: Append new observations to existing Zarr stores
- **Concurrent Write Protection**: File locking prevents data corruption from simultaneous processes
- **Optional Workflow Orchestration**: Prefect integration for production deployments

## Technology Stack

- **Core**: Python 3.12, xarray, dask, zarr
- **Astronomy**: astropy, xradio, python-casacore
- **CLI**: typer, rich (progress bars and formatted output)
- **Workflow**: prefect (optional orchestration)
- **Storage**: Zarr format optimized for cloud access
- **Environment Management**: pixi

## Quick Start

```python
import ovro_lwa_portal

# Load from local zarr store
ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

# Load from remote URL
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

# Load via DOI
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
```

## Getting Help

- [GitHub Issues](https://github.com/uw-ssec/ovro-lwa-portal/issues)
- [GitHub Discussions](https://github.com/uw-ssec/ovro-lwa-portal/discussions)

## License

This project is licensed under the BSD License - see the [LICENSE](https://github.com/uw-ssec/ovro-lwa-portal/blob/main/LICENSE) file for details.
