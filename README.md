# OVRO-LWA Portal

A Python library for radio astronomy data processing and visualization for the
Owens Valley Radio Observatory - Long Wavelength Array (OVRO-LWA).

## Features

- **Unified Data Loading**: Load OVRO-LWA data from local paths, remote URLs
  (S3, HTTPS), or DOI identifiers with a single `open_dataset()` function
- **FITS to Zarr Conversion**: Convert OVRO-LWA FITS image files to
  cloud-optimized Zarr format
- **Command-Line Interface**: User-friendly `ovro-ingest` CLI with progress
  tracking
- **WCS Coordinate Preservation**: Maintain celestial coordinates (RA/Dec) for
  FITS-free analysis
- **Incremental Processing**: Append new observations to existing Zarr stores
- **Concurrent Write Protection**: File locking prevents data corruption from
  simultaneous processes
- **Optional Workflow Orchestration**: Prefect integration for production
  deployments

## Prerequisites

This project uses [Pixi](https://pixi.sh) for dependency management and task
execution. Install Pixi by following the
[installation instructions](https://pixi.sh/latest/#installation).

## Getting Started

### Installation

#### For Users

Install the package using pip:

```bash
pip install git+https://github.com/uw-ssec/ovro-lwa-portal.git
```

Or install from a local clone:

```bash
pip install .
```

#### For Developers

Install dependencies using Pixi:

```bash
# Install dependencies (Pixi will automatically create the environment)
pixi install
```

For detailed installation and development instructions, see
[CONTRIBUTING.md](CONTRIBUTING.md).

### Onboarding

For first-time setup, use the onboarding environment to configure your
development environment:

```bash
pixi run -e onboard onboard
```

This will:

- Install pre-commit hooks in your git repository
- Set up shell completion for ssec-cli
- Run the SSEC onboarding process

## Project Structure

This project is organized using Pixi features for modular dependency management:

- **`pre-commit`**: Code quality and consistency checks
- **`gh-cli`**: GitHub CLI for repository interactions
- **`onboard`**: Tools for project onboarding and setup

## Available Environments

- **`default`**: Standard development environment with pre-commit hooks and
  GitHub CLI
- **`onboard`**: Extended environment including onboarding tools

## Development

### Using Different Environments

Switch between environments as needed:

```bash
# Use default environment
pixi shell

# Use onboard environment
pixi shell -e onboard
```

### Adding Dependencies

Edit `pyproject.toml` to add new dependencies in the `[tool.pixi.dependencies]`
section:

```toml
[tool.pixi.dependencies]
your-package = ">=1.0.0"
```

Then run:

```bash
pixi install
```

or

Directly add packages (this will edit the pyproject.toml and install):

```bash
pixi add your-package
```

## Quick Start

### Loading OVRO-LWA Data

Load data from various sources with a unified interface:

```python
import ovro_lwa_portal

# Load from local zarr store
ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

# Load from remote URL
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

# Load via DOI
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")

# Customize chunking for large datasets
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50}  # or chunks="auto" (default), chunks=None
)
```

For remote data access, install with remote extras:

```bash
pip install 'ovro_lwa_portal[remote]'
```

See the [open_dataset documentation](docs/open_dataset.md) for more details.

### Using the FITS to Zarr Ingest CLI

After installation, convert OVRO-LWA FITS files to Zarr format:

```bash
# Basic conversion
ovro-ingest convert /path/to/fits /path/to/output

# With custom options
ovro-ingest convert /path/to/fits /path/to/output \
    --zarr-name my_data.zarr \
    --chunk-lm 2048 \
    --rebuild

# Show help
ovro-ingest convert --help
```

For detailed documentation on the ingest module, see the
[Ingest Module README](src/ovro_lwa_portal/ingest/README.md).

### Using the Python API

```python
from pathlib import Path
from ovro_lwa_portal.ingest import FITSToZarrConverter
from ovro_lwa_portal.ingest.core import ConversionConfig

# Configure conversion
config = ConversionConfig(
    input_dir=Path("/path/to/fits"),
    output_dir=Path("/path/to/output"),
    zarr_name="ovro_lwa_data.zarr",
    chunk_lm=1024,
)

# Execute conversion
converter = FITSToZarrConverter(config)
result = converter.convert()
print(f"Created: {result}")
```

## Technology Stack

- **Core**: Python 3.12, xarray, dask, zarr
- **Astronomy**: astropy, xradio, python-casacore
- **CLI**: typer, rich (progress bars and formatted output)
- **Workflow**: prefect (optional orchestration)
- **Storage**: Zarr format optimized for cloud access
- **Environment Management**: pixi

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of
conduct and the process for submitting pull requests.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE)
file.

## Project Resources

- eScience Slack channel: 🔒
  [#ssec-ovro-lwa-portal](https://escience-institute.slack.com/archives/C098GJYLNBW)
- SSEC Sharepoint (**INTERNAL SSEC ONLY**): 🔒
  [Projects/OVROXarrraySciPlt](https://uwnetid.sharepoint.com/:f:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt?csf=1&web=1&e=P5QKAc)
- Shared Sharepoint Directory: 🔒
  [UW SSEC Caltech OVRO-LWA Portal Shared Folder](https://uwnetid.sharepoint.com/:f:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt/UW%20SSEC%20Caltech%20OVRO-LWA%20Portal%20Shared%20Folder?csf=1&web=1&e=siXUk2)
- [User Stories Document 🔒](https://uwnetid.sharepoint.com/:w:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt/UW%20SSEC%20Caltech%20OVRO-LWA%20Portal%20Shared%20Folder/SSEC%20OVRO-LWA%20Portal%20User%20Stories.docx?d=w15624ab2d3c0475e95a2865a346e359b&csf=1&web=1&e=ImDH96)

## General Discussions

For general discussion, ideas, and resources please use the
[GitHub Discussions](https://github.com/uw-ssec/ovro-lwa-portal/discussions).
However, if there's an internal discussion that need to happen, please use the
slack channel provided.

- Meeting Notes in GitHub:
  [discussions/meetings](https://github.com/uw-ssec/ovro-lwa-portal/discussions/categories/meetings)

## Questions

If you have any questions about our process, or locations of SSEC resources,
please ask [Anshul Tambay](https://github.com/atambay37).
