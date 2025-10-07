# OVRO-LWA Portal

A Python library for radio astronomy data processing and visualization for the
Owens Valley Radio Observatory - Long Wavelength Array (OVRO-LWA)

## Technology Stack

- **Core**: Python 3.12, xarray, dask, zarr
- **Astronomy**: astropy, image-plane-correction
- **Visualization**: Panel-compatible components
- **Storage**: Zarr format optimized for cloud access
- **Environment Management**: pixi

## Getting Started

This project uses [pixi](https://pixi.sh/) for environment and dependency
management.

### Prerequisites

- [pixi](https://pixi.sh/) installed on your system

### Installation

1. Clone the repository:

```bash
git clone https://github.com/uw-ssec/ovro-lwa-portal.git
cd ovro-lwa-portal
```

1. Install dependencies and set up the environment:

```bash
pixi install
```

### Available Environments

- **default**: Core development environment with pre-commit hooks
- **onboarding**: Development environment with SSEC onboarding tools

### Common Tasks

```bash
# Activate the environment
pixi shell

# Run pre-commit hooks
pixi run pre-commit

# Run all pre-commit hooks on all files
pixi run pre-commit-all

# Install pre-commit hooks
pixi run pre-commit-install

# Complete project onboarding (includes SSEC setup)
pixi run onboard
```

## Relevant Links for project documentations and context

- eScience Slack channel: 🔒
  [#ssec-ovro-lwa-portal](https://escience-institute.slack.com/archives/C098GJYLNBW)
- SSEC Sharepoint (**INTERNAL SSEC ONLY**): 🔒
  [Projects/OVROXarrraySciPlt](https://uwnetid.sharepoint.com/:f:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt?csf=1&web=1&e=P5QKAc)
- Shared Sharepoint Directory: 🔒
  [UW SSEC Caltech OVRO-LWA Portal Shared Folder](https://uwnetid.sharepoint.com/:f:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt/UW%20SSEC%20Caltech%20OVRO-LWA%20Portal%20Shared%20Folder?csf=1&web=1&e=siXUk2)

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
