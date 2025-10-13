# OVRO-LWA Portal

A Python library for radio astronomy data processing and visualization for the
Owens Valley Radio Observatory - Long Wavelength Array (OVRO-LWA).

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

Edit `pixi.toml` to add new dependencies:

```toml
[dependencies]
your-package = ">=1.0.0"
```

Then run:

```bash
pixi install
```

or

Directly add packages (this will edit the pixi toml and install):

```bash
pixi add your-package
```

## Technology Stack

- **Core**: Python 3.12, xarray, dask, zarr
- **Astronomy**: astropy, image-plane-correction
- **Visualization**: Panel-compatible components
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
