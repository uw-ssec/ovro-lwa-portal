# Installation Guide

## For Users

### Install from Source

To install the latest version from source:

```bash
pip install git+https://github.com/uw-ssec/ovro-lwa-portal.git
```

### Install from Local Clone

If you have cloned the repository:

```bash
cd ovro-lwa-portal
pip install .
```

### Install in Development Mode

For development, install in editable mode:

```bash
cd ovro-lwa-portal
pip install -e .
```

## For Developers

### Prerequisites

This project uses [Pixi](https://pixi.sh) for dependency management. Install
Pixi by following the
[installation instructions](https://pixi.sh/latest/#installation).

### Setup Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/uw-ssec/ovro-lwa-portal.git
   cd ovro-lwa-portal
   ```

2. Install dependencies using Pixi:

   ```bash
   pixi install
   ```

3. Install pre-commit hooks:

   ```bash
   pixi run pre-commit-install
   ```

### Running Tests

Run tests using pytest:

```bash
pixi run pytest tests/
```

Or with the Python installation:

```bash
pytest tests/
```

### Building the Package

To build the package locally:

```bash
python -m build
```

This will create wheel and source distributions in the `dist/` directory.

## Verifying Installation

After installation, verify the package is installed correctly:

```bash
python -c "import ovro_lwa_portal; print(ovro_lwa_portal.__version__)"
```

## Dependencies

The package requires:

- Python >= 3.12
- astropy >= 7.1.0
- xarray >= 2025.9.1
- dask >= 2025.9.1
- netcdf4 >= 1.7.2
- zarr >= 3.1.3
- numcodecs >= 0.16.1

See `pyproject.toml` for the complete list of dependencies.
