# Contributing

Thank you for your interest in contributing to OVRO-LWA Portal! This guide covers
how to set up the project locally, run tests, and submit changes.

Please review our [Code of Conduct](https://github.com/uw-ssec/code-of-conduct/blob/main/CODE_OF_CONDUCT.md) before contributing.

## Pull Requests

- Fork the repository and create a branch for your changes
- Follow [Conventional Commits](https://github.com/uw-ssec/rse-guidelines/blob/main/docs/fundamentals/conventional-commits.md) for PR titles
- See [this tutorial](https://www.dataschool.io/how-to-contribute-on-github/) for a general guide to open-source contributions

## Setting Up Your Environment

### Prerequisites

This project uses [Pixi](https://pixi.sh) for dependency management.

=== "macOS / Linux"

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
    ```

### Clone and Install

=== "Pixi (Recommended)"

    ```bash
    git clone https://github.com/uw-ssec/ovro-lwa-portal.git
    cd ovro-lwa-portal
    pixi install
    pixi run pre-commit-install
    ```

=== "pip"

    ```bash
    git clone https://github.com/uw-ssec/ovro-lwa-portal.git
    cd ovro-lwa-portal
    pip install -e ".[dev]"
    ```

### Verify Installation

```bash
python -c "import ovro_lwa_portal; print(ovro_lwa_portal.__version__)"
```

## Running Tests

```bash
# With Pixi
pixi run pytest tests/

# Or directly
pytest tests/
```

## Building the Package

```bash
python -m build
```

This creates wheel and source distributions in the `dist/` directory.

## Pre-commit Hooks

PRs are checked by [pre-commit](https://pre-commit.com/) for style and formatting.
Set up your local repository so checks run automatically on every commit:

```bash
# Install hooks (run once)
pixi run pre-commit-install

# Run manually on staged files
pixi run pre-commit

# Run on all files
pixi run pre-commit-all
```

## Building Documentation

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme.

```bash
# Serve locally with live reload
pixi run mkdocs serve

# Build the static site
pixi run mkdocs build
```

The site will be available at `http://127.0.0.1:8000/ovro-lwa-portal/`.

## Code Style

- **Docstrings**: Use [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings for all public functions and classes
- **PR titles**: Follow [Conventional Commits](https://github.com/uw-ssec/rse-guidelines/blob/main/docs/fundamentals/conventional-commits.md)
- **Formatting**: Enforced automatically by pre-commit hooks

## Working with Radio Astronomy Data

### Test Data

Use the provided test FITS files in `notebooks/test_fits_files/` for development
and testing. These represent typical OVRO-LWA observations.

### Jupyter Notebooks

- Notebooks in `notebooks/` are for data exploration and analysis
- Keep notebooks clean and well-documented
- Use `nbstripout` or similar tools to remove output before committing

### Performance

- Use `dask` for large array operations that don't fit in memory
- Consider chunking strategies for Zarr arrays based on access patterns
- Profile memory usage when working with large datasets

## Project-Specific Notes

### Image Plane Correction

- The `image-plane-correction` package is an external dependency from the OVRO-LWA team
- Currently using the `nikita/dev` branch
- Report issues to the upstream repository when appropriate

### BDSF Integration

For macOS ARM64 users:

- A pre-compiled wheel is provided to avoid compilation issues
- If you encounter problems, check the `pyproject.toml` configuration in the `[tool.pixi]` section
- Linux users should use the standard PyPI package

## ssec CLI

The `ssec` CLI provides convenience functions for working with this repository.
See <https://github.com/uw-ssec/ssec-cli> for details.

```bash
# Set up autocompletions
pixi run ssec-setup

# Run ssec commands
pixi run ssec <options>
```
