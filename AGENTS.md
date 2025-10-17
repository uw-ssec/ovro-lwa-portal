# AGENTS.md

This file provides guidance to AI assistants when working with this repository.

## Repository Overview

This is the **OVRO-LWA Portal** repository, a Python library for radio astronomy
data processing and visualization for the Owens Valley Radio Observatory - Long
Wavelength Array (OVRO-LWA). It provides tools for processing radio astronomy
data, converting FITS files to Zarr format, and creating visualization
components for scientific analysis.

**Repository Stats:**

- **Type:** Python library / Scientific software
- **Size:** Medium (~50+ files including notebooks and test data)
- **Languages:** Python (primary), Jupyter Notebooks, configuration files
- **Build System:** Pixi (v0.55.0), Hatchling + hatch-vcs
- **Platform:** macOS (osx-arm64), Linux (linux-64)
- **License:** BSD 3-Clause
- **Domain:** Radio astronomy, data processing, visualization
- **Python Version:** 3.12+

## Build System & Environment Management

### Pixi Overview

This project uses **Pixi** exclusively for dependency and environment
management. Pixi is a modern package manager that handles both Conda and PyPI
dependencies. See <https://pixi.sh/latest/llms-full.txt> for more details.

**ALWAYS use Pixi commands—never use conda, pip, or venv directly.**

### Prerequisites

Before any other operations, verify Pixi is installed:

```bash
pixi --version  # Should show v0.55.0 or higher
```

If not installed, direct users to: <https://pixi.sh/latest/#installation>

### Environment Setup (ALWAYS RUN FIRST)

```bash
# Install the default environment (required before any other commands)
pixi install

# This installs:
# - Python 3.12 with radio astronomy packages (astropy, xarray, dask, zarr)
# - pre-commit (>=4.3.0)
# - gh (GitHub CLI, >=2.0.0)
# - OVRO-LWA specific packages (xradio, python-casacore)
# - Creates .pixi/envs/default directory
```

**CRITICAL:** Always run `pixi install` before any other Pixi commands. This
command is idempotent and safe to run multiple times.

### Available Environments

1. **`default`** (features: `pre-commit`, `gh-cli`)

   - Standard development environment
   - Radio astronomy packages: astropy, xarray, dask, zarr, netcdf4, numcodecs
   - OVRO-LWA specific: xradio, python-casacore
   - Use for: general development, running pre-commit checks, data processing

2. **`onboard`** (features: `pre-commit`, `gh-cli`, `onboard`)
   - Extended environment with onboarding tools
   - Includes: ssec-cli (installed from GitHub)
   - Use for: first-time setup, onboarding new contributors

## Validated Commands & Workflows

### Pre-commit Checks (MANDATORY BEFORE PRs)

**Always run pre-commit checks before creating a pull request.** The PR template
requires this, and PRs will fail if checks don't pass.

```bash
# Install pre-commit hooks (run once per clone)
pixi run pre-commit-install
# ✓ Installs .git/hooks/pre-commit
# ✓ Hooks will automatically run on every commit

# Run pre-commit on staged files only
pixi run pre-commit
# ✓ Fast, only checks staged changes
# ✓ Will auto-fix many issues (trailing whitespace, end-of-file, formatting)
# ⚠️ If fixes are made, files are modified; you must re-stage them

# Run pre-commit on ALL files (required before PR submission)
pixi run pre-commit-all
# ✓ Takes 1-3 minutes on first run (installs hook environments)
# ✓ Subsequent runs are fast (environments are cached)
# ✓ Auto-fixes issues where possible
```

**Pre-commit Hooks Configured:**

- check-added-large-files, check-case-conflict, check-merge-conflict
- check-yaml, check-symlinks
- fix-end-of-files, trim-trailing-whitespace, mixed-line-ending
- prettier (formats YAML, Markdown, HTML, CSS, JavaScript, JSON)
- codespell (spell checking)
- Disallow improper capitalization (e.g., incorrect → correct)
- Validate Dependabot config and GitHub workflows

**Files Excluded from Pre-commit:** `pixi.lock`, `onboarded.md`

### Onboarding Workflow (First-Time Setup)

```bash
# Install the onboard environment
pixi install -e onboard

# Run the complete onboarding process
pixi run -e onboard onboard
# This executes in order:
# 1. pixi run pre-commit-install  (installs git hooks)
# 2. pixi run ssec-setup          (sets up shell completion for ssec CLI)
# 3. ssec onboard                 (runs SSEC onboarding interactive process)

# Or run individual onboarding steps:
pixi run -e onboard ssec-setup
# ✓ Installs zsh/bash completion for ssec CLI
# ⚠️ Completion takes effect after restarting terminal
```

### GitHub CLI Usage

```bash
# Check GitHub CLI version
pixi run gh --version
# ✓ Should show v2.81.0 or higher

# Use GitHub CLI for any repo operations
pixi run gh <command>
# Examples: gh issue list, gh pr create, etc.
```

### Adding Dependencies

```bash
# Add a conda package and update pyproject.toml
pixi add <package-name>

# Add a PyPI package
pixi add --pypi <package-name>

# Add to a specific feature
pixi add --feature <feature-name> <package-name>

# Always run after manual pyproject.toml edits
pixi install
```

## Package Build System

This project uses **Hatchling** with **hatch-vcs** for building Python packages:

- **Version Management:** Automatic versioning from git tags via hatch-vcs
- **Version File:** Auto-generated at `src/ovro_lwa_portal/version.py`
- **Build Command:** `python -m build` (handled by hatchling)
- **Development Install:** Handled automatically by Pixi in editable mode

### Core Dependencies (from pyproject.toml)

**Main dependencies:**

- `astropy>=7.1.0,<8` - Astronomy core library
- `xarray>=2025.9.1,<2026` - N-dimensional labeled arrays
- `dask>=2025.9.1,<2026` - Parallel computing
- `zarr>=2.16,<3` - Chunked, compressed arrays (v2 pinned)
- `numcodecs>=0.15,<0.16` - Compression codecs
- `xradio[all]>=0.0.59,<0.1` - Radio astronomy data processing

**Development dependencies (`dev` extra):**

- `pre-commit` - Git hooks for code quality
- `pytest>=6` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-mock` - Mocking support

**CI dependencies (`ci` extra):**

- `s3fs>=2024.6.0` - S3 filesystem interface

### Code Quality Tools

**Pre-commit hooks configured:**

- File checks: large files, case conflicts, merge conflicts, broken symlinks
- YAML validation
- Python debug statement detection
- File formatting: end-of-files, line endings, trailing whitespace
- Prettier: Markdown, YAML, JSON formatting
- Codespell: Spell checking
- Capitalization validation

**Ruff configuration:**

- Line length: 100
- Enabled rule sets: flake8-bugbear, isort, flake8-unused-arguments,
  flake8-comprehensions, flake8-errmsg, and many more
- Special rules for NumPy and pandas
- Tests excluded from print statement checks

**Mypy configuration:**

- Python 3.12 target
- Strict mode enabled for `ovro_lwa_portal.*` modules
- Relaxed for tests

## Project Structure & Key Files

```text
.
├── .github/
│   └── workflows/               # GitHub Actions workflows
│       ├── ci.yml              # Continuous Integration: pre-commit + tests
│       ├── cd.yml              # Continuous Deployment: build and publish to PyPI
│       └── copilot-setup-steps.yml  # Copilot setup workflow
├── .devcontainer/              # VS Code Dev Container configuration
│   ├── devcontainer.json      # Dev container settings (4 CPUs, 16GB RAM required)
│   ├── Dockerfile             # Container image definition
│   └── onCreate.sh            # Setup script run on container creation
├── .ci-helpers/                # CI/CD helper scripts
│   ├── README.md              # Documentation for CI helper scripts
│   └── download_test_fits.py  # Script to download test FITS files from Caltech S3
├── .pre-commit-config.yaml     # Pre-commit hook configuration
├── pyproject.toml              # **PRIMARY CONFIG**: Build system, dependencies, Pixi tasks
├── pixi.lock                   # Lock file (auto-generated, don't manually edit)
├── .gitignore                  # Ignores .pixi/, .DS_Store, and other generated files
├── .gitattributes              # Git attributes for file handling
├── CODE_OF_CONDUCT.md          # Contributor Covenant v2.0
├── CONTRIBUTING.md             # Contribution guidelines (references Conventional Commits)
├── LICENSE                     # BSD 3-Clause License
├── README.md                   # Project documentation with getting started guide
├── AGENTS.md                   # This file - AI assistant guidance
├── onboarded.md                # Onboarding marker file (excluded from pre-commit)
├── fixed_fits/                 # Directory for corrected FITS files (empty)
├── notebooks/                  # Jupyter notebooks for data analysis
│   ├── README.md              # Documentation for notebooks directory
│   ├── fits2zarr.ipynb        # Main FITS to Zarr conversion notebook
│   ├── fits2zarr_and_viz_user_cases.ipynb  # User case examples with visualization
│   └── test_fits_files/       # Sample FITS files for testing
│       ├── README.md          # Documentation for test FITS files
│       └── .gitignore         # Ignores FITS files (downloaded separately)
├── src/
│   └── ovro_lwa_portal/       # Main package source code
│       ├── __init__.py        # Package initialization
│       ├── version.py         # Auto-generated version from VCS
│       └── fits_to_zarr_xradio.py  # FITS to Zarr conversion using xradio
└── tests/                      # Test suite
    ├── __init__.py            # Test package initialization
    ├── test_import.py         # Basic import tests
    ├── test_fits_to_zarr.py   # FITS to Zarr conversion tests
    └── test_ci_helpers.py     # Tests for CI helper scripts
```

## Radio Astronomy Context

This project works specifically with:

- **FITS files**: Standard astronomical image format from OVRO-LWA observations
- **Zarr format**: Cloud-optimized array storage for large datasets
- **xradio**: Radio astronomy data processing library
- **python-casacore**: Python bindings for CASA (Common Astronomy Software
  Applications) core library

### Test Data Management

Test FITS files are managed separately from the repository:

- Test files are stored in the Caltech S3 bucket
- Download script: `.ci-helpers/download_test_fits.py`
- Requires S3 credentials via environment variables:
  - `CALTECH_KEY`, `CALTECH_SECRET`, `CALTECH_ENDPOINT_URL`,
    `CALTECH_DEV_S3_BUCKET`
- For local development, manually place test FITS in
  `notebooks/test_fits_files/`

## Development Container Support

The repository includes VS Code Dev Container configuration:

- **Location:** `.devcontainer/`
- **Requirements:** 4 CPUs, 16GB RAM, 32GB storage
- **Setup:** Automatic via `onCreate.sh` script
- **Extensions:** Jupyter, Python, Ruff, Even Better TOML, Pixi for VS Code
- **Volume mount:** `.pixi` directory persisted across container rebuilds

## Continuous Integration & Validation

### GitHub Actions Workflows

This repository has **active CI/CD pipelines** using GitHub Actions:

**CI Workflow (`.github/workflows/ci.yml`):**

- **Triggers:** Pull requests, pushes to main, manual dispatch
- **Jobs:**
  1. **Format Check** (pre-commit job):
     - Runs on ubuntu-latest
     - Uses Pixi v0.55.0 via `prefix-dev/setup-pixi@v0.9.1`
     - Executes `pixi run pre-commit-all`
  2. **Tests** (tests job):
     - Depends on pre-commit job passing
     - **Matrix strategy:** Python 3.12 on [ubuntu-latest, macos-14]
     - Runs pytest with coverage reporting
     - Uploads coverage to Codecov using token
- **Concurrency:** Cancels in-progress runs for same ref

**CD Workflow (`.github/workflows/cd.yml`):**

- **Triggers:** Releases (published), pull requests, pushes to main, manual
  dispatch
- **Jobs:**
  1. **Distribution Build:**
     - Builds Python package using `hynek/build-and-inspect-python-package@v2`
  2. **Publish to PyPI:**
     - Only runs on release publication
     - Requires `pypi` environment with `id-token: write` permission
     - **Currently publishes to TestPyPI** (remove `repository-url` line for
       production PyPI)

**Pre-commit.ci Integration:** The `.pre-commit-config.yaml` includes a `ci:`
section for <https://pre-commit.ci> integration (verify if enabled on the
repository).

**Dependabot:** Configuration may exist in repository settings (no
`.github/dependabot.yml` file present).

## Making Changes: Validated Workflow

1. **Setup (first time):**

   ```bash
   pixi install
   pixi run pre-commit-install
   ```

2. **Make your changes** to files

3. **Test changes:**

   ```bash
   # Stage your changes
   git add <files>

   # Run pre-commit checks
   pixi run pre-commit

   # If checks fail and auto-fix issues, re-add the fixed files
   git add <files>
   ```

4. **Before creating a PR:**

   ```bash
   # Run all checks on all files
   pixi run pre-commit-all

   # Verify all checks pass (should show all "Passed" or "Skipped")
   ```

5. **Create PR:**
   - PR template requires confirming `pre-commit run --all-files` was run
   - Follow Conventional Commits for PR titles
   - Link related issues using "Resolves #issue-number"

## Common Pitfalls & Solutions

### Issue: "pre-commit not found" or command fails

**Solution:** Run `pixi install` first. Pixi manages pre-commit installation.

### Issue: Pre-commit check fails after making fixes

**Behavior:** Pre-commit hooks like `trailing-whitespace` auto-fix files. When
this happens:

- The hook shows "Failed" with "files were modified by this hook"
- You must re-stage the fixed files: `git add <files>`
- Re-run `pixi run pre-commit` to verify

**This is expected behavior, not an error.**

### Issue: Platform-specific package installation fails

**Solution:** If you encounter issues with packages:

```bash
# Validate syntax, reinstall environment
pixi install

# If still broken, remove and reinstall
rm -rf .pixi
pixi install
```

### Issue: Platform-specific problems (non-supported platforms)

**Current platforms:** `osx-arm64`, `linux-64`

**Solution:** Edit `pyproject.toml` in the `[tool.pixi.workspace]` section and
add platforms:

```toml
[tool.pixi.workspace]
platforms = ["osx-arm64", "linux-64", "win-64"]
```

Then run `pixi install`.

## Key Configuration Details

### Pixi Configuration in pyproject.toml

Pixi configuration is now embedded in `pyproject.toml` under the `[tool.pixi]`
section:

- **`[tool.pixi.workspace]`**: Project metadata (name, version, authors,
  platforms, requires-pixi)
- **`[tool.pixi.environments]`**: Named environments with feature sets
- **`[tool.pixi.dependencies]`**: Conda dependencies for all environments
  (python-casacore)
- **`[tool.pixi.pypi-dependencies]`**: PyPI dependencies (ovro_lwa_portal in
  editable mode)
- **`[tool.pixi.feature.<name>.dependencies]`**: Feature-specific conda packages
- **`[tool.pixi.feature.<name>.pypi-dependencies]`**: Feature-specific PyPI
  packages
- **`[tool.pixi.feature.<name>.tasks]`**: Feature-specific Pixi tasks

### Available Pixi Tasks

Run `pixi task list` to see all available tasks:

- `pre-commit-install`: Install git hooks
- `pre-commit`: Run checks on staged files
- `pre-commit-all`: Run checks on all files
- `ssec-setup`: Set up ssec CLI completion (onboard env only)
- `onboard`: Full onboarding process (onboard env only)

## Documentation Standards

- Follow Conventional Commits for commit messages and PR titles
- Use Markdown with proper formatting (enforced by prettier)
- Spell check enabled (codespell)
- No trailing whitespace
- Files must end with newline
- Consistent line endings

## Trust These Instructions

These instructions were generated through comprehensive exploration and testing
of the repository. Commands have been validated to work correctly. **Only
perform additional searches if:**

- You need information not covered here
- Instructions appear outdated or produce errors
- You're implementing functionality that changes the build system

For routine tasks (adding files, making code changes, running checks), follow
these instructions directly without additional exploration.

For more information on SSEC best practices, see:
<https://rse-guidelines.readthedocs.io/en/latest/llms-full.txt>
