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
- **Size:** Medium (~50+ files including notebooks and data)
- **Languages:** Python (primary), Jupyter Notebooks, configuration files
- **Build System:** Pixi (v0.49.0+)
- **Platform:** macOS (osx-arm64), Linux (linux-64)
- **License:** BSD 3-Clause
- **Domain:** Radio astronomy, data processing, visualization

## Build System & Environment Management

### Pixi Overview

This project uses **Pixi** exclusively for dependency and environment
management. Pixi is a modern package manager that handles both Conda and PyPI
dependencies. See <https://pixi.sh/latest/llms-full.txt> for more details.

**ALWAYS use Pixi commands—never use conda, pip, or venv directly.**

### Prerequisites

Before any other operations, verify Pixi is installed:

```bash
pixi --version  # Should show v0.49.0 or higher
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
# - OVRO-LWA specific packages (image-plane-correction, bdsf)
# - Creates .pixi/envs/default directory
```

**CRITICAL:** Always run `pixi install` before any other Pixi commands. This
command is idempotent and safe to run multiple times.

### Available Environments

1. **`default`** (features: `pre-commit`, `gh-cli`)

   - Standard development environment
   - Radio astronomy packages: astropy, xarray, dask, zarr, netcdf4, numcodecs
   - OVRO-LWA specific: image-plane-correction, bdsf (macOS ARM64)
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

## Project Structure & Key Files

```text
.
├── .github/
│   ├── dependabot.yml           # Dependabot config for GitHub Actions
│   ├── pull_request_template.md # PR template (requires pre-commit checks)
│   ├── release.yml              # Release notes configuration
│   └── ISSUE_TEMPLATE/          # Issue templates (bug, feature, docs, onboard, etc.)
├── .pre-commit-config.yaml      # Pre-commit hook configuration
├── pyproject.toml               # **PRIMARY CONFIG**: Dependencies, tasks, features (includes [tool.pixi] section)
├── pixi.lock                    # Lock file (auto-generated, don't manually edit)
├── .gitignore                   # Ignores .pixi/ and .DS_Store
├── .gitattributes               # Git attributes for file handling
├── CODE_OF_CONDUCT.md           # Contributor Covenant v2.0
├── CONTRIBUTING.md              # Contribution guidelines (references Conventional Commits)
├── LICENSE                      # BSD 3-Clause License
├── README.md                    # Project documentation
├── onboarded.md                 # Empty file (excluded from pre-commit)
├── notebooks/                   # Jupyter notebooks for data analysis
│   ├── fits2zarr.ipynb         # Main FITS to Zarr conversion notebook
└── └── test_fits_files/        # Sample FITS files for testing
```

## Radio Astronomy Context

This project works specifically with:

- **FITS files**: Standard astronomical image format from OVRO-LWA observations
- **Zarr format**: Cloud-optimized array storage for large datasets

## Continuous Integration & Validation

**Current State:** This repository has **no GitHub Actions workflows** or CI
pipelines defined. Pre-commit checks run locally only.

**Pre-commit.ci Integration:** The `.pre-commit-config.yaml` includes a `ci:`
section, suggesting integration with <https://pre-commit.ci> for automated PR
checks. Verify if enabled on the repository.

**Dependabot:** Configured to update GitHub Actions weekly (groups all action
updates together).

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

### Issue: BDSF installation fails on macOS

**Solution:** The project includes a pre-compiled wheel for macOS ARM64. If you
encounter issues:

```bash
# Force reinstall
rm -rf .pixi
pixi install
```

### Issue: Modifying pyproject.toml breaks the environment

**Solution:**

```bash
# Validate syntax, reinstall environment
pixi install

# If still broken, remove and reinstall
rm -rf .pixi
pixi install
```

### Issue: Platform-specific problems (non-supported platforms)

**Current platforms:** `osx-arm64`, `linux-64`

**Solution:** Edit `pyproject.toml` in the `[tool.pixi.workspace]` section and add platforms:

```toml
[tool.pixi.workspace]
platforms = ["osx-arm64", "linux-64", "win-64"]
```

Then run `pixi install`.

## Key Configuration Details

### Pixi Configuration in pyproject.toml

Pixi configuration is now embedded in `pyproject.toml` under the `[tool.pixi]` section:

- **`[tool.pixi.workspace]`**: Project metadata (name, version, authors, platforms)
- **`[tool.pixi.environments]`**: Named environments with feature sets
- **`[tool.pixi.dependencies]`**: Conda dependencies for all environments (Python,
  astropy, xarray, etc.)
- **`[tool.pixi.pypi-dependencies]`**: PyPI dependencies (image-plane-correction)
- **`[tool.pixi.target.osx-arm64.pypi-dependencies]`**: Platform-specific packages (bdsf)
- **`[tool.pixi.feature.<name>.dependencies]`**: Feature-specific conda packages
- **`[tool.pixi.feature.<name>.pypi-dependencies]`**: Feature-specific PyPI packages
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
