# Contributing to OVRO-LWA Portal

Thank you for your interest in contributing to **OVRO-LWA Portal**! This guide
provides step-by-step instructions to set up the project locally. Follow these
guidelines to get started.

Read our
[Code of Conduct](https://github.com/uw-ssec/code-of-conduct/blob/main/CODE_OF_CONDUCT.md)
to keep our community approachable and respectable.

## Pull Requests

We welcome contributions! Please follow these guidelines when submitting a Pull
Request:

- It may be helpful to review
  [this tutorial](https://www.dataschool.io/how-to-contribute-on-github/) on how
  to contribute to open source projects. A typical task workflow is:

  - [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the
    code repository specified in the task and
    [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
    it locally.
  - Review the repo's README.md and CONTRIBUTING.md files to understand what is
    required to run and modify this code.
  - Create a branch in your local repo to implement the task.
  - Commit your changes to the branch and push it to the remote repo.
  - Create a pull request, adding the task owner as the reviewer.

- Please follow the
  [Conventional Commits](https://github.com/uw-ssec/rse-guidelines/blob/main/conventional-commits.md)
  naming for pull request titles.

Your contributions make this project better—thank you for your support! 🚀

## Development

### Prerequisites

#### Install Pixi

Pixi is used in this project to manage dependencies and environments and to set
up some convenient ways to run code and tools within those environments.

To get started, install [Pixi](https://pixi.sh/latest/) using either the
[instructions on their website](https://pixi.sh/latest/#installation), or the
commands below:

**macOS/Linux:**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

### Configure pre-commit

PRs will fail style and formatting checks as configured by
[pre-commit](https://pre-commit.com/), but you can set up your local repository
such that precommit runs every time you commit. This way, you can fix any errors
before you send out pull requests!

#### Configure pre-commit to run on every commit

Then, once Pixi is installed, run the following command to set up pre-commit
checks on every commit

```bash
pixi run pre-commit-install
```

#### Manually run pre-commit on non-committed files

```bash
pixi run pre-commit
```

#### Manually run pre-commit on all files

```bash
pixi run pre-commit-all
```

### Access `ssec` CLI

The `ssec` CLI contains some convenience functions for setting up and working
with this repository. More information about the tool can be found here:
<https://github.com/uw-ssec/ssec-cli>

#### Set up autocompletions

```bash
pixi run ssec-setup
```

#### Run `ssec` command

```bash
pixi run ssec <options>
```

#### Run `ssec` command with autocompletion

Open Pixi shell

```bash
pixi shell
```

Start typing :)

```bash
ssec <tab>
```

## Working with Radio Astronomy Data

This project focuses on radio astronomy data processing. When contributing:

### Testing with Sample Data

Use the provided test FITS files in `notebooks/test_fits_files/` for development
and testing. These files represent typical OVRO-LWA observations.

### Jupyter Notebooks

- Notebooks are primarily for data exploration and analysis
- Keep notebooks clean and well-documented
- Use `nbstripout` or similar tools to remove output before committing

### Performance Considerations

- Use `dask` for large array operations that don't fit in memory
- Consider chunking strategies for Zarr arrays based on access patterns
- Profile memory usage when working with large datasets

## Project-Specific Guidelines

### Image Plane Correction

When working with the `image-plane-correction` package:

- This is an external dependency from the OVRO-LWA team
- Currently using the `nikita/dev` branch
- Report issues to the upstream repository when appropriate

### BDSF Integration

For macOS ARM64 users:

- A pre-compiled wheel is provided to avoid compilation issues
- If you encounter problems, check the `pixi.toml` configuration
- Linux users should use the standard PyPI package

### Documentation

- Follow NumPy-style docstrings for functions and classes
- Include examples in docstrings when helpful
- Update README.md for significant changes to workflow or setup
