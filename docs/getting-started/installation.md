# Installation

## Prerequisites

This project uses [Pixi](https://pixi.sh) for dependency management and task
execution. Install Pixi by following the
[installation instructions](https://pixi.sh/latest/#installation).

## For Users

Install the package using pip:

```bash
pip install git+https://github.com/uw-ssec/ovro-lwa-portal.git
```

Or install from a local clone:

```bash
pip install .
```

### Optional Dependencies

For remote data access (S3, GCS, HTTP):

```bash
pip install 'ovro_lwa_portal[remote]'
```

For Prefect workflow orchestration:

```bash
pip install 'ovro_lwa_portal[prefect]'
```

## For Developers

### 1. Clone the Repository

```bash
git clone https://github.com/uw-ssec/ovro-lwa-portal.git
cd ovro-lwa-portal
```

### 2. Install Dependencies

Install dependencies using Pixi:

```bash
pixi install
```

Pixi will automatically create the environment and install all required
dependencies.

### 3. Onboarding

For first-time setup, use the onboarding environment to configure your
development environment:

```bash
pixi run -e onboard onboard
```

This will:

- Install pre-commit hooks in your git repository
- Set up shell completion for ssec-cli
- Run the SSEC onboarding process

### 4. Available Environments

- **`default`**: Standard development environment with pre-commit hooks and
  GitHub CLI
- **`onboard`**: Extended environment including onboarding tools

Switch between environments as needed:

```bash
# Use default environment
pixi shell

# Use onboard environment
pixi shell -e onboard
```

## Verifying Installation

Test your installation:

```python
import ovro_lwa_portal
print(ovro_lwa_portal.__version__)
```

Or from the command line:

```bash
ovro-ingest --help
```
