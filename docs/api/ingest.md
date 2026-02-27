# Ingest Module

The `ovro_lwa_portal.ingest` package provides tools for converting OVRO-LWA FITS
image files to cloud-optimized Zarr format, with support for incremental
processing, WCS coordinate preservation, and concurrent write protection.

## CLI Reference

The `ovro-ingest` command-line tool is installed automatically with the package.

### `ovro-ingest convert`

Convert FITS files to a single Zarr store:

```
ovro-ingest convert INPUT_DIR OUTPUT_DIR [OPTIONS]
```

| Option                 | Short | Default                      | Description                                   |
| ---------------------- | ----- | ---------------------------- | --------------------------------------------- |
| `--zarr-name`          | `-z`  | `ovro_lwa_full_lm_only.zarr` | Name of output Zarr store                     |
| `--fixed-dir`          | `-f`  | `OUTPUT_DIR/fixed_fits`      | Directory for fixed FITS files                |
| `--chunk-lm`           | `-c`  | `1024`                       | Chunk size for l/m dimensions (0 to disable)  |
| `--rebuild`            | `-r`  | `False`                      | Overwrite existing store instead of appending |
| `--skip-header-fixing` | `-s`  | `False`                      | Skip header fixing (assume pre-fixed)         |
| `--log-level`          | `-l`  | `info`                       | Logging level (debug/info/warning/error)      |

### `ovro-ingest fix-headers`

Fix FITS headers as a separate step before conversion:

```
ovro-ingest fix-headers INPUT_DIR FIXED_DIR [OPTIONS]
```

| Option                        | Short | Default           | Description                             |
| ----------------------------- | ----- | ----------------- | --------------------------------------- |
| `--skip-existing/--overwrite` |       | `--skip-existing` | Skip files with existing fixed versions |
| `--log-level`                 | `-l`  | `info`            | Logging level                           |

### Examples

```bash
# Basic conversion (fixes headers on-demand)
ovro-ingest convert /data/fits /data/output

# Rebuild with verbose logging
ovro-ingest convert /data/fits /data/output --rebuild --log-level debug

# Custom Zarr name and chunk size
ovro-ingest convert /data/fits /data/output \
    --zarr-name my_data.zarr --chunk-lm 2048

# Two-step workflow: fix headers first, then convert
ovro-ingest fix-headers /data/fits /data/fixed_fits
ovro-ingest convert /data/fits /data/output \
    --fixed-dir /data/fixed_fits --skip-header-fixing
```

## Python API

### FITSToZarrConverter

::: ovro_lwa_portal.ingest.FITSToZarrConverter options: show_root_heading: true
show_root_full_path: false members_order: source

### ConversionConfig

::: ovro_lwa_portal.ingest.ConversionConfig options: show_root_heading: true
show_root_full_path: false members_order: source

### ProgressCallback

::: ovro_lwa_portal.ingest.ProgressCallback options: show_root_heading: true
show_root_full_path: false

## Optional Prefect Integration

For orchestrated workflows, the package includes optional
[Prefect](https://www.prefect.io/)-based workflow orchestration with automatic
retries, logging, and monitoring.

### Installation

```bash
pip install 'ovro_lwa_portal[prefect]'
```

### Usage

```python
from ovro_lwa_portal.ingest.prefect_workflow import run_conversion_flow

result = run_conversion_flow(
    input_dir="/data/fits",
    output_dir="/data/output",
    rebuild=False,
)
```

### run_conversion_flow

::: ovro_lwa_portal.ingest.prefect_workflow.run_conversion_flow options:
show_root_heading: true show_root_full_path: false

### fits_to_zarr_flow

`fits_to_zarr_flow` is the underlying Prefect `@flow`-decorated function called
by `run_conversion_flow`. It accepts the same parameters (`input_dir`,
`output_dir`, `zarr_name`, `fixed_dir`, `chunk_lm`, `rebuild`, `verbose`) and
orchestrates three Prefect tasks in sequence: configuration validation,
directory preparation, and the conversion itself (with automatic retries).

!!! note

    `fits_to_zarr_flow` is conditionally defined depending on whether Prefect is
    installed. Use `run_conversion_flow` as the stable entry point — it checks
    for Prefect availability and provides a clear error message if it is missing.
