# OVRO-LWA FITS to Zarr Ingest Module

The `ovro_lwa_portal.ingest` module provides tools for converting OVRO-LWA FITS
image files into optimized Zarr data stores. It supports both command-line usage
via the `ovro-ingest` CLI and programmatic Python API access.

## Features

- **CLI Interface**: User-friendly command-line tool with progress tracking
- **Incremental Processing**: Append new time steps to existing Zarr stores
- **WCS Preservation**: Maintains celestial coordinates (RA/Dec) for FITS-free
  plotting
- **Automatic FITS Fixing**: Corrects BSCALE/BZERO and missing WCS keywords
- **Concurrent Write Protection**: File locking prevents simultaneous writes
- **Optional Prefect Integration**: Workflow orchestration with retry logic
- **Configurable Chunking**: Optimize spatial chunking for your access patterns

## Installation

### Basic Installation (CLI + Core API)

```bash
# Using pip
pip install ovro_lwa_portal

# Using uv
uv pip install ovro_lwa_portal
```

### With Prefect Support (Optional)

```bash
# Using pip
pip install 'ovro_lwa_portal[prefect]'

# Using uv
uv pip install 'ovro_lwa_portal[prefect]'
```

## Quick Start

### Command-Line Usage

After installation, the `ovro-ingest` command will be available:

```bash
# Basic conversion
ovro-ingest convert /path/to/fits /path/to/output

# Rebuild existing store with verbose logging
ovro-ingest convert /path/to/fits /path/to/output --rebuild --log-level debug

# Custom Zarr name and chunk size
ovro-ingest convert /path/to/fits /path/to/output \
    --zarr-name my_data.zarr \
    --chunk-lm 2048

# Show help
ovro-ingest convert --help

# Show version
ovro-ingest --version
```

### Python API Usage

#### Core API (Framework-Independent)

```python
from pathlib import Path
from ovro_lwa_portal.ingest import FITSToZarrConverter
from ovro_lwa_portal.ingest.core import ConversionConfig

# Create configuration
config = ConversionConfig(
    input_dir=Path("/path/to/fits"),
    output_dir=Path("/path/to/output"),
    zarr_name="ovro_lwa_data.zarr",
    chunk_lm=1024,
    rebuild=False,
    verbose=True,
)

# Optional: Define progress callback
def progress_callback(stage, current, total, message):
    print(f"[{stage}] {message} ({current}/{total})")

# Execute conversion
converter = FITSToZarrConverter(config, progress_callback=progress_callback)
result = converter.convert()
print(f"Created: {result}")
```

#### With Prefect Orchestration (Optional)

```python
from ovro_lwa_portal.ingest.prefect_workflow import fits_to_zarr_flow

# Run as a Prefect flow with automatic retries and logging
result = fits_to_zarr_flow(
    input_dir="/path/to/fits",
    output_dir="/path/to/output",
    zarr_name="ovro_lwa_data.zarr",
    rebuild=False,
)
```

## CLI Options

### `ovro-ingest convert`

| Option              | Type   | Default                      | Description                                 |
| ------------------- | ------ | ---------------------------- | ------------------------------------------- |
| `input_dir`         | Path   | Required                     | Directory containing input FITS files       |
| `output_dir`        | Path   | Required                     | Directory for output Zarr store             |
| `--zarr-name`, `-z` | str    | `ovro_lwa_full_lm_only.zarr` | Name of output Zarr store                   |
| `--fixed-dir`, `-f` | Path   | `OUTPUT_DIR/fixed_fits`      | Directory for fixed FITS files              |
| `--chunk-lm`, `-c`  | int    | 1024                         | Chunk size for l,m dimensions (0=disable)   |
| `--rebuild`, `-r`   | flag   | False                        | Overwrite existing Zarr store               |
| `--log-level`, `-l` | choice | info                         | Logging level (debug, info, warning, error) |

## File Naming Convention

Input FITS files must follow the OVRO-LWA naming pattern:

```
YYYYMMDD_HHMMSS_<FREQ>MHz_averaged_*-I-image.fits
```

Examples:

- `20240524_050019_41MHz_averaged_v1-I-image.fits`
- `20240524_050019_59MHz_averaged_v1-I-image.fits`

Files are automatically grouped by observation time (`YYYYMMDD_HHMMSS`) and
sorted by frequency.

## Architecture

### Module Structure

```
ovro_lwa_portal/ingest/
├── __init__.py              # Public API exports
├── core.py                  # Framework-independent conversion logic
├── cli.py                   # Typer-based CLI interface
├── prefect_workflow.py      # Optional Prefect orchestration
└── README.md               # This file
```

### Design Principles

1. **Separation of Concerns**: Core conversion logic is independent of CLI and
   orchestration frameworks
2. **Optional Dependencies**: Prefect is optional; core functionality works
   without it
3. **Progress Reporting**: Callback-based progress tracking works with any UI
4. **Error Handling**: Clear, actionable error messages for common issues

## Advanced Usage

### Custom Progress Tracking

Implement a custom progress callback for integration with your UI:

```python
from ovro_lwa_portal.ingest import FITSToZarrConverter
from ovro_lwa_portal.ingest.core import ConversionConfig

class ProgressTracker:
    def __call__(self, stage, current, total, message):
        # Integrate with your UI framework
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"[{stage}] {percentage:.1f}% - {message}")

config = ConversionConfig(
    input_dir="/path/to/fits",
    output_dir="/path/to/output",
)

tracker = ProgressTracker()
converter = FITSToZarrConverter(config, progress_callback=tracker)
result = converter.convert()
```

### Handling Concurrent Writes

The converter uses file locking to prevent concurrent writes:

```python
try:
    converter = FITSToZarrConverter(config)
    result = converter.convert()
except RuntimeError as e:
    if "lock" in str(e).lower():
        print("Another process is writing to this output location")
    else:
        raise
```

### Integration with Prefect Cloud

Deploy the Prefect flow to Prefect Cloud for distributed execution:

```python
from ovro_lwa_portal.ingest.prefect_workflow import fits_to_zarr_flow

# Deploy to Prefect Cloud
fits_to_zarr_flow.deploy(
    name="ovro-lwa-ingest-prod",
    work_pool_name="kubernetes-pool",
    cron="0 2 * * *",  # Daily at 2 AM
)
```

## Output Format

The resulting Zarr store contains:

- **Dimensions**: `(time, frequency, polarization, l, m)`
- **Coordinates**:
  - `time`: Observation timestamps
  - `frequency`: Frequency values in Hz
  - `l`, `m`: Spatial pixel coordinates
  - `right_ascension`, `declination`: 2D celestial coordinates (degrees,
    FK5/J2000)
- **Data Variables**: Intensity values (e.g., `SKY`, `BEAM`)
- **Metadata**: WCS header for FITS-free plotting

### Reading the Output

```python
import xarray as xr

# Load the Zarr store
ds = xr.open_zarr("/path/to/output/ovro_lwa_full_lm_only.zarr")

# Access data
print(ds)
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Frequency range: {ds.frequency.values.min():.2e} to {ds.frequency.values.max():.2e} Hz")

# Plot with WCS coordinates
import matplotlib.pyplot as plt
from astropy.wcs import WCS

# Reconstruct WCS from stored header
wcs_header_str = ds.attrs.get('fits_wcs_header') or ds.wcs_header_str.item().decode('utf-8')
wcs = WCS(wcs_header_str)

# Create WCS-aware plot
fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(ds.SKY.isel(time=0, frequency=0, polarization=0).values)
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
plt.show()
```

## Troubleshooting

### No FITS Files Found

**Error**: `FileNotFoundError: No matching FITS found in /path/to/input`

**Solution**:

- Verify files exist in the input directory
- Check files follow the naming pattern: `YYYYMMDD_HHMMSS_*MHz_*-I-image.fits`
- Use `--log-level debug` to see which files are being discovered

### Lock Acquisition Failed

**Error**:
`RuntimeError: Cannot acquire lock on .ovro_lwa_full_lm_only.zarr.lock`

**Solution**:

- Check if another conversion process is running
- Remove stale lock file: `rm /path/to/output/.ovro_lwa_full_lm_only.zarr.lock`

### Spatial Grid Mismatch

**Error**:
`RuntimeError: l/m grids differ across times; aborting to avoid misalignment`

**Solution**:

- Verify all FITS files have the same spatial grid (l, m coordinates)
- Check FITS headers for consistent NAXIS1, NAXIS2, CDELT1, CDELT2, CRPIX1,
  CRPIX2

### Memory Issues

For very large datasets, reduce memory usage:

- Increase `chunk_lm` value: `--chunk-lm 2048`
- Process fewer time steps per run
- Ensure sufficient disk space in the output directory

## Testing

Run the test suite:

```bash
# All tests
pytest tests/ingest/

# Core conversion tests
pytest tests/ingest/test_core.py

# CLI tests
pytest tests/ingest/test_cli.py

# With coverage
pytest tests/ingest/ --cov=ovro_lwa_portal.ingest
```

## Contributing

When contributing to the ingest module:

1. Follow the existing code style (ruff formatting)
2. Add type hints to all functions
3. Write tests for new features
4. Update this README for user-facing changes
5. Ensure core logic remains framework-independent

## References

- [OVRO-LWA Portal Documentation](https://github.com/uw-ssec/ovro-lwa-portal)
- [xradio Documentation](https://xradio.readthedocs.io/)
- [Zarr Documentation](https://zarr.readthedocs.io/)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Prefect Documentation](https://docs.prefect.io/)
