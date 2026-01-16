# Loading OVRO-LWA Data with `open_dataset()`

The `open_dataset()` function provides a unified interface for loading OVRO-LWA
data from multiple sources, including local file paths, remote URLs, and DOI
identifiers.

## Installation

### Basic Installation

The core `open_dataset()` function works with local files using the base
installation:

```bash
pip install ovro_lwa_portal
```

### Remote Data Access

For remote data access (S3, Google Cloud Storage, DOI resolution), install with
the `remote` extras:

```bash
pip install 'ovro_lwa_portal[remote]'
```

This installs additional dependencies:

- `requests` - For DOI resolution
- `s3fs` - For AWS S3 access
- `gcsfs` - For Google Cloud Storage access
- `fsspec` - Unified filesystem interface
- `caltechdata_api` - Caltech Data API integration

## Quick Start

### Load from Local Path

```python
import ovro_lwa_portal

# Load a local zarr store
ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

# Access data
print(ds)
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Frequency range: {ds.frequency.values.min():.2e} to {ds.frequency.values.max():.2e} Hz")
```

### Load from Remote URL

```python
# Load from HTTPS
ds = ovro_lwa_portal.open_dataset("https://data.ovro.caltech.edu/obs_12345.zarr")

# Load from S3 (requires s3fs)
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

# Load from Google Cloud Storage (requires gcsfs)
ds = ovro_lwa_portal.open_dataset("gs://ovro-lwa-data/obs_12345.zarr")
```

### Load via DOI

```python
# With 'doi:' prefix
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")

# Without prefix
ds = ovro_lwa_portal.open_dataset("10.5281/zenodo.1234567")
```

## Function Signature

```python
def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    engine: str = "zarr",
    validate: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
```

### Parameters

- **source** (str or Path): Data source, can be:
  - Local file path (e.g., `"/path/to/data.zarr"`)
  - Remote URL (e.g., `"s3://bucket/data.zarr"`, `"https://..."`)
  - DOI string (e.g., `"doi:10.xxxx/xxxxx"` or `"10.xxxx/xxxxx"`)

- **chunks** (dict, str, or None, default `"auto"`): Chunking strategy for lazy
  loading:
  - `dict`: Explicit chunk sizes per dimension, e.g.,
    `{"time": 100, "frequency": 50}`
  - `"auto"`: Let xarray/dask determine optimal chunks (recommended)
  - `None`: Load entire dataset into memory (not recommended for large data)

- **engine** (str, default `"zarr"`): Backend engine for loading data. Currently
  supports `"zarr"`.

- **validate** (bool, default `True`): If True, validate that loaded data
  conforms to OVRO-LWA data model.

- **kwargs**: Additional arguments passed to the underlying loader (e.g.,
  `xr.open_zarr`)

### Returns

- **xr.Dataset**: OVRO-LWA dataset with standardized structure

## Usage Examples

### Custom Chunking

For large datasets, customize chunking to optimize memory usage and performance:

```python
# Explicit chunk sizes
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50, "l": 512, "m": 512}
)

# Auto chunking (recommended for most cases)
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks="auto"
)

# Load entire dataset into memory (small datasets only)
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks=None
)
```

### Disable Validation

Skip validation for faster loading when you're confident about the data
structure:

```python
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    validate=False
)
```

### Working with Cloud Storage

#### AWS S3

Set up AWS credentials via environment variables or `~/.aws/credentials`:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

Then load data:

```python
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/observation.zarr")
```

#### Google Cloud Storage

Set up GCS credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

Then load data:

```python
ds = ovro_lwa_portal.open_dataset("gs://ovro-lwa-data/observation.zarr")
```

### DOI Resolution

The function automatically resolves DOIs to data URLs using the Caltech Data
API:

```python
# DOI resolution happens automatically
ds = ovro_lwa_portal.open_dataset("doi:10.22002/abc123")

# The function:
# 1. Detects the DOI format
# 2. Resolves it to a data URL via Caltech Data API
# 3. Loads the data from the resolved URL
```

## Data Validation

By default, `open_dataset()` validates that the loaded data conforms to the
OVRO-LWA data model:

- **Expected dimensions**: `time`, `frequency`, `l`, `m`
- **Expected variables**: `SKY`, `BEAM`

If validation fails, warnings are logged but the dataset is still returned. This
allows working with non-standard or experimental data formats.

To disable validation:

```python
ds = ovro_lwa_portal.open_dataset("path/to/data.zarr", validate=False)
```

## Error Handling

The function provides clear error messages for common issues:

```python
from ovro_lwa_portal.io import DataSourceError

try:
    ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except DataSourceError as e:
    print(f"Failed to load data: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

### Common Errors

1. **FileNotFoundError**: Local path doesn't exist

   ```
   FileNotFoundError: Local path does not exist: /path/to/data.zarr
   ```

2. **ImportError**: Missing dependency for remote access

   ```
   ImportError: s3fs is required for S3 access. Install with: pip install s3fs
   ```

3. **DataSourceError**: Failed to load or resolve data
   ```
   DataSourceError: Failed to resolve DOI 10.xxxx/xxxxx: ...
   ```

## Performance Tips

### Lazy Loading

By default, `open_dataset()` uses lazy loading with dask, which means:

- Data is not loaded into memory immediately
- Only the chunks you access are loaded
- You can work with datasets larger than available RAM

```python
# Data is not loaded yet
ds = ovro_lwa_portal.open_dataset("large_dataset.zarr")

# Only this subset is loaded into memory
subset = ds.sel(time=slice(0, 10), frequency=slice(0, 5))
result = subset.SKY.mean().compute()
```

### Optimal Chunking

Choose chunk sizes based on your access patterns:

```python
# For time-series analysis (accessing many time steps)
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 1000, "frequency": 10, "l": 256, "m": 256}
)

# For spatial analysis (accessing full images)
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 1, "frequency": 1, "l": 1024, "m": 1024}
)
```

### Caching Remote Data

For frequently accessed remote data, consider caching locally:

```python
import fsspec

# Use fsspec caching
with fsspec.open_local(
    "simplecache::s3://bucket/data.zarr",
    s3={"anon": False},
    simplecache={"cache_storage": "/tmp/cache"}
) as local_path:
    ds = ovro_lwa_portal.open_dataset(local_path)
```

## Integration with Analysis Workflows

### Basic Analysis

```python
import ovro_lwa_portal
import matplotlib.pyplot as plt

# Load data
ds = ovro_lwa_portal.open_dataset("observation.zarr")

# Compute mean intensity over time
mean_intensity = ds.SKY.mean(dim="time")

# Plot
plt.figure(figsize=(10, 8))
mean_intensity.isel(frequency=0, polarization=0).plot()
plt.title("Mean Sky Intensity")
plt.show()
```

### WCS-Aware Plotting

```python
from astropy.wcs import WCS
import matplotlib.pyplot as plt

# Load data
ds = ovro_lwa_portal.open_dataset("observation.zarr")

# Reconstruct WCS from stored header
wcs_header_str = ds.attrs.get('fits_wcs_header') or ds.wcs_header_str.item().decode('utf-8')
wcs = WCS(wcs_header_str)

# Create WCS-aware plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(ds.SKY.isel(time=0, frequency=0, polarization=0).values)
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.grid(color='white', ls='dotted')
plt.show()
```

### Parallel Processing

```python
import ovro_lwa_portal
from dask.distributed import Client

# Start dask client for parallel processing
client = Client()

# Load data (lazy)
ds = ovro_lwa_portal.open_dataset("large_dataset.zarr")

# Parallel computation
result = ds.SKY.mean(dim=["l", "m"]).compute()

print(result)
client.close()
```

## Using the `radport` Accessor

After loading a dataset, you can access OVRO-LWA-specific visualization and
analysis features through the `radport` xarray accessor.

### Basic Usage

```python
import ovro_lwa_portal

# Load data
ds = ovro_lwa_portal.open_dataset("observation.zarr")

# Create a default visualization
fig = ds.radport.plot()
```

The accessor is automatically registered when you import `ovro_lwa_portal`, so
it's available on any xarray Dataset that has the required OVRO-LWA structure.

### Plotting Options

The `plot()` method supports various customization options:

```python
# Plot a specific time, frequency, and polarization
fig = ds.radport.plot(
    time_idx=5,      # Time index
    freq_idx=10,     # Frequency index
    pol=0,           # Polarization index
)

# Customize the colormap and scale
fig = ds.radport.plot(
    cmap="viridis",  # Matplotlib colormap
    vmin=-1.0,       # Minimum value for color scale
    vmax=16.0,       # Maximum value for color scale
)

# Use robust scaling for data with outliers
fig = ds.radport.plot(robust=True)

# Customize figure size
fig = ds.radport.plot(figsize=(12, 10))

# Plot without colorbar
fig = ds.radport.plot(add_colorbar=False)
```

### Plot Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `var` | str | `"SKY"` | Variable to plot (`"SKY"` or `"BEAM"`) |
| `time_idx` | int | `0` | Time index for the snapshot |
| `freq_idx` | int | `0` | Frequency index for the snapshot |
| `pol` | int | `0` | Polarization index |
| `freq_mhz` | float | `None` | Select frequency by MHz (overrides `freq_idx`) |
| `time_mjd` | float | `None` | Select time by MJD (overrides `time_idx`) |
| `cmap` | str | `"inferno"` | Matplotlib colormap name |
| `vmin` | float | `None` | Minimum value for color scale |
| `vmax` | float | `None` | Maximum value for color scale |
| `robust` | bool | `False` | Use 2nd/98th percentile for scaling |
| `mask_radius` | int | `None` | Circular mask radius in pixels |
| `figsize` | tuple | `(8, 6)` | Figure size in inches |
| `add_colorbar` | bool | `True` | Whether to add a colorbar |

### Selecting by Value (MHz, MJD)

Instead of using indices, you can select by physical values:

```python
# Select frequency by MHz (more intuitive than index)
fig = ds.radport.plot(freq_mhz=50.0)

# Select time by MJD value
fig = ds.radport.plot(time_mjd=60000.5)

# Combine both
fig = ds.radport.plot(freq_mhz=50.0, time_mjd=60000.5)
```

### Selection Helper Methods

The accessor provides helper methods for finding indices:

```python
# Find index for a specific frequency in MHz
freq_idx = ds.radport.nearest_freq_idx(50.0)  # Returns index nearest to 50 MHz

# Find index for a specific time in MJD
time_idx = ds.radport.nearest_time_idx(60000.5)  # Returns index nearest to MJD

# Find indices for specific (l, m) coordinates
l_idx, m_idx = ds.radport.nearest_lm_idx(0.0, 0.0)  # Returns indices for center
```

### Circular Masking

For all-sky images, edge pixels may be invalid. Use `mask_radius` to apply a
circular mask:

```python
# Mask pixels outside radius of 1800 pixels from center
fig = ds.radport.plot(mask_radius=1800)
```

### Plotting BEAM Data

If your dataset contains BEAM data, you can plot it by specifying the variable:

```python
# Check if BEAM data is available
if ds.radport.has_beam:
    fig = ds.radport.plot(var="BEAM")
```

### Dataset Validation

The accessor automatically validates that the dataset has the required structure
for OVRO-LWA data when you access it. If validation fails, you'll get an
informative error message:

```python
import xarray as xr

# This will raise a ValueError with details about what's missing
invalid_ds = xr.Dataset({"other_var": (["x", "y"], [[1, 2], [3, 4]])})
try:
    invalid_ds.radport.plot()
except ValueError as e:
    print(f"Validation error: {e}")
```

Required dataset structure:
- **Dimensions**: `time`, `frequency`, `polarization`, `l`, `m`
- **Variables**: `SKY` (required), `BEAM` (optional)

### Complete Example

```python
import ovro_lwa_portal

# Load data from DOI
ds = ovro_lwa_portal.open_dataset("doi:10.22002/example")

# Print dataset info
print(f"Time steps: {len(ds.time)}")
print(f"Frequencies: {len(ds.frequency)}")
print(f"Has BEAM data: {ds.radport.has_beam}")

# Create visualization at specific frequency (by value in MHz)
fig = ds.radport.plot(
    freq_mhz=50.0,      # Select 50 MHz directly
    time_idx=0,
    cmap="inferno",
    robust=True,
    mask_radius=1800,   # Apply circular mask
    figsize=(10, 8),
)

# Save the figure
fig.savefig("observation_snapshot.png", dpi=150, bbox_inches="tight")
```

## API Reference

For complete API documentation, see:

- [ovro_lwa_portal.io module](../src/ovro_lwa_portal/io.py)
- [xarray documentation](https://docs.xarray.dev/)
- [dask documentation](https://docs.dask.org/)

## Troubleshooting

### Issue: "s3fs is required for S3 access"

**Solution**: Install the remote extras:

```bash
pip install 'ovro_lwa_portal[remote]'
```

### Issue: "Failed to resolve DOI"

**Possible causes**:

1. DOI doesn't exist or is malformed
2. Network connectivity issues
3. Caltech Data API is unavailable

**Solution**: Check the DOI is correct and try accessing it directly in a
browser.

### Issue: "Dataset may not be OVRO-LWA format"

**Cause**: The loaded dataset doesn't have expected dimensions or variables.

**Solution**: This is just a warning. If you're working with non-standard data,
disable validation:

```python
ds = ovro_lwa_portal.open_dataset("data.zarr", validate=False)
```

### Issue: Memory errors with large datasets

**Solution**: Use smaller chunks or ensure lazy loading is enabled:

```python
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 10, "frequency": 10, "l": 256, "m": 256}
)
```

## See Also

- [FITS to Zarr Conversion](../src/ovro_lwa_portal/ingest/README.md)
- [xarray I/O documentation](https://docs.xarray.dev/en/stable/user-guide/io.html)
- [fsspec documentation](https://filesystem-spec.readthedocs.io/)
- [Caltech Data](https://data.caltech.edu/)
