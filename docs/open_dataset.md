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
