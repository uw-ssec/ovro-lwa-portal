# Loading Data

The `open_dataset()` function is the primary interface for loading OVRO-LWA data. It provides a unified way to access data from various sources.

## Basic Usage

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")
```

## Supported Data Sources

### Local Files

Load data from your local filesystem:

```python
ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")
```

### Remote URLs

#### S3 (Amazon Web Services)

```python
ds = ovro_lwa_portal.open_dataset("s3://bucket-name/observation.zarr")
```

#### HTTPS

```python
ds = ovro_lwa_portal.open_dataset("https://example.com/data.zarr")
```

#### GCS (Google Cloud Storage)

```python
ds = ovro_lwa_portal.open_dataset("gs://bucket-name/observation.zarr")
```

### DOI-based Access

Load datasets published with DOI identifiers:

```python
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
```

The function will automatically resolve the DOI and download the data.

## Chunking Strategies

Chunking controls how data is loaded into memory. Choose the right strategy for your use case:

### Automatic Chunking (Recommended)

Let xarray decide the optimal chunk sizes:

```python
ds = ovro_lwa_portal.open_dataset("path/to/data.zarr", chunks="auto")
```

This is the **default behavior** and works well for most cases.

### Custom Chunking

Specify chunk sizes for specific dimensions:

```python
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50, "l": 512, "m": 512}
)
```

Use custom chunking when:

- You know your access patterns
- You want to optimize for specific operations
- You need to control memory usage

### No Chunking

Load all data into memory at once:

```python
ds = ovro_lwa_portal.open_dataset("path/to/data.zarr", chunks=None)
```

!!! warning
    Use `chunks=None` only for small datasets that fit in memory.

## Storage Options

For cloud storage, you may need to provide authentication credentials:

```python
storage_options = {
    "key": "your-access-key",
    "secret": "your-secret-key"
}

ds = ovro_lwa_portal.open_dataset(
    "s3://bucket/data.zarr",
    storage_options=storage_options
)
```

## Return Value

The function returns an `xarray.Dataset` with the following typical structure:

- **Dimensions**: `time`, `frequency`, `l`, `m`
- **Coordinates**: Time stamps, frequency channels, sky coordinates
- **Data Variables**: Intensity, Stokes parameters, etc.
- **Attributes**: Metadata about the observation

## Examples

### Load and Inspect Data

```python
import ovro_lwa_portal

# Load dataset
ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")

# Inspect structure
print(ds)

# Access data variables
intensity = ds['intensity']

# Select subset
subset = ds.sel(time="2024-01-01", frequency=slice(30e6, 50e6))
```

### Work with Remote Data

```python
# Load from S3
ds = ovro_lwa_portal.open_dataset(
    "s3://ovro-lwa-public/obs_12345.zarr",
    chunks={"time": 50, "frequency": 100}
)

# Compute statistics on chunked data
mean_intensity = ds['intensity'].mean(dim='time').compute()
```
