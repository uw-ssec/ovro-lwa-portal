# Loading Data

The OVRO-LWA Portal provides the `open_dataset()` function for loading data from
various sources.

## Basic Usage

```python
import ovro_lwa_portal as ovro

# Load from local zarr store
ds = ovro.open_dataset("/path/to/observation.zarr")
```

Once loaded, the dataset has a `radport` accessor providing over 40 analysis
methods.

## Supported Data Sources

### Local Files

```python
ds = ovro.open_dataset("/path/to/observation.zarr")
```

### Remote URLs

#### S3 (Amazon Web Services)

```python
ds = ovro.open_dataset("s3://bucket-name/observation.zarr")
```

#### HTTPS

```python
ds = ovro.open_dataset("https://example.com/data.zarr")
```

### DOI-based Access

```python
ds = ovro.open_dataset("doi:10.5281/zenodo.1234567")
```

## Chunking

Control memory usage with chunking:

```python
# Automatic chunking (default)
ds = ovro.open_dataset("path/to/data.zarr", chunks="auto")

# Custom chunking
ds = ovro.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50}
)

# No chunking (caution: loads all data into memory)
ds = ovro.open_dataset("path/to/data.zarr", chunks=None)
```

## Dataset Structure

OVRO-LWA datasets typically contain:

- **Dimensions**: `time`, `frequency`, `l`, `m` (and optionally `beam`)
- **Coordinates**: Time stamps, frequency channels, sky coordinates
- **Data Variables**: `intensity`, `SKY`, `BEAM`, etc.
- **Attributes**: Observation metadata

## Accessing the radport Accessor

After loading, use the `.radport` accessor for analysis:

```python
ds = ovro.open_dataset("path/to/data.zarr")

# The radport accessor is now available
ds.radport.plot()
ds.radport.dynamic_spectrum()
ds.radport.find_peaks()
```

See the [API Reference](../api/radport-accessor.md) for all available methods.

## Next Steps

- Learn about [basic plotting](basic-plotting.md)
- Understand [coordinate systems](coordinate-systems.md)
- Explore [visualization methods](../user-guide/visualization.md)
