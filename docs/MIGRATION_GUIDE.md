# Migration Guide: Using `open_dataset()`

This guide helps you migrate from direct xarray/zarr loading to the new `open_dataset()` function.

## Why Migrate?

The new `open_dataset()` function provides:
- Unified interface for all data sources (local, remote, DOI)
- Automatic source type detection
- Built-in validation
- Better error messages
- Consistent chunking behavior
- Future-proof API

## Before and After

### Loading Local Files

**Before:**
```python
import xarray as xr

ds = xr.open_zarr("/path/to/data.zarr")
```

**After:**
```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset("/path/to/data.zarr")
```

### Loading with Chunks

**Before:**
```python
import xarray as xr

ds = xr.open_zarr("/path/to/data.zarr", chunks={"time": 100})
```

**After:**
```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset(
    "/path/to/data.zarr",
    chunks={"time": 100}
)
```

### Loading Remote Data

**Before:**
```python
import xarray as xr
import s3fs

fs = s3fs.S3FileSystem()
store = s3fs.S3Map(root="s3://bucket/data.zarr", s3=fs)
ds = xr.open_zarr(store)
```

**After:**
```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset("s3://bucket/data.zarr")
```

### Loading from DOI

**Before:**
```python
import xarray as xr
import requests

# Manually resolve DOI
doi = "10.5281/zenodo.1234567"
response = requests.get(f"https://doi.org/{doi}")
url = response.url

# Load from resolved URL
ds = xr.open_zarr(url)
```

**After:**
```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
```

## Compatibility

The `open_dataset()` function returns a standard `xarray.Dataset`, so all your existing analysis code will work without changes:

```python
# Load with new function
ds = ovro_lwa_portal.open_dataset("data.zarr")

# Use with existing code
mean = ds.SKY.mean(dim="time")
subset = ds.sel(frequency=slice(40e6, 60e6))
result = ds.SKY.compute()
```

## Breaking Changes

None! The new function is additive and doesn't change existing functionality.

## Deprecation Timeline

No deprecation planned. Both approaches will continue to work:
- Direct `xr.open_zarr()` - for advanced users who need fine control
- `ovro_lwa_portal.open_dataset()` - recommended for most users

## Migration Checklist

- [ ] Install latest version: `pip install --upgrade ovro_lwa_portal`
- [ ] For remote access: `pip install 'ovro_lwa_portal[remote]'`
- [ ] Update imports: `import ovro_lwa_portal`
- [ ] Replace `xr.open_zarr()` with `ovro_lwa_portal.open_dataset()`
- [ ] Test with your data
- [ ] Update documentation/notebooks

## Need Help?

- See [full documentation](open_dataset.md)
- Check [examples notebook](../notebooks/open_dataset_examples.ipynb)
- Open an issue on GitHub

## Gradual Migration

You can migrate gradually - both approaches work:

```python
import xarray as xr
import ovro_lwa_portal

# Old code (still works)
ds1 = xr.open_zarr("old_data.zarr")

# New code
ds2 = ovro_lwa_portal.open_dataset("new_data.zarr")

# Both are xarray Datasets
combined = xr.concat([ds1, ds2], dim="time")
```
