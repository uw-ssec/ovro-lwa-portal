# Data Loading

The `ovro_lwa_portal.io` module provides a unified interface for loading
OVRO-LWA datasets from local paths, remote URLs, and DOI identifiers.

## Quick Reference

```python
import ovro_lwa_portal

# Local path
ds = ovro_lwa_portal.open_dataset("/path/to/data.zarr")

# Remote URL (S3, HTTPS, GCS)
ds = ovro_lwa_portal.open_dataset("s3://bucket/data.zarr")

# DOI identifier
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")

# Custom chunking
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50},
)
```

## Supported Protocols

| Protocol   | Example                         | Notes                           |
| ---------- | ------------------------------- | ------------------------------- |
| Local path | `/data/obs.zarr`                | Checks existence before loading |
| S3         | `s3://bucket/data.zarr`         | Via fsspec                      |
| HTTPS      | `https://example.com/data.zarr` | Via fsspec                      |
| GCS        | `gs://bucket/data.zarr`         | Via fsspec                      |
| Azure      | `abfs://container/data.zarr`    | Via fsspec                      |
| DOI        | `doi:10.5281/zenodo.1234567`    | Resolves via DataCite API       |

## Full API Reference

::: ovro_lwa_portal.io
    options:
      show_root_heading: true
      show_root_full_path: false
      members_order: source
