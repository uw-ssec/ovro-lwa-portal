# Quick Reference: `open_dataset()`

## Installation

```bash
# Basic (local files only)
pip install ovro_lwa_portal

# With remote access (S3, GCS, DOI)
pip install 'ovro_lwa_portal[remote]'
```

## Basic Usage

```python
import ovro_lwa_portal

# Local file
ds = ovro_lwa_portal.open_dataset("/path/to/data.zarr")

# Remote URL
ds = ovro_lwa_portal.open_dataset("s3://bucket/data.zarr")

# DOI
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
```

## Common Patterns

### Custom Chunking

```python
# Explicit chunks
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 100, "frequency": 50, "l": 512, "m": 512}
)

# Auto chunks (recommended)
ds = ovro_lwa_portal.open_dataset("data.zarr", chunks="auto")

# No chunks (load all)
ds = ovro_lwa_portal.open_dataset("data.zarr", chunks=None)
```

### Disable Validation

```python
ds = ovro_lwa_portal.open_dataset("data.zarr", validate=False)
```

### Error Handling

```python
from ovro_lwa_portal.io import DataSourceError

try:
    ds = ovro_lwa_portal.open_dataset("data.zarr")
except FileNotFoundError:
    print("File not found")
except DataSourceError:
    print("Failed to load data")
except ImportError:
    print("Missing dependency - install with: pip install 'ovro_lwa_portal[remote]'")
```

## Cloud Storage Setup

### AWS S3

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

### Google Cloud Storage

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## Function Signature

```python
def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    engine: str = "zarr",
    validate: bool = True,
    **kwargs: Any,
) -> xr.Dataset
```

## Supported Sources

| Type | Example | Requires |
|------|---------|----------|
| Local path | `/path/to/data.zarr` | Base install |
| HTTP/HTTPS | `https://example.com/data.zarr` | Base install |
| S3 | `s3://bucket/data.zarr` | `[remote]` extras |
| GCS | `gs://bucket/data.zarr` | `[remote]` extras |
| DOI | `doi:10.5281/zenodo.1234567` | `[remote]` extras |

## Common Issues

| Error | Solution |
|-------|----------|
| `FileNotFoundError` | Check path exists |
| `ImportError: s3fs required` | `pip install 'ovro_lwa_portal[remote]'` |
| `DataSourceError: Failed to resolve DOI` | Check DOI is valid |
| Memory error | Use smaller chunks |

## See Also

- [Full Documentation](open_dataset.md)
- [Examples Notebook](../notebooks/open_dataset_examples.ipynb)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
