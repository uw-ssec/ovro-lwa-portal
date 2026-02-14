# Quick Start

## Loading OVRO-LWA Data

The `open_dataset()` function provides a unified interface for loading OVRO-LWA data from various sources:

```python
import ovro_lwa_portal

# Load from local zarr store
ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

# Load from remote URL
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

# Load from HTTPS URL
ds = ovro_lwa_portal.open_dataset("https://example.com/data.zarr")

# Load via DOI
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
```

### Customizing Chunking

For large datasets, you can customize how data is chunked in memory:

```python
# Use automatic chunking (default)
ds = ovro_lwa_portal.open_dataset("path/to/data.zarr", chunks="auto")

# Specify custom chunk sizes
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50}
)

# Load without chunking (use with caution for large datasets)
ds = ovro_lwa_portal.open_dataset("path/to/data.zarr", chunks=None)
```

## Converting FITS to Zarr

### Using the CLI

Convert OVRO-LWA FITS files to Zarr format:

```bash
# Basic conversion
ovro-ingest convert /path/to/fits /path/to/output

# With custom options
ovro-ingest convert /path/to/fits /path/to/output \
    --zarr-name my_data.zarr \
    --chunk-lm 2048 \
    --rebuild

# Show help
ovro-ingest convert --help
```

### Using the Python API

```python
from pathlib import Path
from ovro_lwa_portal.ingest import FITSToZarrConverter
from ovro_lwa_portal.ingest.core import ConversionConfig

# Configure conversion
config = ConversionConfig(
    input_dir=Path("/path/to/fits"),
    output_dir=Path("/path/to/output"),
    zarr_name="ovro_lwa_data.zarr",
    chunk_lm=1024,
)

# Execute conversion
converter = FITSToZarrConverter(config)
result = converter.convert()
print(f"Created: {result}")
```

## Next Steps

- Learn more about [loading data](../user-guide/loading-data.md)
- Explore [FITS to Zarr conversion](../user-guide/fits-to-zarr.md)
- Check out the [API Reference](../api/overview.md)
