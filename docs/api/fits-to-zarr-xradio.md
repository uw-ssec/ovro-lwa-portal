# FITS to Zarr (Low-Level)

The `ovro_lwa_portal.fits_to_zarr_xradio` module provides the low-level
functions for converting OVRO-LWA FITS image files to Zarr format using xradio.

!!! tip "Prefer the high-level API for most use cases"

    The [`FITSToZarrConverter`](ingest.md#ovro_lwa_portal.ingest.FITSToZarrConverter)
    class in the ingest module wraps these functions with FileLock-based concurrency
    protection, progress callbacks, and a simpler interface. Use the low-level
    functions here only when you need fine-grained control over the conversion
    process.

## Quick Reference

```python
from pathlib import Path
from ovro_lwa_portal.fits_to_zarr_xradio import (
    convert_fits_dir_to_zarr,
    fix_fits_headers,
)

# Fix headers first (optional — convert_fits_dir_to_zarr can do this on-demand)
fits_files = sorted(Path("/data/fits").glob("*.fits"))
fixed = fix_fits_headers(fits_files, Path("/data/fixed_fits"))

# Convert to Zarr
result = convert_fits_dir_to_zarr(
    input_dir="/data/fits",
    out_dir="/data/output",
    fixed_dir="/data/fixed_fits",
    fix_headers_on_demand=False,  # already fixed above
)
```

## API Reference

### convert_fits_dir_to_zarr

::: ovro_lwa_portal.fits_to_zarr_xradio.convert_fits_dir_to_zarr
    options:
      show_root_heading: true
      show_root_full_path: false

### fix_fits_headers

::: ovro_lwa_portal.fits_to_zarr_xradio.fix_fits_headers
    options:
      show_root_heading: true
      show_root_full_path: false
