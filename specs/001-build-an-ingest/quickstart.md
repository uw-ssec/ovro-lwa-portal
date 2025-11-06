# Quickstart: FITS to Zarr Ingest Package

**Date**: October 20, 2025 **Feature**: 001-build-an-ingest **Purpose**:
End-to-end validation guide for the FITS to Zarr ingestion pipeline

## Overview

This quickstart guide provides step-by-step instructions to validate all
acceptance scenarios from the feature specification. It serves as both user
documentation and integration test validation.

## Prerequisites

- Python 3.12+ environment with `ovro_lwa_portal` installed
- Test FITS files (download or use existing samples)
- ~10GB free disk space for test outputs
- Terminal with color support (for rich output)

## Installation

```bash
# Install ovro_lwa_portal package
pixi install

# Verify CLI is available
ovro-lwa-ingest --help

# Check version
ovro-lwa-ingest version
```

**Expected Output**:

```
Usage: ovro-lwa-ingest [OPTIONS] INPUT_DIR OUTPUT_DIR

OVRO-LWA FITS to Zarr ingestion pipeline

...
```

## Test Data Setup

```bash
# Create test directories
mkdir -p test_data/fits test_data/zarr test_data/fixed

# Download test FITS files (if needed)
python .ci-helpers/download_test_fits.py

# Or copy sample files
cp notebooks/test_fits_files/*.fits test_data/fits/
```

## Acceptance Scenario 1: Basic Conversion

**Requirement**: Convert FITS files with standard naming to single Zarr store
(FR-001-FR-009)

```bash
# Run basic conversion
ovro-lwa-ingest \
    test_data/fits \
    test_data/zarr \
    --zarr-name test_output.zarr

# Verify output
ls -lh test_data/zarr/test_output.zarr/
```

**Expected Behavior**:

1. Progress indicators showing each time step
2. Summary statistics at completion:
   ```
   ✓ Conversion complete
   Total files processed: 120
   Time steps: 10
   Frequency range: 27-88 MHz
   Output: test_data/zarr/test_output.zarr
   ```
3. Zarr store with correct structure:
   ```
   test_output.zarr/
   ├── .zarray
   ├── .zattrs
   ├── time/
   ├── frequency/
   ├── l/
   ├── m/
   └── SKY/
   ```

**Validation**:

```python
import xarray as xr

# Load Zarr store
ds = xr.open_zarr("test_data/zarr/test_output.zarr")

# Verify dimensions
assert "time" in ds.dims
assert "frequency" in ds.dims
assert "l" in ds.dims
assert "m" in ds.dims

# Verify data integrity
assert ds.dims["time"] == 10
assert ds.dims["frequency"] > 0
assert "SKY" in ds.data_vars

print("✓ Acceptance Scenario 1: PASSED")
```

## Acceptance Scenario 2: Append to Existing Zarr

**Requirement**: Append new time steps without corrupting existing data (FR-015)

```bash
# Create additional test FITS files in a separate directory
mkdir -p test_data/fits_new
cp test_data/fits/*20240524_051019*.fits test_data/fits_new/

# Append to existing Zarr
ovro-lwa-ingest \
    test_data/fits_new \
    test_data/zarr \
    --zarr-name test_output.zarr

# No --rebuild flag, so should append
```

**Expected Behavior**:

1. Detects existing Zarr store
2. Appends new time steps
3. Preserves existing data

**Validation**:

```python
import xarray as xr

# Reload Zarr store
ds = xr.open_zarr("test_data/zarr/test_output.zarr")

# Verify time dimension increased
assert ds.dims["time"] > 10, "Time dimension should have increased"

# Verify data integrity (no NaNs in original data)
original_time_slice = ds.isel(time=0)
assert not original_time_slice["SKY"].isnull().any()

print("✓ Acceptance Scenario 2: PASSED")
```

## Acceptance Scenario 3: Automatic Header Correction

**Requirement**: Generate fixed FITS files automatically (FR-010-FR-013)

```bash
# Run conversion with custom fixed directory
ovro-lwa-ingest \
    test_data/fits \
    test_data/zarr \
    --zarr-name corrected.zarr \
    --fixed-dir test_data/fixed \
    --rebuild

# Verify fixed files were created
ls -lh test_data/fixed/
```

**Expected Behavior**:

1. Creates `*_fixed.fits` files in `test_data/fixed/`
2. Fixed files have corrected headers (BSCALE/BZERO applied, WCS keywords added)
3. Subsequent runs reuse fixed files (no regeneration)

**Validation**:

```python
from pathlib import Path
from astropy.io import fits

fixed_dir = Path("test_data/fixed")
fixed_files = list(fixed_dir.glob("*_fixed.fits"))

assert len(fixed_files) > 0, "Fixed files should be created"

# Check one fixed file
with fits.open(fixed_files[0]) as hdul:
    header = hdul[0].header

    # Verify corrections
    assert "RESTFREQ" in header or "RESTFRQ" in header
    assert header.get("SPECSYS") == "LSRK"
    assert header.get("TIMESYS") == "UTC"
    assert header.get("RADESYS") == "FK5"
    assert "BSCALE" not in header  # Should be materialized
    assert "BZERO" not in header

print("✓ Acceptance Scenario 3: PASSED")
```

## Acceptance Scenario 4: Progress Information

**Requirement**: Display progress during conversion (FR-023, NFR-001)

```bash
# Run with verbose logging
ovro-lwa-ingest \
    test_data/fits \
    test_data/zarr \
    --zarr-name progress_test.zarr \
    --verbose \
    --rebuild
```

**Expected Behavior**:

1. Progress bar showing time steps
2. Verbose details per file (file names, timing, validation)
3. Summary statistics at end

**Example Output**:

```
Processing FITS files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 10/10 time steps
[DEBUG] Loading 20240524_050019_41MHz_averaged_...-I-image_fixed.fits (1.2s)
[DEBUG] Loading 20240524_050019_55MHz_averaged_...-I-image_fixed.fits (1.1s)
[INFO] Combined 12 frequency subbands for 20240524_050019
[DEBUG] Validated spatial grid: (2048, 2048) - matches reference
[INFO] Wrote time step 20240524_050019 to Zarr (3.5s)
...
✓ Conversion complete
Total files processed: 120
Time steps: 10
Frequency range: 27-88 MHz
Total time: 45.2s
Output: test_data/zarr/progress_test.zarr
```

## Acceptance Scenario 5: Grid Mismatch Detection

**Requirement**: Detect and abort on spatial grid mismatches (FR-009)

```bash
# Create test with mismatched grids (requires synthetic data)
# This test would use pytest fixtures in actual implementation

# Expected: Clear error message with grid details
```

**Expected Error Output**:

```
ERROR: Spatial grid mismatch detected.
Expected grid shape: (2048, 2048)
Got grid shape: (1024, 1024)
Problematic file: test_data/fits/20240524_052019_41MHz_averaged_...-I-image.fits

This usually indicates FITS files from different observations.
Ensure all files in the input directory belong to the same dataset.
```

**Validation**: This scenario is best tested via unit tests, not quickstart.

## Acceptance Scenario 6: Rebuild from Scratch

**Requirement**: Overwrite existing Zarr with --rebuild flag (FR-016)

```bash
# Create initial Zarr
ovro-lwa-ingest test_data/fits test_data/zarr --zarr-name rebuild_test.zarr

# Note the creation time
stat test_data/zarr/rebuild_test.zarr

# Rebuild
ovro-lwa-ingest \
    test_data/fits \
    test_data/zarr \
    --zarr-name rebuild_test.zarr \
    --rebuild

# Verify modification time changed
stat test_data/zarr/rebuild_test.zarr
```

**Expected Behavior**:

1. `--rebuild` flag bypasses append logic
2. Existing Zarr store is overwritten
3. No state file or resume prompt

**Validation**:

```python
import xarray as xr
from pathlib import Path

zarr_path = Path("test_data/zarr/rebuild_test.zarr")
state_file = zarr_path.with_suffix(".zarr.state.json")

# Verify Zarr exists
assert zarr_path.exists()

# Verify no state file (rebuild doesn't use state)
assert not state_file.exists()

# Verify data is fresh
ds = xr.open_zarr(zarr_path)
assert ds.dims["time"] == 10  # Should match input, not append

print("✓ Acceptance Scenario 6: PASSED")
```

## Acceptance Scenario 7: Help Documentation

**Requirement**: Display clear usage with --help (FR-031)

```bash
# Test help flag
ovro-lwa-ingest --help

# Test command-specific help
ovro-lwa-ingest convert --help
```

**Expected Output**:

```
Usage: ovro-lwa-ingest [OPTIONS] INPUT_DIR OUTPUT_DIR

OVRO-LWA FITS to Zarr ingestion pipeline

Arguments:
  INPUT_DIR   Directory containing OVRO-LWA FITS files  [required]
  OUTPUT_DIR  Directory for output Zarr store  [required]

Options:
  -n, --zarr-name TEXT       Name of output Zarr store  [default: ovro_lwa_full_lm_only.zarr]
  -f, --fixed-dir PATH       Directory for corrected FITS files  [default: fixed_fits]
  -c, --chunk-lm INTEGER     Chunk size for L/M spatial dimensions  [default: 1024]
  -r, --rebuild              Rebuild Zarr store from scratch
  -p, --use-prefect          Use Prefect orchestration (if installed)
  -v, --verbose              Enable verbose logging
  -q, --quiet                Suppress all output except errors
  -l, --log-file PATH        Write logs to file
  --help                     Show this message and exit
```

## Acceptance Scenario 8: WCS Coordinate Preservation and Plotting

**Requirement**: Verify WCS coordinates are preserved and can be used for
plotting (FR-019a-f)

```bash
# Run conversion with default settings
ovro-lwa-ingest test_data/fits test_data/zarr --zarr-name wcs_test.zarr
```

**Expected Behavior**:

1. Zarr store contains RA/Dec coordinates
2. WCS header preserved in multiple locations
3. WCS can be reconstructed for plotting

**Validation**:

```python
import xarray as xr
import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS

# Load Zarr store
ds = xr.open_zarr("test_data/zarr/wcs_test.zarr", consolidated=False)

# Verify RA/Dec coordinates exist
assert "right_ascension" in ds.coords
assert "declination" in ds.coords

# Verify 2D coordinate arrays with correct dimensions
assert ds.coords["right_ascension"].dims == ("m", "l")
assert ds.coords["declination"].dims == ("m", "l")

# Verify coordinate attributes
assert ds["right_ascension"].attrs["units"] == "deg"
assert ds["right_ascension"].attrs["frame"] == "fk5"
assert ds["right_ascension"].attrs["equinox"] == "J2000"

# Verify WCS header in multiple locations (redundant storage)
locations_checked = []

# 1. Dataset global attrs
if "fits_wcs_header" in ds.attrs:
    locations_checked.append("global_attrs")

# 2. 0-D variable (robust across operations)
if "wcs_header_str" in ds.data_vars:
    val = ds["wcs_header_str"].values
    if isinstance(val, np.ndarray):
        val = val.item()
    # Handle np.bytes_ (NumPy 2.0)
    if isinstance(val, (bytes, bytearray)) or type(val).__name__ == "bytes_":
        hdr_str = val.decode("utf-8")
        locations_checked.append("0d_variable")

# 3. Per-variable attrs
var = "SKY" if "SKY" in ds.data_vars else list(ds.data_vars)[0]
if "fits_wcs_header" in ds[var].attrs:
    locations_checked.append("per_variable_attrs")

# 4. Coordinate attrs
if "fits_wcs_header" in ds["right_ascension"].attrs:
    locations_checked.append("coord_attrs")

print(f"✓ WCS header found in {len(locations_checked)} locations: {locations_checked}")
assert len(locations_checked) >= 1, "WCS header must be preserved in at least one location"

print("✓ Acceptance Scenario 8: PASSED")
```

**WCS Reconstruction and Plotting**:

```python
import matplotlib.pyplot as plt
from astropy.io.fits import Header
from astropy.wcs import WCS

def get_wcs_from_zarr(z: xr.Dataset, var: str = "SKY") -> WCS:
    """Reconstruct WCS from Zarr store without original FITS files."""
    if var not in z.data_vars:
        var = "BEAM" if "BEAM" in z.data_vars else list(z.data_vars)[0]

    # Prefer per-variable attr (most robust)
    hdr_str = z[var].attrs.get("fits_wcs_header")

    if not hdr_str:
        # Fallback to 0-D variable
        val = z["wcs_header_str"].values
        if isinstance(val, np.ndarray):
            val = val.item()
        if isinstance(val, (bytes, bytearray)) or type(val).__name__ == "bytes_":
            hdr_str = val.decode("utf-8")
        else:
            hdr_str = str(val)

    return WCS(Header.fromstring(hdr_str, sep="\n"))

# Reconstruct WCS
wcs = get_wcs_from_zarr(ds, var="SKY")
print(f"✓ Reconstructed WCS: {wcs.wcs.ctype}")

# Plot with WCS projection
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(projection=wcs)

# Select time and frequency slice
tsel = 0
fsel = ds.sizes.get("frequency", 1) // 2
img = ds["SKY"].isel(time=tsel, frequency=fsel).values

# Plot
ax.imshow(img, origin="lower", cmap="inferno")
ax.set_xlabel("RA")
ax.set_ylabel("Dec")

# Add WCS grid overlay
overlay = ax.get_coords_overlay("fk5")
overlay.grid(color="white", ls=":", lw=1.0, alpha=0.8)

plt.tight_layout()
plt.savefig("test_data/wcs_plot.png")
print("✓ WCS plot saved to test_data/wcs_plot.png")
```

**Benefits**:

1. **FITS-free analysis**: WCS can be reconstructed without original FITS files
   (FR-019f)
2. **Robust storage**: Redundant WCS header storage survives xarray operations
   (FR-019c)
3. **Exact coordinates**: RA/Dec computed at pixel centers for FITS standard
   alignment (FR-019d)
4. **WCS-aware plotting**: Direct matplotlib WCSAxes support for
   publication-quality figures

## Edge Case: No Matching FITS Files

**Requirement**: Report error if input directory has no matching files

```bash
# Create empty directory
mkdir -p test_data/empty

# Attempt conversion
ovro-lwa-ingest test_data/empty test_data/zarr --zarr-name empty_test.zarr
```

**Expected Error**:

```
ERROR: No matching FITS files found in test_data/empty

Ensure the directory contains FITS files matching the OVRO-LWA naming pattern:
  YYYYMMDD_HHMMSS_<SB>MHz_averaged_*-I-image.fits

Example: 20240524_050019_41MHz_averaged_...-I-image.fits
```

## Edge Case: Concurrent Access

**Requirement**: Detect concurrent writes and fail fast (FR-019)

```bash
# Terminal 1: Start long-running conversion
ovro-lwa-ingest test_data/fits test_data/zarr --zarr-name concurrent.zarr &

# Terminal 2: Attempt concurrent write (should fail immediately)
ovro-lwa-ingest test_data/fits test_data/zarr --zarr-name concurrent.zarr
```

**Expected Error in Terminal 2**:

```
ERROR: Another process is currently writing to this Zarr store.
Lock held by process: 12345
Lock file: test_data/zarr/concurrent.zarr.lock

Suggested actions:
  1. Wait for the other process to complete
  2. If process crashed, manually remove the lock file:
     rm test_data/zarr/concurrent.zarr.lock
```

## Edge Case: Interrupted Pipeline (Resume/Rebuild)

**Requirement**: Prompt user for resume or rebuild (FR-027, NFR-002)

```bash
# Start conversion and interrupt with Ctrl+C
ovro-lwa-ingest test_data/fits test_data/zarr --zarr-name interrupted.zarr
# Press Ctrl+C after a few time steps

# Restart conversion (should prompt)
ovro-lwa-ingest test_data/fits test_data/zarr --zarr-name interrupted.zarr
```

**Expected Interactive Prompt**:

```
Incomplete conversion detected for test_data/zarr/interrupted.zarr
Last processed: 20240524_050019
Completed: 3 time steps

Resume from last checkpoint? [Y/n]: y
Resuming conversion...
Processing FITS files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 70% 7/10 time steps
...
```

**If user selects "no"**:

```
Resume from last checkpoint? [Y/n]: n
Rebuild from scratch (delete existing data)? [y/N]: y
Rebuilding from scratch...
Processing FITS files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10% 1/10 time steps
...
```

## Cleanup

```bash
# Remove test outputs
rm -rf test_data/zarr/
rm -rf test_data/fixed/

# Keep FITS files for future tests
```

## Summary

This quickstart validates:

- ✓ Basic FITS to Zarr conversion (Scenario 1)
- ✓ Append functionality (Scenario 2)
- ✓ Automatic header correction (Scenario 3)
- ✓ Progress indicators and logging (Scenario 4)
- ✓ Grid validation (Scenario 5 - unit tests)
- ✓ Rebuild from scratch (Scenario 6)
- ✓ Help documentation (Scenario 7)
- ✓ Error handling for edge cases

All acceptance scenarios from the specification are covered and validated.

## Next Steps

- Run automated integration tests: `pytest tests/ingest/test_cli.py`
- Performance validation: Process full 96TB dataset
- Prefect integration test: `ovro-lwa-ingest ... --use-prefect`
