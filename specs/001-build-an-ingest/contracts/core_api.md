# API Contract: Core Conversion Functions

**Module**: `ovro_lwa_portal.ingest.core` **Purpose**: Framework-independent
FITS to Zarr conversion logic **Requirements**: FR-020, FR-021

## Functions

### convert_fits_to_zarr

**Signature**:

```python
def convert_fits_to_zarr(
    options: ConversionOptions,
    metadata: Optional[ConversionMetadata] = None,
) -> tuple[Path, ConversionMetadata]:
    """
    Convert FITS files to Zarr store with framework-independent logic.

    Parameters
    ----------
    options : ConversionOptions
        Conversion configuration (input dir, output dir, zarr name, etc.)
    metadata : ConversionMetadata, optional
        Existing metadata for resume operations. If None, creates new metadata.

    Returns
    -------
    zarr_path : Path
        Path to the created/updated Zarr store
    metadata : ConversionMetadata
        Updated conversion metadata with processing statistics

    Raises
    ------
    FileNotFoundError
        If no matching FITS files found in input directory
    ValidationError
        If spatial grids differ across time steps
    LockAcquisitionError
        If Zarr store is locked by another process
    ValueError
        If configuration is invalid

    Notes
    -----
    This function is the main entry point for conversion logic. It delegates
    to discovery, validation, and writing functions but maintains no state
    itself (FR-021: framework-independent).

    Examples
    --------
    >>> from pathlib import Path
    >>> options = ConversionOptions(
    ...     input_dir=Path("/data/fits"),
    ...     output_dir=Path("/data/zarr"),
    ...     zarr_name="ovro_lwa.zarr",
    ...     fixed_dir=Path("/data/fixed")
    ... )
    >>> zarr_path, metadata = convert_fits_to_zarr(options)
    >>> print(f"Processed {metadata.files_processed} files")
    """
    ...
```

**Contract**:

- MUST return valid Path to Zarr store
- MUST return ConversionMetadata with accurate statistics
- MUST raise ValidationError on grid mismatch (FR-009)
- MUST raise LockAcquisitionError if concurrent access detected (FR-019)
- MUST be deterministic (same inputs → same output, NFR-003)
- MUST NOT depend on any orchestration framework (FR-021)

---

### fix_fits_headers

**Signature**:

```python
def fix_fits_headers(
    fits_file: FITSImageFile,
    output_dir: Path,
) -> FixedFITSFile:
    """
    Apply BSCALE/BZERO and add mandatory WCS keywords to FITS file.

    Parameters
    ----------
    fits_file : FITSImageFile
        Source FITS file to fix
    output_dir : Path
        Directory to write fixed FITS file

    Returns
    -------
    fixed_file : FixedFITSFile
        Fixed FITS file metadata

    Raises
    ------
    ValueError
        If output directory doesn't exist or isn't writable
    IOError
        If FITS file cannot be read or written

    Notes
    -----
    Applies corrections from existing `_fix_headers()` in fits_to_zarr_xradio.py:
    - Materializes BSCALE/BZERO to float32 data
    - Adds RESTFREQ/RESTFRQ, SPECSYS=LSRK, TIMESYS=UTC, RADESYS=FK5
    - Sets LATPOLE=90.0, identity PC matrix
    - Adds nominal beam parameters (BMAJ/BMIN=6 arcmin)
    - Sets BUNIT=Jy/beam

    Reuses existing fixed files if present (FR-013).

    Examples
    --------
    >>> fits_file = FITSImageFile(
    ...     path=Path("/data/20240524_050019_41MHz.fits"),
    ...     observation_date="20240524",
    ...     observation_time="050019",
    ...     subband_mhz=41,
    ...     is_fixed=False
    ... )
    >>> fixed = fix_fits_headers(fits_file, Path("/data/fixed"))
    >>> assert fixed.fixed_path.exists()
    """
    ...
```

**Contract**:

- MUST reuse existing fixed file if present (FR-013)
- MUST apply all corrections listed in FR-010, FR-011
- MUST preserve original data values after BSCALE/BZERO application
- MUST create output directory if it doesn't exist
- MUST write valid FITS file compatible with xradio

---

### load_fits_for_zarr

**Signature**:

```python
def load_fits_for_zarr(
    fixed_file: FixedFITSFile,
    chunk_lm: int = 1024,
) -> tuple[xr.Dataset, WCSHeader]:
    """
    Load fixed FITS file into xarray Dataset with WCS coordinates for Zarr conversion.

    Parameters
    ----------
    fixed_file : FixedFITSFile
        Fixed FITS file to load
    chunk_lm : int, optional
        Chunk size for L/M dimensions (default: 1024)

    Returns
    -------
    dataset : xr.Dataset
        xarray Dataset with L/M coordinates, RA/Dec coordinates, and WCS header
    wcs_header : WCSHeader
        Extracted WCS header with RA/Dec coordinate arrays

    Raises
    ------
    FileNotFoundError
        If fixed FITS file doesn't exist
    ValueError
        If chunk_lm <= 0

    Notes
    -----
    Uses xradio.read_image() with do_sky_coords=False, compute_mask=False to read
    pixels, then manually computes exact RA/Dec coordinates at pixel centers (origin=0)
    using astropy WCS.

    Attaches 2D coordinates:
      - right_ascension (m, l): degrees, FK5/J2000
      - declination (m, l): degrees, FK5/J2000

    Preserves FITS celestial WCS header redundantly in 4 locations (FR-019c):
      1. Dataset attrs: xds.attrs['fits_wcs_header']
      2. 0-D variable: wcs_header_str (uses np.bytes_ for NumPy 2.0)
      3. Per-variable attrs: xds[var].attrs['fits_wcs_header']
      4. Coordinate attrs: on right_ascension and declination

    Applies optional LM chunking for memory efficiency.

    Examples
    --------
    >>> dataset, wcs = load_fits_for_zarr(fixed_file, chunk_lm=1024)
    >>> assert "l" in dataset.dims
    >>> assert "m" in dataset.dims
    >>> assert "right_ascension" in dataset.coords
    >>> assert "declination" in dataset.coords
    >>> assert dataset.coords["right_ascension"].dims == ("m", "l")
    >>> assert "fits_wcs_header" in dataset.attrs
    >>> assert "wcs_header_str" in dataset.data_vars
    """
    ...
```

**Contract**:

- MUST use xradio for FITS reading (FR-014 compatibility)
- MUST compute and attach RA/Dec coordinates at pixel centers (FR-019a, FR-019d)
- MUST preserve WCS header in 4 redundant locations (FR-019c)
- MUST use np.bytes\_ for 0-D WCS variable (NumPy 2.0 compatibility)
- MUST apply LM chunking if chunk_lm > 0 (FR-018)
- MUST return Dataset with WCS coordinates and WCSHeader object (FR-019e,
  FR-019f)

---

### combine_frequency_subbands

**Signature**:

```python
def combine_frequency_subbands(
    time_step: TimeStepGroup,
    fixed_dir: Path,
    chunk_lm: int,
) -> tuple[xr.Dataset, SpatialGrid, WCSHeader]:
    """
    Combine multiple frequency subbands for a single time step.

    Parameters
    ----------
    time_step : TimeStepGroup
        Group of FITS files for one time step
    fixed_dir : Path
        Directory containing fixed FITS files
    chunk_lm : int
        Chunk size for L/M dimensions

    Returns
    -------
    combined_dataset : xr.Dataset
        Combined dataset with all frequency subbands and WCS coordinates
    spatial_grid : SpatialGrid
        Extracted spatial grid for validation
    wcs_header : WCSHeader
        WCS header from first file (used for all subbands)

    Raises
    ------
    RuntimeError
        If subbands have incompatible structures for combining
    FileNotFoundError
        If fixed FITS files are missing

    Notes
    -----
    Sorts files by frequency for deterministic ordering (FR-006).
    Uses xr.combine_by_coords() or xr.concat() depending on structure.
    Logs warnings for frequency gaps (edge case).
    Preserves WCS coordinates and header through combine/concat operations (FR-019b).
    WCS header is extracted from first file and applied to all subbands.

    Examples
    --------
    >>> dataset, grid, wcs = combine_frequency_subbands(time_step, fixed_dir, 1024)
    >>> assert "frequency" in dataset.dims
    >>> assert dataset.frequency.size == len(time_step.files)
    >>> assert "right_ascension" in dataset.coords
    >>> assert "fits_wcs_header" in dataset.attrs
    """
    ...
```

**Contract**:

- MUST sort by frequency (FR-006, deterministic ordering)
- MUST return combined dataset with sorted frequency dimension
- MUST extract and return SpatialGrid for validation
- MUST extract and return WCSHeader from first file (FR-019a, FR-019b)
- MUST preserve WCS coordinates through combine/concat (FR-019c)
- MUST log warnings for frequency gaps (edge case behavior)

---

### write_zarr_store

**Signature**:

```python
def write_zarr_store(
    dataset: xr.Dataset,
    zarr_store: ZarrStore,
    append: bool = False,
) -> None:
    """
    Write or append xarray Dataset to Zarr store with atomic safety.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to write
    zarr_store : ZarrStore
        Target Zarr store
    append : bool, optional
        If True, append to existing store (default: False)

    Raises
    ------
    LockAcquisitionError
        If Zarr store is locked by another process
    IOError
        If write operation fails

    Notes
    -----
    Implements safe atomic writes (FR-017):
    - First write: Direct write via xradio.write_image()
    - Append: Read existing (lazy), combine, write to temp, atomic swap

    Uses file locking to prevent concurrent writes (FR-019).

    Examples
    --------
    >>> write_zarr_store(dataset, zarr_store, append=False)
    >>> assert zarr_store.path.exists()
    """
    ...
```

**Contract**:

- MUST acquire lock before writing (FR-019)
- MUST use atomic swap for append operations (FR-017)
- MUST use xradio.write_image() for compatibility (FR-014)
- MUST preserve coordinate ordering (time, frequency sorted)
- MUST release lock after operation completes or fails
