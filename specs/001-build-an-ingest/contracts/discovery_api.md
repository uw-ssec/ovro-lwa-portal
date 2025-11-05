# API Contract: File Discovery

**Module**: `ovro_lwa_portal.ingest.discovery` **Purpose**: FITS file discovery
and grouping by time step **Requirements**: FR-005, FR-006

## Functions

### discover_fits_files

**Signature**:

```python
def discover_fits_files(input_dir: Path) -> list[FITSImageFile]:
    """
    Discover all OVRO-LWA FITS files in input directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing FITS files

    Returns
    -------
    fits_files : list[FITSImageFile]
        List of discovered FITS files matching OVRO-LWA naming pattern

    Raises
    ------
    FileNotFoundError
        If input directory doesn't exist
    ValueError
        If no matching FITS files found (edge case from spec)

    Notes
    -----
    Matches pattern: YYYYMMDD_HHMMSS_<SB>MHz_averaged_*-I-image[_fixed].fits
    Extracts: observation_date, observation_time, subband_mhz, is_fixed

    Examples
    --------
    >>> files = discover_fits_files(Path("/data/fits"))
    >>> assert all(isinstance(f, FITSImageFile) for f in files)
    >>> assert all(f.filename_pattern_match for f in files)
    """
    ...
```

**Contract**:

- MUST match OVRO-LWA naming pattern (FR-005)
- MUST raise ValueError if no files found (edge case)
- MUST extract all required metadata from filename
- MUST handle both regular and \_fixed.fits files

---

### group_by_time_step

**Signature**:

```python
def group_by_time_step(
    fits_files: list[FITSImageFile],
) -> list[TimeStepGroup]:
    """
    Group FITS files by observation timestamp.

    Parameters
    ----------
    fits_files : list[FITSImageFile]
        List of discovered FITS files

    Returns
    -------
    time_steps : list[TimeStepGroup]
        List of time step groups, sorted by time_step_key

    Notes
    -----
    Groups by time_step_key (YYYYMMDD_HHMMSS).
    Sorts files within each group by frequency (FR-006).
    Sorts groups by time_step_key for deterministic ordering.
    Logs warnings for frequency gaps within time steps (edge case).

    Examples
    --------
    >>> groups = group_by_time_step(fits_files)
    >>> assert all(isinstance(g, TimeStepGroup) for g in groups)
    >>> assert groups == sorted(groups, key=lambda g: g.time_step_key)
    >>> for group in groups:
    ...     assert group.files == sorted(group.files, key=lambda f: f.subband_mhz)
    """
    ...
```

**Contract**:

- MUST group by time_step_key (YYYYMMDD_HHMMSS)
- MUST sort files within groups by frequency (FR-006)
- MUST sort groups by time_step_key (deterministic)
- MUST detect and log frequency gaps (edge case)
- MUST return at least one group (validated by discover_fits_files)
