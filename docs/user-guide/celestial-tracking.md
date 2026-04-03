# Celestial Coordinate Tracking

The `radport` accessor supports per-time-step celestial coordinate tracking,
allowing you to follow a source at a fixed (RA, Dec) as it drifts across the
image due to Earth rotation.

## Overview

OVRO-LWA images use a fixed (l, m) direction-cosine grid. A celestial source at
constant (RA, Dec) maps to **different pixels at different times** because the
hour angle changes as the Earth rotates. The tracking system computes this
mapping using the closed-form SIN projection equations:

```
H   = LST - RA                                    (hour angle)
l   = cos(Dec) * sin(H)
m   = sin(Dec)*cos(lat) - cos(Dec)*sin(lat)*cos(H)
```

where `lat` is the observatory geodetic latitude (37.2339° for OVRO-LWA).

## Basic Usage

All spatial extraction methods accept either `(l, m)` or `(ra, dec)`
coordinates:

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

# Fixed pixel (same pixel at every time step)
dynspec = ds.radport.dynamic_spectrum(l=0.0, m=0.0)

# Tracked source (different pixel at each time step)
dynspec = ds.radport.dynamic_spectrum(ra=299.868, dec=40.734)  # Cygnus A
```

When using `(ra, dec)`, the accessor:

1. Computes LST at each MJD timestamp
2. Projects (RA, Dec) to (l, m) at each time step via SIN projection
3. Finds the nearest pixel at each step
4. Extracts the per-time pixel values
5. NaN-fills time steps where the source is below the horizon

## Methods Supporting RA/Dec Tracking

### Per-time tracking (pixel changes with time)

| Method                        | Description                                   |
| ----------------------------- | --------------------------------------------- |
| `dynamic_spectrum(ra=, dec=)` | Time-frequency waterfall following source     |
| `light_curve(ra=, dec=)`      | Time series at one frequency following source |

### Single-time-step (pixel resolved at specific time)

| Method                            | Description                                            |
| --------------------------------- | ------------------------------------------------------ |
| `spectrum(ra=, dec=, time_idx=)`  | Frequency spectrum at one time                         |
| `cutout(ra_center=, dec_center=)` | Spatial cutout centred on source                       |
| `spectral_index(ra=, dec=)`       | Power-law slope between two frequencies                |
| `integrated_flux(ra=, dec=)`      | Band-integrated flux density                           |
| `find_peaks()`                    | Includes RA/Dec in peak metadata when WCS is available |

!!! note Single-time methods resolve RA/Dec against the **requested time step**,
not the dataset's WCS reference epoch. This means
`spectrum(ra=..., dec=...,     time_idx=5)` converts coordinates using the SIN
projection at time step 5.

## Visibility and Below-Horizon Handling

A source is considered visible when `sin(altitude) > 0`:

```
sin(alt) = sin(Dec)*sin(lat) + cos(Dec)*cos(lat)*cos(H)
```

For per-time tracking methods, below-horizon time steps are NaN-filled:

```python
lc = ds.radport.light_curve(ra=180.0, dec=-60.0)
# Time steps where the source is below the horizon will be NaN
```

If the source is **never** visible during the observation, a `UserWarning` is
issued and the output is all-NaN.

For single-time methods, requesting a below-horizon source raises a
`ValueError`:

```python
# Raises ValueError if Cyg A is below horizon at time_idx=0
spec = ds.radport.spectrum(ra=299.868, dec=40.734, time_idx=0)
```

## Cutouts with Celestial Coordinates

Cutouts support both `(l_center, m_center, dl, dm)` and
`(ra_center, dec_center, dra, ddec)`:

```python
# Cutout in direction cosines
cutout = ds.radport.cutout(l_center=0.0, m_center=0.0, dl=0.1, dm=0.1)

# Cutout in celestial coordinates (degrees)
cutout = ds.radport.cutout(
    ra_center=299.868, dec_center=40.734,
    dra=5.0, ddec=5.0,
    time_idx=3,
)
```

When using `dra`/`ddec`, the angular extent is converted to direction cosines
using the SIN projection at the centre declination. Near the celestial poles
(|Dec| > ~85°), the RA extent degenerates to near-zero in `l` — use `dl`/`dm`
directly for polar cutouts.

## Performance

### Dask-backed data

The tracked extraction path avoids loading the full spatial grid. Instead, it
selects the exact pixel needed at each time step and uses `dask.compute()` to
read only the required chunks:

| Operation                    | 4096x4096 image, 10 time x 10 freq |
| ---------------------------- | ---------------------------------- |
| `dynamic_spectrum(l=0, m=0)` | ~0.02s                             |
| `dynamic_spectrum(ra=CygA)`  | ~1.0s                              |
| `light_curve(l=0, m=0)`      | ~0.01s                             |
| `light_curve(ra=CygA)`       | ~1.0s                              |

The ~1s for RA/Dec paths is dominated by astropy's one-time module
initialization. Subsequent calls to different sources on the same dataset are
nearly instant thanks to LST caching.

### LST caching

The Local Sidereal Time computation is cached per `(timestamps, longitude)` pair
on the accessor instance. This means:

```python
ds.radport.dynamic_spectrum(ra=299.868, dec=40.734)  # ~1.0s (cold)
ds.radport.light_curve(ra=187.706, dec=12.391)        # ~0.005s (warm)
```

The cache key is the raw bytes of the MJD array plus the observatory longitude,
so it invalidates correctly when timestamps change.

### Eager loading threshold

For the `(l, m)` fixed-pixel path, small results (< 10 MB) are eagerly loaded
into memory to avoid dask graph scheduling overhead. This threshold is set by
`_EAGER_LOAD_THRESHOLD` in `accessor.py`.

### Progress reporting

Long-running dask computations show a progress bar via
`dask.diagnostics.ProgressBar`. It activates automatically when extraction takes
more than 2 seconds.

## Observatory Location

The default observatory is OVRO-LWA:

- **Latitude:** 37.2339° N
- **Longitude:** 118.2817° W
- **Height:** 1222 m

Override with a custom `astropy.coordinates.EarthLocation`:

```python
from astropy.coordinates import EarthLocation
from astropy import units as u

vla = EarthLocation(lat=34.0784*u.deg, lon=-107.6184*u.deg, height=2124*u.m)
dynspec = ds.radport.dynamic_spectrum(ra=299.868, dec=40.734, observatory=vla)
```

## See Also

- [WCS Coordinates](wcs-coordinates.md) -- Static WCS pixel-to-sky conversion
- [Coordinate Systems](../getting-started/coordinate-systems.md) -- Overview of
  (l, m) and celestial coordinate systems
- [Spectral Analysis](spectral-analysis.md) -- `spectrum()` and
  `spectral_index()` usage
