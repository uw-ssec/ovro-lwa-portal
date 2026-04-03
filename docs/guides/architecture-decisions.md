# Architecture Decisions

Technical decisions made during development, with rationale and outcomes. Newest
entries appear first within each section.

## Celestial Coordinate Tracking

### 2026-04-03: Negate l in SIN projection to match RA direction (Accepted)

- **Decision:** Changed the SIN projection l formula from `l = cos(δ)·sin(H)` to
  `l = -cos(δ)·sin(H)` in both `_compute_pixel_track` and
  `_compute_pixel_at_time`.
- **Why:** Casey reported that increasing `ra_center` in `plot_cutout` moved the
  source in the opposite direction. The standard SIN projection
  `l = cos(δ)·sin(H)` gives l positive for sources _west_ of the meridian
  (positive hour angle), but the FITS WCS convention (CDELT1 < 0) defines l
  increasing _eastward_ (with increasing RA). The sign mismatch meant the RA→l
  conversion was inverted. Dec was unaffected because m only depends on
  `cos(H)`, which is symmetric.
- **Result:** RA shifts now move cutouts in the correct direction. The l/m and
  Dec paths were already correct (no projection involved for l/m; m formula is
  sign-symmetric in H). All 417 tests pass after fixing the same formula in the
  test helper `_compute_lm_track`.

### 2026-04-03: Use coordinate order not min/max for imshow extent (Accepted)

- **Decision:** Changed all spatial `imshow` extent arrays from
  `[l_min, l_max, m_min, m_max]` to `[l[0], l[-1], m[0], m[-1]]`.
- **Why:** After the `.T` fix (see below), the data array has l[0] on the left
  edge of the image. When l is in descending order (l[0]=1.117, l[-1]=-1.117),
  using `[l_min, l_max]` = `[-1.117, 1.117]` flipped the extent relative to the
  data, making l-direction shifts appear inverted.
- **Result:** Extent now matches the actual data ordering regardless of whether
  coordinates are ascending or descending.

### 2026-04-03: Transpose spatial data in imshow to align l→x, m→y (Accepted)

- **Decision:** Added `.T` to all 6 spatial `imshow` calls (`plot`,
  `plot_cutout`, `plot_diff`, `plot_grid`, `plot_time_average`,
  `plot_frequency_average`).
- **Why:** The xarray Dataset stores images as `(l, m)` where l=NAXIS1 (RA/x)
  and m=NAXIS2 (Dec/y). But `imshow` maps array axis 0→y and axis 1→x. Without
  `.T`, the l data ended up on the y-axis and m on the x-axis. The axis labels
  said x=l, y=m — matching each other but not the pixel data. This caused
  Casey's original report: reading a coordinate off the plot and feeding it back
  as `ra_center`/`dec_center` shifted the cutout in the wrong direction because
  RA and Dec values were effectively swapped.
- **Result:** Plot pixel data now aligns with axis labels. The animation methods
  and `plot_wcs` already had `.T` and were unaffected.

### 2026-04-02: Guard find_peaks ra/dec against WCS NaN at projection boundary (Accepted)

- **Decision:** `find_peaks` now checks `np.isfinite` on `pixel_to_coords`
  return values before assigning to the peak dict. Peaks outside the SIN
  projection domain keep `ra=None, dec=None`.
- **Why:** QA on real OSN data found that peaks near l^2+m^2 ~ 1 (the SIN
  projection boundary) got `ra=nan, dec=nan` instead of `None`. Astropy's WCS
  returns `nan` for these edge pixels, overwriting the `None` default.
- **Result:** Callers can reliably test `if peak["ra"] is not None` without also
  checking for `nan`.

### 2026-04-02: Single-time methods resolve RA/Dec at the requested frame (Accepted)

- **Decision:** `spectrum()`, `spectral_index()`, `integrated_flux()`, and
  `cutout()` now pass `time_idx` to `coords_to_pixel` so the SIN projection uses
  the LST at the requested time step, not the static WCS reference epoch.
- **Why:** The static WCS path converts RA/Dec to pixel using the WCS header's
  CRVAL (reference RA), which is only correct at the reference time. Away from
  that epoch, Earth rotation shifts the source to a different pixel. Single-time
  methods already know which time step they need — using the static path
  silently sampled the wrong pixel.
- **Result:** All RA/Dec-to-pixel conversions are now time-aware. Tests updated
  to use time steps where the source is above the horizon. A lightweight
  `_compute_pixel_at_time` method computes LST for just one timestamp instead of
  the full time axis.

### 2026-04-02: RA axis inverted in celestial cutout plots (Accepted)

- **Decision:** `plot_cutout()` in celestial mode sets the imshow extent to
  `[ra_max, ra_min, ...]` so RA increases to the left.
- **Why:** Standard astronomical convention: RA increases leftward on sky
  images. Without inversion, the cutout appears mirrored relative to
  WCS-projected plots.
- **Result:** Celestial cutout plots now match the orientation of `plot_wcs()`.

### 2026-04-01: Cache LST per (timestamps, longitude) on the accessor (Accepted)

- **Decision:** Store computed LST arrays in `self._lst_cache` keyed by
  `(mjd_array.tobytes(), lon_deg)`.
- **Why:** `_compute_pixel_track` is called by `dynamic_spectrum`,
  `light_curve`, and (indirectly) by `coords_to_pixel`. Tracking multiple
  sources on the same dataset would recompute the same LST array each time.
  Astropy's `Time.sidereal_time()` has a ~200ms one-time init cost plus ~0.5ms
  per call regardless of array size.
- **Result:** Second call to a different source on the same dataset drops from
  ~1s to ~5ms. Cache is scoped to the accessor instance (tied to the dataset
  lifetime), so there is no cross-dataset leakage.

### 2026-04-01: Use per-pixel dask.compute instead of full spatial grid load (Accepted)

- **Decision:** Replace `data_var.isel(time=vis_times).load()` with per-pixel
  `isel(time=t, l=li, m=mi)` selections computed in a single `dask.compute()`
  call.
- **Why:** The old approach loaded the entire spatial grid (e.g., 4096x4096) for
  all visible time steps just to extract one pixel per step. For 10 time steps
  on a 4096x4096 image, this loaded ~3.4 GB when only ~80 bytes were needed.
  Casey reported `plot_dynamic_spectrum` hanging for minutes.
- **Result:** `dynamic_spectrum(ra=CygA)` on 4096x4096 data completes in ~1s
  (down from minutes). Dask deduplicates chunk reads automatically when multiple
  pixels fall in the same chunk.

### 2026-04-01: Eager load small results below 10 MB threshold (Accepted)

- **Decision:** `_maybe_load()` eagerly loads dask-backed DataArrays smaller
  than `_EAGER_LOAD_THRESHOLD` (10 MB).
- **Why:** For the `(l, m)` fixed-pixel path, the result is just n_times x
  n_freqs scalars — tiny, but dask still builds a task graph with per-chunk
  scheduling overhead. Eager loading collapses this to one read.
- **Result:** `dynamic_spectrum(l=0, m=0)` on 4096x4096 data completes in
  ~0.02s. The 10 MB threshold covers all realistic OVRO-LWA single-pixel
  extractions while keeping pathologically large results lazy.

!!! warning "Tradeoff" Eager loading breaks lazy composition — if a caller
chains operations before materialising (e.g., `.sel(frequency=slice(...))`), the
full result is computed upfront. In practice this matters little since the
result is already tiny after single-pixel selection.

### 2026-03-31: Use out-of-range sentinels instead of -1 for invisible pixels (Accepted)

- **Decision:** `_compute_pixel_track` marks invisible time steps with sentinel
  values `n_l` / `n_m` (one past the last valid index) instead of `-1`.
- **Why:** In NumPy, `-1` is a valid index (last element). If a caller forgets
  to check the visibility mask, `-1` silently extracts the last pixel — a wrong
  but plausible-looking result. An out-of-range index raises an `IndexError`.
- **Result:** Fail-loud behaviour for forgotten visibility checks.

### 2026-03-31: SIN projection uses argsort to map sorted indices back (Accepted)

- **Decision:** `_compute_pixel_track` uses `np.argsort` on the l/m coordinate
  arrays and maps `searchsorted` results back to original coordinate order.
- **Why:** The l/m coordinates may be stored in descending order (common in FITS
  images). `searchsorted` requires sorted input, so we sort internally. Without
  mapping back, the returned indices would be into the sorted copy, not the
  original array — causing silent wrong-pixel extraction.
- **Result:** Correct nearest-neighbor pixel lookup regardless of coordinate
  ordering.

## Performance Testing

### 2026-04-02: Production-scale perf tests with Cygnus A injection (Accepted)

- **Decision:** Added `tests/test_perf_realistic.py` with a 4096x4096
  dask-backed dataset and a synthetic Cygnus A source injected at 100 Jy via
  `map_blocks`.
- **Why:** The unit test fixtures use 50x50 images which don't exercise dask
  chunking or realistic I/O patterns. Casey's performance issues only manifested
  at production scale.
- **Result:** 14 tests covering both performance thresholds (<30s) and
  correctness (source detection at visible time steps). The fixture is fully
  lazy — background generated via `da.random.RandomState.uniform`, source
  injected per-chunk via `map_blocks` — so no 3.2 GB eager allocation.

## Profiling Learnings

### Astropy LST scaling is flat, not linear

After the one-time ~200ms module initialization, `Time.sidereal_time()` takes
~0.5ms regardless of whether you pass 1 or 1000 timestamps. The overhead is
dominated by IERS table loading on first call, not per-element computation. This
means the single-timestamp optimization in `_compute_pixel_at_time` saves numpy
overhead (searchsorted, argsort for N elements), not astropy time.

### dask.compute() deduplicates chunk reads

When extracting pixels from different (l, m) positions across time steps, each
`isel(time=t, l=li, m=mi)` creates a separate dask task. But
`dask.compute(*pixel_arrays)` builds a unified graph, and dask's scheduler reads
each underlying zarr chunk at most once even if multiple pixels fall in the same
chunk.

### ProgressBar minimum threshold avoids noise

`dask.diagnostics.ProgressBar(minimum=2.0)` only shows the bar if computation
takes more than 2 seconds. This keeps fast operations silent while giving
feedback on slow ones — important for interactive use where Casey wants to know
"is it doing something?"

## QA on Real S3 Data (2026-04-02)

Ran the full accessor workflow against `all_subbands_2024-05-24_first10.zarr` on
OSN — 10 time x 10 freq x 4096x4096, chunks `(1,1,1,4096,4096)`.

### Chunk layout dominates remote pixel access time

With `(4096,4096)` spatial chunks, every single-pixel `isel(l=idx, m=idx)`
requires fetching the full ~64 MB compressed chunk from S3.
`dynamic_spectrum(l=0, m=0)` reads 100 chunks (10t x 10f) = ~3 GB of network
transfer for 100 scalar values, taking ~115s. This is inherent to the storage
layout, not a code issue.

| Chunk layout | Estimated pixel access (100 chunks) |
| ------------ | ----------------------------------- |
| (4096, 4096) | ~115s over S3                       |
| (512, 512)   | ~2s (64x fewer bytes per chunk)     |

**Recommendation:** Rechunk production zarr stores to (512, 512) or (1024, 1024)
spatial chunks for workflows that mix full-image rendering with point-pixel
access.

### Single-frame operations are fast regardless of chunk size

`spectrum()`, `cutout()`, and `spectral_index()` with RA/Dec all completed in <
0.01s during QA because they read only 1 chunk (one time step at one frequency).
The `_compute_pixel_at_time` optimization confirmed working — no unnecessary LST
computation for the full time axis.

### WCS returns NaN at the SIN projection boundary

Peaks detected near l^2 + m^2 ~ 1 (the edge of the hemisphere) get `nan` from
`astropy.wcs.WCS.world_to_pixel` / `pixel_to_world`. The accessor now catches
these and keeps `ra=None, dec=None` in peak dicts rather than leaking `nan`.
