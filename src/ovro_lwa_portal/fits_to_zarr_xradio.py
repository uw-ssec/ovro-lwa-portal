"""xradio-powered FITS → Zarr conversion for OVRO-LWA.

This module converts per-time, per-subb and FITS images into a single LM-only Zarr store.
It uses `xradio` for FITS I/O and Zarr writing, and enforces deterministic ordering by
sorting on frequency and time. It also materializes FITS scaling (BSCALE/BZERO) and adds
a minimal set of header keywords so `xradio` can parse images reliably.

Library usage
-------------
    # Option 1: Fix headers on-demand during conversion (default behavior)
    from ovro_lwa_portal.ingest.fit_to_zarr_xradio import convert_fits_dir_to_zarr
    out = convert_fits_dir_to_zarr(
        input_dir="/path/to/fits",
        out_dir="zarr_out",
        zarr_name="ovro_lwa_full_lm_only.zarr",
        fixed_dir="fixed_fits",
        chunk_lm=1024,
        rebuild=False,
    )

    # Option 2: Fix headers ahead of time, then convert
    from pathlib import Path
    from ovro_lwa_portal.ingest.fit_to_zarr_xradio import fix_fits_headers, convert_fits_dir_to_zarr

    # Step 1: Fix all headers first
    input_files = list(Path("/path/to/fits").glob("*.fits"))
    fixed_dir = Path("fixed_fits")
    fix_fits_headers(input_files, fixed_dir)

    # Step 2: Convert using pre-fixed headers
    out = convert_fits_dir_to_zarr(
        input_dir="/path/to/fits",
        out_dir="zarr_out",
        zarr_name="ovro_lwa_full_lm_only.zarr",
        fixed_dir="fixed_fits",
        chunk_lm=1024,
        rebuild=False,
        fix_headers_on_demand=False,  # Skip fixing since already done
    )

Notes
-----
* Discovery groups files by observation time/frequency from FITS headers first.
  Filename parsing is only used as a fallback when header metadata is missing.
* LM grids must match across time steps after global and per-step mixed-resolution normalization;
  a mismatch raises a RuntimeError.
* Within a single time step, mixed LM shapes are regridded onto the reference grid before combine.
* On append, the existing Zarr is read and re-written with the appended time step.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from numpy.typing import NDArray
from xradio.image import read_image, write_image

__all__ = ["convert_fits_dir_to_zarr", "fix_fits_headers"]

logger = logging.getLogger(__name__)

# Match: 20240524_050019_41MHz_averaged_...-I-image(.fits|_fixed.fits)
PAT = re.compile(
    r"^(?P<date>\d{8})_(?P<hms>\d{6})_(?P<sb>\d+)MHz_averaged_.*-I-image(?:_fixed)?\.fits$"
)
MHZ_RE = re.compile(r"_(\d+)MHz_")


def _time_key_from_name(p: Path) -> Optional[str]:
    """Extract a normalized ``YYYYMMDD_HHMMSS`` key from filename when possible."""
    m = PAT.match(p.name)
    if not m:
        return None
    return f"{m.group('date')}_{m.group('hms')}"


def _mhz_from_name(p: Path) -> int:
    """Extract the subband MHz from a filename; return a large sentinel if absent.

    Parameters
    ----------
    p : Path
        Path object with filename to extract MHz from.

    Returns
    -------
    int
        Subband frequency in MHz, or 10**9 if not found.
    """
    m = MHZ_RE.search(p.name)
    return int(m.group(1)) if m else 10**9


def _time_key_from_header(header: fits.Header) -> Optional[str]:
    """Extract observation time from FITS headers as ``YYYYMMDD_HHMMSS``.

    Header precedence:
      1) ``DATE-OBS`` (optionally with ``TIME-OBS`` if date-only)
      2) ``MJD-OBS``
      3) ``MJD``

    Returns ``None`` when no usable timestamp is found.
    """
    date_obs = header.get("DATE-OBS")
    if date_obs:
        date_obs = str(date_obs).strip()
        time_obs = header.get("TIME-OBS")
        if time_obs and "T" not in date_obs:
            dt_value = f"{date_obs}T{str(time_obs).strip()}"
        else:
            dt_value = date_obs
        try:
            t = Time(dt_value, format="isot", scale="utc")
            return t.to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            logger.debug(f"Could not parse DATE-OBS/TIME-OBS timestamp: {dt_value}")

    for mjd_key in ("MJD-OBS", "MJD"):
        if mjd_key in header:
            try:
                t = Time(float(header[mjd_key]), format="mjd", scale="utc")
                return t.to_datetime().strftime("%Y%m%d_%H%M%S")
            except Exception:
                logger.debug(f"Could not parse {mjd_key} timestamp: {header.get(mjd_key)}")

    return None


def _frequency_hz_from_header(header: fits.Header) -> Optional[float]:
    """Extract frequency in Hz from FITS headers.

    Header precedence:
      1) ``RESTFREQ``
      2) ``RESTFRQ``
      3) ``CRVAL3``
      4) ``FREQ``
    """
    for key in ("RESTFREQ", "RESTFRQ", "CRVAL3", "FREQ"):
        value = header.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.debug(f"Could not parse {key} frequency: {value}")
    return None


def _extract_group_metadata(fp: Path) -> Tuple[Optional[str], Optional[float], List[str]]:
    """Extract grouping metadata from headers with filename fallback.

    Returns
    -------
    Tuple[Optional[str], Optional[float], List[str]]
        Tuple of ``(time_key, frequency_hz, fallback_notes)`` where fallback_notes
        records when filename-based fallback was used.
    """
    time_key: Optional[str] = None
    frequency_hz: Optional[float] = None
    notes: List[str] = []

    header: Optional[fits.Header] = None
    try:
        header = fits.getheader(fp, ext=0)
    except Exception as e:
        logger.warning(f"Could not read FITS header for {fp.name}: {e}")

    if header is not None:
        time_key = _time_key_from_header(header)
        frequency_hz = _frequency_hz_from_header(header)

    if time_key is None:
        name_time_key = _time_key_from_name(fp)
        if name_time_key is not None:
            time_key = name_time_key
            notes.append("time-from-filename")

    if frequency_hz is None:
        mhz = _mhz_from_name(fp)
        if mhz != 10**9:
            frequency_hz = float(mhz * 1e6)
            notes.append("frequency-from-filename")

    return time_key, frequency_hz, notes


def _frequency_sort_tuple(fp: Path) -> Tuple[float, str]:
    """Sort key for deterministic frequency ordering with fallback."""
    _, frequency_hz, _ = _extract_group_metadata(fp)
    if frequency_hz is None:
        return (float(10**15), fp.name)
    return (float(frequency_hz), fp.name)


def _fix_headers(path_in: Path, path_out: Path) -> None:
    """Write a *_fixed.fits with BSCALE/BZERO applied and minimal WCS/spectral keys.

    Adds/ensures:
      RESTFREQ/RESTFRQ, SPECSYS=LSRK, TIMESYS=UTC, RADESYS=FK5, LATPOLE=90,
      identity PC matrix for LM, nominal beam (BMAJ/BMIN=6 arcmin), BUNIT=Jy/beam.

    Parameters
    ----------
    path_in : Path
        Input FITS file path.
    path_out : Path
        Output fixed FITS file path.
    """
    with fits.open(path_in, memmap=True) as hdul:
        hdu = hdul[0]
        data = hdu.data
        hdr = hdu.header.copy()

        bscale = float(hdr.get("BSCALE", 1.0))
        bzero = float(hdr.get("BZERO", 0.0))
        if (bscale != 1.0) or (bzero != 0.0):
            data = data.astype(np.float32) * bscale + bzero
            for k in ("BSCALE", "BZERO"):
                if k in hdr:
                    del hdr[k]

        phdu = fits.PrimaryHDU(data=data, header=hdr)
        H = phdu.header

        # Spectral / frame basics
        if "CRVAL3" in H:  # fall back to CRVAL3 if set
            H["RESTFREQ"] = H["CRVAL3"]
        H["RESTFRQ"] = (H.get("RESTFREQ", 1.0), "Rest frequency in Hz")
        H["SPECSYS"] = "LSRK"
        H["TIMESYS"] = "UTC"
        H["RADESYS"] = "FK5"
        H["LATPOLE"] = 90.0

        # Identity PC for LM axes
        H["PC1_1"] = 1.0
        H["PC1_2"] = 0.0
        H["PC2_1"] = 0.0
        H["PC2_2"] = 1.0

        # Nominal synthesized beam (6 arcmin)
        H["BMAJ"] = 6 / 60
        H["BMIN"] = 6 / 60
        if "BPA" not in H:
            H["BPA"] = 0.0
        if "BUNIT" not in H:
            H["BUNIT"] = "Jy/beam"

        phdu.writeto(path_out, overwrite=True)


def _get_fixed_paths(files: List[Path], fixed_dir: Path) -> List[Path]:
    """Get paths to fixed FITS files, assuming they already exist.

    This function assumes that headers have already been fixed using
    :func:`fix_fits_headers` and simply returns the paths to the
    ``*_fixed.fits`` files.

    Parameters
    ----------
    files : List[Path]
        List of FITS file paths (may be original or already-fixed files).
    fixed_dir : Path
        Directory containing the ``*_fixed.fits`` files.

    Returns
    -------
    List[Path]
        List of paths to fixed FITS files, sorted by frequency.
    """
    fixed_paths: List[Path] = []
    for f in sorted(files, key=_frequency_sort_tuple):
        if f.name.endswith("_fixed.fits"):
            fixed_paths.append(f)
        else:
            fixed = fixed_dir / (f.stem + "_fixed.fits")
            fixed_paths.append(fixed)
    return fixed_paths


def fix_fits_headers(
    files: List[Path],
    fixed_dir: Path,
    *,
    skip_existing: bool = True,
) -> List[Path]:
    """Fix FITS headers for a list of files, creating ``*_fixed.fits`` files.

    This function processes FITS files to ensure they have the necessary
    headers for xradio conversion. It can be run ahead of time before
    calling :func:`convert_fits_dir_to_zarr` to separate the header
    fixing step from the conversion process.

    Parameters
    ----------
    files : List[Path]
        List of FITS file paths to process.
    fixed_dir : Path
        Directory where ``*_fixed.fits`` files will be written.
    skip_existing : bool, optional
        If True, skip files that already have corresponding fixed versions.
        Default is True.

    Returns
    -------
    List[Path]
        List of paths to the fixed FITS files.

    Notes
    -----
    * Files already ending with ``_fixed.fits`` are considered already fixed
      and are returned as-is.
    * The :func:`_fix_headers` function applies BSCALE/BZERO and adds minimal
      WCS/spectral keywords required by xradio.

    Examples
    --------
    >>> from pathlib import Path
    >>> from ovro_lwa_portal.fits_to_zarr_xradio import fix_fits_headers
    >>> input_files = list(Path("input").glob("*.fits"))
    >>> fixed_dir = Path("fixed_fits")
    >>> fixed_dir.mkdir(exist_ok=True)
    >>> fixed_paths = fix_fits_headers(input_files, fixed_dir)
    >>> print(f"Fixed {len(fixed_paths)} files")
    """
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fixed_paths: List[Path] = []

    for f in sorted(files, key=_frequency_sort_tuple):
        if f.name.endswith("_fixed.fits"):
            # Already fixed, use as-is
            fixed_paths.append(f)
            logger.debug(f"Skipping already-fixed file: {f.name}")
        else:
            fixed = fixed_dir / (f.stem + "_fixed.fits")
            if skip_existing and fixed.exists():
                logger.debug(f"Skipping existing fixed file: {fixed.name}")
            else:
                logger.info(f"Fixing headers: {f.name} -> {fixed.name}")
                _fix_headers(f, fixed)
            fixed_paths.append(fixed)

    return fixed_paths


def _load_for_combine(fp: Path, *, chunk_lm: int = 1024) -> xr.Dataset:
    """
    Load a FITS image, attach *sky* coordinates from the FITS celestial WCS,
    and persist the exact WCS header for FITS-free WCSAxes plotting later.

    This function:
      • reads pixels via :func:`xradio.image.read_image` with sky coords disabled
      • evaluates RA/Dec at pixel centers (origin=0) using the 2D celestial WCS
      • attaches 2D ``right_ascension``/``declination`` coordinates (deg; FK5/J2000)
      • stores the exact celestial WCS header string redundantly so it survives merges

    Parameters
    ----------
    fp : Path
        Path to the FITS image (original or ``*_fixed.fits``) to load.
    chunk_lm : int, optional
        Chunk size for the ``l`` and ``m`` dimensions. Set to ``0`` to disable
        chunking. Default is ``1024``.

    Returns
    -------
    xarray.Dataset
        Dataset with:
          • data vars from the input FITS (e.g., ``SKY``/``BEAM``)
          • 2D coords: ``right_ascension`` and ``declination`` in degrees
          • WCS header persisted in multiple locations (see Notes)

    Notes
    -----
    * RA/Dec are computed at pixel **centers** via
      ``WCS(header).celestial.all_pix2world(xx, yy, origin=0)`` and therefore
      exactly match the FITS celestial WCS.
    * The celestial WCS header is stored redundantly as:
        - ``xds.attrs['fits_wcs_header']`` (global attrs)
        - a 0-D variable ``wcs_header_str`` (robust across combines)
        - per-variable ``.attrs['fits_wcs_header']``
        - on the RA/Dec coord attrs
      This redundancy ensures at least one copy survives downstream combine/concat
      operations and writers that may drop attrs.
    * Uses :class:`numpy.bytes_` (NumPy ≥ 2.0) for the scalar variable payload.

    """
    # 1) Load image pixels via xradio (no sky coord math here)
    xds = read_image(str(fp), do_sky_coords=False, compute_mask=False)

    # 2) Open FITS header and extract 2D celestial WCS matching the image plane
    with fits.open(str(fp), memmap=True) as hdul:
        H = hdul[0].header
    w2d = WCS(H).celestial  # 2D (RA/Dec) WCS

    # 3) Compute RA/Dec at pixel centers (origin=0) with shape (m,l)
    ny = int(xds.sizes["m"])
    nx = int(xds.sizes["l"])
    yy, xx = np.indices((ny, nx), dtype=float)
    ra2d, dec2d = w2d.all_pix2world(xx, yy, 0)  # degrees, pixel centers

    # 4) Attach coords exactly equal to FITS WCS
    xds = xds.assign_coords(
        right_ascension=(("m", "l"), ra2d),
        declination=(("m", "l"), dec2d),
    )
    xds["right_ascension"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})
    xds["declination"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})

    # 5) Persist the exact celestial WCS header so we can re-create WCSAxes later without FITS
    cel_hdr = w2d.to_header()                 # astropy.io.fits.Header
    hdr_str = cel_hdr.tostring(sep="\n")
    xds.attrs["fits_wcs_header"] = hdr_str

    # 6) Hygiene + optional LM chunking
    xds.attrs.pop("history", None)  # keep attrs minimal
    for v in xds.data_vars:
        xds[v].encoding = {}

    if chunk_lm and {"l", "m"} <= set(xds.dims):
        xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})

    # ---- persist the exact celestial WCS header redundantly ----
    # 2) 0-D variable that always survives (NumPy ≥ 2.0: use np.bytes_)
    xds = xds.assign(wcs_header_str=((), np.bytes_(hdr_str.encode("utf-8"))))

    # 3) per-variable attrs (survive merges)
    for dv in xds.data_vars:
        xds[dv].attrs["fits_wcs_header"] = hdr_str

    # 4) also stash on coords for convenience
    xds["right_ascension"].attrs["fits_wcs_header"] = hdr_str
    xds["declination"].attrs["fits_wcs_header"] = hdr_str

    return xds


def _lm_shape(xds: xr.Dataset) -> Tuple[int, int]:
    """Return dataset LM shape as ``(m, l)``."""
    return int(xds.sizes["m"]), int(xds.sizes["l"])


def _select_reference_shape_index(shapes: List[Tuple[int, int]]) -> int:
    """Select deterministic reference index from LM shapes.

    Selection rule:
      1) largest pixel count (m * l)
      2) largest m
      3) largest l
      4) first occurrence on ties
    """
    if not shapes:
        msg = "Cannot select reference shape from empty list."
        raise RuntimeError(msg)

    best_idx = 0
    best_shape = shapes[0]
    best_score = (best_shape[0] * best_shape[1], best_shape[0], best_shape[1])

    for idx, shape in enumerate(shapes[1:], start=1):
        score = (shape[0] * shape[1], shape[0], shape[1])
        if score > best_score:
            best_idx = idx
            best_shape = shape
            best_score = score

    return best_idx


def _peek_lm_shape(fp: Path) -> Tuple[int, int]:
    """Return LM shape ``(m, l)`` from FITS using xradio, without full WCS attachment."""
    xds = read_image(str(fp), do_sky_coords=False, compute_mask=False)
    return int(xds.sizes["m"]), int(xds.sizes["l"])


def _load_global_lm_reference_dataset(
    by_time: Dict[str, List[Path]],
    fixed_dir: Path,
    *,
    chunk_lm: int,
    fix_headers_on_demand: bool,
) -> xr.Dataset:
    """Load the dataset whose LM grid has the largest shape across *all* time steps.

    Per-time reprojection alone is insufficient: different observation times can
    imply different in-step max shapes (e.g. only 3122² files in one step and
    mixed 4096²+3122² in another). A single global reference ensures every step
    normalizes to the same ``l``/``m`` so :func:`_assert_same_lm` can succeed.
    """
    candidates: List[Tuple[Path, Tuple[int, int]]] = []
    for tkey in sorted(by_time.keys()):
        files = by_time[tkey]
        if fix_headers_on_demand:
            fixed_paths = fix_fits_headers(files, fixed_dir, skip_existing=True)
        else:
            fixed_paths = _get_fixed_paths(files, fixed_dir)
        for fp in fixed_paths:
            candidates.append((fp, _peek_lm_shape(fp)))

    if not candidates:
        msg = "No FITS paths available to build a global LM reference."
        raise RuntimeError(msg)

    shapes = [sh for _, sh in candidates]
    win_idx = _select_reference_shape_index(shapes)
    ref_fp, ref_shape = candidates[win_idx]
    logger.info(
        "Global LM reference grid (m,l)=%s from %s",
        ref_shape,
        ref_fp.name,
    )
    return _load_for_combine(ref_fp, chunk_lm=chunk_lm)


def _regrid_to_reference_lm(xds: xr.Dataset, ref: xr.Dataset) -> xr.Dataset:
    """Interpolate ``xds`` onto ``ref``'s ``l`` / ``m`` coordinate grid.

    Uses linear interpolation in ``(l, m)``. Sky coordinates and persisted FITS
    WCS header metadata are taken from ``ref`` so the result is consistent with the
    reference pixel grid. No-op when ``xds`` already matches ``ref`` LM shape.

    Parameters
    ----------
    xds
        Source dataset (e.g. from :func:`_load_for_combine`).
    ref
        Reference dataset whose ``l`` and ``m`` define the target grid.

    Returns
    -------
    xarray.Dataset
        Dataset on the reference LM grid with writer-safe encodings cleared.

    Raises
    ------
    RuntimeError
        If interpolation fails (e.g. incompatible coordinates).
    """
    if _lm_shape(xds) == _lm_shape(ref):
        return xds

    # Materialize for scipy-backed interp; reference coords may be lazy.
    xds = xds.load()
    target_l = ref["l"].load() if hasattr(ref["l"].data, "compute") else ref["l"]
    target_m = ref["m"].load() if hasattr(ref["m"].data, "compute") else ref["m"]

    try:
        regridded = xds.interp(l=target_l, m=target_m, method="linear")
    except Exception as exc:
        msg = f"LM regridding onto reference grid failed: {exc}"
        raise RuntimeError(msg) from exc

    # Sky coords and WCS metadata match the reference physical grid.
    regridded = regridded.assign_coords(
        right_ascension=ref["right_ascension"].load()
        if hasattr(ref["right_ascension"].data, "compute")
        else ref["right_ascension"],
        declination=ref["declination"].load()
        if hasattr(ref["declination"].data, "compute")
        else ref["declination"],
    )

    hdr_str = ref.attrs.get("fits_wcs_header")
    if hdr_str is None and "wcs_header_str" in ref:
        raw = ref["wcs_header_str"].values.item()
        hdr_str = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)

    regridded.attrs.pop("history", None)
    if hdr_str is not None:
        regridded.attrs["fits_wcs_header"] = hdr_str
        regridded["right_ascension"].attrs["fits_wcs_header"] = hdr_str
        regridded["declination"].attrs["fits_wcs_header"] = hdr_str
        for dv in regridded.data_vars:
            regridded[dv].attrs["fits_wcs_header"] = hdr_str

    if "wcs_header_str" in ref:
        regridded["wcs_header_str"] = ref["wcs_header_str"].copy()

    for v in regridded.data_vars:
        regridded[v].encoding = {}

    return regridded


def _discover_groups(
    in_dir: Path,
    duplicate_resolver: Optional[Callable[[str, float, List[Path]], Path]] = None,
) -> Dict[str, List[Path]]:
    """Group input FITS by observation time and frequency (headers first, filename fallback).

    Parameters
    ----------
    in_dir : Path
        Directory containing input FITS files.
    duplicate_resolver
        Optional callback ``(time_key, frequency_hz, candidates) -> Path`` when multiple
        files share the same time and frequency.

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping time keys to lists of FITS file paths.
    """
    by_time: Dict[str, List[Path]] = {}
    by_time_freq: Dict[str, Dict[int, List[Path]]] = {}
    for f in sorted(in_dir.glob("*.fits")):
        time_key, frequency_hz, notes = _extract_group_metadata(f)
        if time_key is None:
            logger.warning(
                f"Skipping {f.name}: missing usable observation time in FITS headers and filename."
            )
            continue
        if frequency_hz is None:
            logger.warning(
                f"Could not determine frequency for {f.name}; duplicate detection disabled for this file."
            )
            by_time.setdefault(time_key, []).append(f)
            continue

        freq_key = int(round(frequency_hz))
        time_freq_map = by_time_freq.setdefault(time_key, {})
        candidates = time_freq_map.setdefault(freq_key, [])
        candidates.append(f)

        if len(candidates) > 1:
            duplicate_names = [p.name for p in candidates]
            if duplicate_resolver is None:
                msg = (
                    "Duplicate FITS files detected for the same time/frequency group "
                    f"(time={time_key}, frequency_hz={float(freq_key)}): {duplicate_names}. "
                    "Provide unique inputs or pass duplicate_resolver to choose one."
                )
                raise RuntimeError(msg)

            selected = duplicate_resolver(time_key, float(freq_key), candidates.copy())
            if selected not in candidates:
                msg = (
                    f"Duplicate resolver returned unknown file {selected} for "
                    f"time={time_key}, frequency_hz={float(freq_key)}."
                )
                raise RuntimeError(msg)

            by_time.setdefault(time_key, [])
            by_time[time_key] = [p for p in by_time[time_key] if p not in candidates]
            by_time[time_key].append(selected)
            # Replace the pending duplicate bucket so the next file with this (time, freq)
            # starts from the chosen file only. Otherwise `candidates` never shrinks and
            # grows as [f1, f2, f3, ...], re-invoking the resolver with stale paths and
            # widening the filter that strips `by_time[time_key]`.
            time_freq_map[freq_key] = [selected]
            logger.warning(
                "Duplicate FITS files for time=%s, frequency_hz=%.1f. Selected: %s. Candidates: %s",
                time_key,
                float(freq_key),
                selected.name,
                duplicate_names,
            )
            continue

        if notes:
            logger.warning(f"Using fallback metadata for {f.name}: {', '.join(notes)}")
        by_time.setdefault(time_key, []).append(f)

    for time_key, files in by_time.items():
        by_time[time_key] = sorted(files, key=_frequency_sort_tuple)
    return by_time


def _combine_time_step(
    files: List[Path],
    fixed_dir: Path,
    *,
    chunk_lm: int,
    fix_headers_on_demand: bool = True,
    lm_reference_ds: Optional[xr.Dataset] = None,
) -> Tuple[xr.Dataset, List[float]]:
    """Create a single-time dataset by combining frequency slices from subbands.

    Parameters
    ----------
    files : List[Path]
        List of FITS file paths for a single time step.
    fixed_dir : Path
        Directory to place generated ``*_fixed.fits`` files.
    chunk_lm : int
        LM chunk size for in-memory xarray datasets.
    fix_headers_on_demand : bool, optional
        If True, fix headers on-demand if they don't exist. If False,
        assume headers are already fixed. Default is True.
    lm_reference_ds : xarray.Dataset, optional
        If provided, regrid all slices to this dataset's ``l``/``m`` grid (used for
        a conversion-wide max-shape reference). If omitted, the largest slice in
        this time step is the reference.

    Returns
    -------
    Tuple[xr.Dataset, List[float]]
        Tuple of (combined dataset, sorted list of unique frequencies in Hz).
    """
    if fix_headers_on_demand:
        # Fix headers if needed (skips existing fixed files)
        fixed_paths = fix_fits_headers(files, fixed_dir, skip_existing=True)
    else:
        # Just get the paths to already-fixed files
        fixed_paths = _get_fixed_paths(files, fixed_dir)

    xds_list: List[xr.Dataset] = []
    freqs_seen: List[float] = []
    for fp in fixed_paths:
        xds = _load_for_combine(fp, chunk_lm=chunk_lm)
        fvals = np.atleast_1d(xds.frequency.values)
        freqs_seen.extend([float(f) for f in fvals])
        xds_list.append(xds)

    lm_shapes = [_lm_shape(xds) for xds in xds_list]
    reference_idx = _select_reference_shape_index(lm_shapes)
    unique_shapes = sorted(set(lm_shapes))

    if lm_reference_ds is not None:
        ref_ds = lm_reference_ds
        if len(unique_shapes) > 1 or any(_lm_shape(xds) != _lm_shape(ref_ds) for xds in xds_list):
            logger.info(
                "Using global LM reference grid (m,l)=%s; this time step shapes: %s",
                _lm_shape(ref_ds),
                unique_shapes,
            )
    else:
        ref_ds = xds_list[reference_idx]
        if len(unique_shapes) > 1:
            logger.info(
                "Detected mixed LM shapes %s; selected reference shape %s from %s",
                unique_shapes,
                lm_shapes[reference_idx],
                fixed_paths[reference_idx].name,
            )
    for i, xds in enumerate(xds_list):
        if _lm_shape(xds) != _lm_shape(ref_ds):
            logger.info(
                "Regridding %s from LM shape %s onto reference %s",
                fixed_paths[i].name,
                _lm_shape(xds),
                _lm_shape(ref_ds),
            )
        xds_list[i] = _regrid_to_reference_lm(xds, ref_ds)

    try:
        xds_t = xr.combine_by_coords(
            xds_list,
            combine_attrs="drop",
            data_vars="minimal",
            coords="minimal",
            compat="no_conflicts",
        )
    except Exception:
        # Fallback requires each subband to have frequency size == 1
        for ds in xds_list:
            if "frequency" in ds.dims and ds.sizes["frequency"] != 1:
                msg = "A subband has frequency dimension != 1; cannot concat."
                raise RuntimeError(msg)
        xds_t = xr.concat(xds_list, dim="frequency")

    if "frequency" in xds_t.coords:
        xds_t = xds_t.sortby("frequency")
    if "time" in xds_t.coords:
        xds_t = xds_t.sortby("time")

    xds_t.attrs = {}
    for v in xds_t.data_vars:
        xds_t[v].encoding = {}

    return xds_t, sorted(set(freqs_seen))


def _assert_same_lm(
    reference: Tuple[NDArray[np.floating], NDArray[np.floating]],
    current: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> None:
    """Ensure the LM grids match across time steps.

    Parameters
    ----------
    reference : Tuple[NDArray[np.floating], NDArray[np.floating]]
        Reference (l, m) grid arrays.
    current : Tuple[NDArray[np.floating], NDArray[np.floating]]
        Current (l, m) grid arrays to compare.

    Raises
    ------
    RuntimeError
        If l or m grids differ across time steps.
    """
    ref_l, ref_m = reference[0], reference[1]
    cur_l, cur_m = current[0], current[1]
    if ref_l.shape != cur_l.shape or ref_m.shape != cur_m.shape:
        msg = (
            "l/m coordinate length mismatch across time steps "
            f"(l: {ref_l.shape} vs {cur_l.shape}, m: {ref_m.shape} vs {cur_m.shape}). "
            "After mixed-resolution normalization, every time step must share one LM grid."
        )
        raise RuntimeError(msg)
    same_l = np.allclose(ref_l, cur_l)
    same_m = np.allclose(ref_m, cur_m)
    if not (same_l and same_m):
        raise RuntimeError("l/m grids differ across times; aborting to avoid misalignment.")


def _write_or_append_zarr(xds_t: xr.Dataset, out_zarr: Path, first_write: bool) -> None:
    """Safe write/append to Zarr store.

    First write: write directly.
    Append: read existing lazily, build combined, write to a TEMP path,
    then atomically swap into place so we never delete the source store
    while still reading from it.

    Parameters
    ----------
    xds_t : xr.Dataset
        Dataset to write or append.
    out_zarr : Path
        Path to output Zarr store.
    first_write : bool
        If True, write a new Zarr store; if False, append to existing.
    """
    from shutil import rmtree, move

    if first_write or (not out_zarr.exists()):
        write_image(xds_t, str(out_zarr), out_format="zarr", overwrite=True)
        return

    # 1) Open existing lazily
    existing = xr.open_zarr(str(out_zarr))
    existing.attrs = {}
    for v in existing.data_vars:
        existing[v].encoding = {}

    # 2) Sort for determinism
    if "time" in existing.coords: existing = existing.sortby("time")
    if "frequency" in existing.coords: existing = existing.sortby("frequency")
    if "time" in xds_t.coords:       xds_t = xds_t.sortby("time")
    if "frequency" in xds_t.coords:  xds_t = xds_t.sortby("frequency")

    # 3) Build combined (still lazy)
    combined = xr.concat([existing, xds_t], dim="time")
    if "time" in combined.coords:      combined = combined.sortby("time")
    if "frequency" in combined.coords: combined = combined.sortby("frequency")

    # 4) Write to a temporary path (so the original store remains readable)
    tmp = out_zarr.with_suffix(out_zarr.suffix + ".tmpwrite")
    if tmp.exists():
        rmtree(tmp)
    write_image(combined, str(tmp), out_format="zarr", overwrite=True)

    # 5) Swap: remove old store, move tmp into place
    if out_zarr.exists():
        rmtree(out_zarr)
    move(str(tmp), str(out_zarr))



def convert_fits_dir_to_zarr(
    input_dir: str | Path,
    out_dir: str | Path,
    zarr_name: str = "ovro_lwa_full_lm_only.zarr",
    fixed_dir: str | Path = "fixed_fits",
    chunk_lm: int = 1024,
    rebuild: bool = False,
    fix_headers_on_demand: bool = True,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    duplicate_resolver: Optional[Callable[[str, float, List[Path]], Path]] = None,
) -> Path:
    """Convert all matching FITS in a directory into a single LM-only Zarr store.

    Parameters
    ----------
    input_dir
        Directory containing input FITS files.
    out_dir
        Directory where the Zarr store will be written.
    zarr_name
        Name of the Zarr store directory (under ``out_dir``).
    fixed_dir
        Directory to place generated ``*_fixed.fits`` files.
    chunk_lm
        Optional LM chunk size for the in-memory xarray datasets (0 disables).
    rebuild
        If True, overwrite any existing Zarr; otherwise append to it.
    fix_headers_on_demand
        If True, fix FITS headers on-demand during conversion if they don't exist.
        If False, assume headers are already fixed using :func:`fix_fits_headers`.
        Default is True.
    progress_callback
        Optional callback function for progress reporting. Should accept
        (stage: str, current: int, total: int, message: str).
    duplicate_resolver
        Optional callback to resolve duplicate files that map to the same
        time/frequency group. Signature: ``(time_key, frequency_hz, candidates) -> selected_path``.

    Returns
    -------
    Path
        Path to the resulting Zarr store directory.

    Raises
    ------
    FileNotFoundError
        If no matching FITS files are found.
    RuntimeError
        If LM grids differ across time steps.
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir = Path(fixed_dir)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = out_dir / zarr_name

    by_time = _discover_groups(input_dir, duplicate_resolver=duplicate_resolver)
    total_files = sum(len(v) for v in by_time.values())
    logger.info(f"Discovered {total_files} FITS across {len(by_time)} time step(s).")
    for k, v in by_time.items():
        freqs_hz = [_extract_group_metadata(p)[1] for p in v]
        logger.info(f"  time {k}: {len(v)} file(s), frequencies (Hz): {freqs_hz}")

    if not by_time:
        raise FileNotFoundError(f"No matching FITS found in {input_dir}")

    lm_ref_ds = _load_global_lm_reference_dataset(
        by_time,
        fixed_dir,
        chunk_lm=chunk_lm,
        fix_headers_on_demand=fix_headers_on_demand,
    )
    lm_reference = (lm_ref_ds["l"].values.copy(), lm_ref_ds["m"].values.copy())

    # Decide whether we write a fresh store or append to an existing one
    first_write = not (out_zarr.exists() and not rebuild)

    total_time_steps = len(by_time)
    for idx, tkey in enumerate(sorted(by_time.keys())):
        files = by_time[tkey]

        logger.info(f"[read/combine] time {tkey}")
        xds_t, freqs = _combine_time_step(
            files,
            fixed_dir,
            chunk_lm=chunk_lm,
            fix_headers_on_demand=fix_headers_on_demand,
            lm_reference_ds=lm_ref_ds,
        )
        logger.info(f"  combined dims: {dict(xds_t.dims)}")
        logger.info(f"  combined freqs (Hz): {freqs[:8]}{' ...' if len(freqs) > 8 else ''}")

        lm_current = (xds_t["l"].values, xds_t["m"].values)
        _assert_same_lm(lm_reference, lm_current)
        logger.info("  l/m grid matches global reference")

        logger.info(f"[{'write new' if first_write else 'append'}] {out_zarr}")
        _write_or_append_zarr(xds_t, out_zarr, first_write=first_write)
        first_write = False

        # Report progress after completing this time step
        if progress_callback:
            progress_callback(
                "converting",
                idx + 1,
                total_time_steps,
                f"Completed time step {idx + 1}/{total_time_steps}"
            )

    logger.info(f"[done] All times appended into: {out_zarr}")
    return out_zarr
