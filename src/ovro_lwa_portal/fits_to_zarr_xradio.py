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
* Filenames must contain ``YYYYMMDD_HHMMSS`` (observation timestamp) and
  ``<freq>MHz`` (subband frequency). All ``.fits`` files in the input
  directory are assumed to be valid images for grouping.
* LM grids must match across time steps; a mismatch raises a RuntimeError.
* Each time step is appended incrementally via ``to_zarr(append_dim="time")``;
  missing frequency subbands are NaN-filled so every write has a consistent
  frequency grid.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray
from xradio.image import read_image, write_image

__all__ = ["convert_fits_dir_to_zarr", "fix_fits_headers"]

logger = logging.getLogger(__name__)

_DATETIME_RE = re.compile(r"(?P<date>\d{8})_(?P<hms>\d{6})")
MHZ_RE = re.compile(r"(\d+)MHz")


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
    for f in sorted(files, key=_mhz_from_name):
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

    for f in sorted(files, key=_mhz_from_name):
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


def _discover_groups(in_dir: Path) -> Dict[str, List[Path]]:
    """Group input FITS by time key ``YYYYMMDD_HHMMSS``.

    Every ``.fits`` file in *in_dir* is considered a candidate.  The
    observation timestamp is extracted by searching for an
    ``YYYYMMDD_HHMMSS`` pattern anywhere in the filename.  Files whose
    names do not contain a recognisable timestamp are silently skipped.

    Parameters
    ----------
    in_dir : Path
        Directory containing input FITS files.

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping time keys to lists of FITS file paths.
    """
    by_time: Dict[str, List[Path]] = {}
    for f in sorted(in_dir.glob("*.fits")):
        m = _DATETIME_RE.search(f.name)
        if not m:
            continue
        key = f"{m.group('date')}_{m.group('hms')}"
        by_time.setdefault(key, []).append(f)
    return by_time


def _discover_freq_grid(
    by_time: Dict[str, List[Path]],
    fixed_dir: Path,
    fix_headers_on_demand: bool,
) -> NDArray[np.floating]:
    """Pre-scan one FITS header per unique subband to build the full frequency grid.

    Only headers are read (via ``memmap``), so no pixel data is loaded.

    Parameters
    ----------
    by_time : Dict[str, List[Path]]
        Time-grouped FITS file paths from :func:`_discover_groups`.
    fixed_dir : Path
        Directory containing (or to create) ``*_fixed.fits`` files.
    fix_headers_on_demand : bool
        If True, fix headers for representative files if not already done.

    Returns
    -------
    NDArray[np.floating]
        Sorted array of unique frequency values in Hz.
    """
    representatives: Dict[int, Path] = {}
    for files in by_time.values():
        for f in files:
            mhz = _mhz_from_name(f)
            representatives.setdefault(mhz, f)

    freq_hz: List[float] = []
    for mhz in sorted(representatives):
        f = representatives[mhz]

        if fix_headers_on_demand:
            fixed_paths = fix_fits_headers([f], fixed_dir, skip_existing=True)
            f = fixed_paths[0]
        elif not f.name.endswith("_fixed.fits"):
            f = fixed_dir / (f.stem + "_fixed.fits")

        with fits.open(str(f), memmap=True) as hdul:
            freq_hz.append(float(hdul[0].header.get("CRVAL3", mhz * 1e6)))

    return np.array(sorted(freq_hz))


def _combine_time_step(
    files: List[Path], fixed_dir: Path, *, chunk_lm: int, fix_headers_on_demand: bool = True
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
    same_l = np.allclose(reference[0], current[0])
    same_m = np.allclose(reference[1], current[1])
    if not (same_l and same_m):
        raise RuntimeError("l/m grids differ across times; aborting to avoid misalignment.")


def convert_fits_dir_to_zarr(
    input_dir: str | Path,
    out_dir: str | Path,
    zarr_name: str = "ovro_lwa_full_lm_only.zarr",
    fixed_dir: str | Path = "fixed_fits",
    chunk_lm: int = 1024,
    rebuild: bool = False,
    fix_headers_on_demand: bool = True,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
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

    by_time = _discover_groups(input_dir)
    total_files = sum(len(v) for v in by_time.values())
    logger.info(f"Discovered {total_files} FITS across {len(by_time)} time step(s).")
    for k, v in by_time.items():
        mhz_sorted = sorted(_mhz_from_name(p) for p in v)
        logger.info(f"  time {k}: {len(v)} file(s), subbands (MHz): {mhz_sorted}")

    if not by_time:
        raise FileNotFoundError(f"No matching FITS found in {input_dir}")

    # Pre-scan to build the full frequency grid across all time steps.
    # Only reads one FITS header per unique subband — no pixel data.
    freq_grid = _discover_freq_grid(by_time, fixed_dir, fix_headers_on_demand)
    logger.info(f"Frequency grid ({len(freq_grid)} subbands): {freq_grid.tolist()}")

    first_write = rebuild or not out_zarr.exists()
    lm_reference: Tuple[NDArray[np.floating], NDArray[np.floating]] | None = None

    total_time_steps = len(by_time)
    for idx, tkey in enumerate(sorted(by_time.keys())):
        files = by_time[tkey]

        logger.info(f"[read/combine] time {tkey}")
        xds_t, freqs = _combine_time_step(
            files, fixed_dir, chunk_lm=chunk_lm, fix_headers_on_demand=fix_headers_on_demand
        )
        logger.info(f"  combined dims: {dict(xds_t.dims)}")
        logger.info(f"  combined freqs (Hz): {freqs[:8]}{' ...' if len(freqs) > 8 else ''}")

        # Pad to the full frequency grid so every write has the same shape.
        xds_t = xds_t.reindex(frequency=freq_grid)

        lm_current = (xds_t["l"].values, xds_t["m"].values)
        if lm_reference is None:
            lm_reference = (lm_current[0].copy(), lm_current[1].copy())
            logger.info("  stored l/m grid as reference")
        else:
            _assert_same_lm(lm_reference, lm_current)
            logger.info("  l/m grid matches reference")

        if first_write:
            logger.info(f"[write new] {out_zarr}")
            write_image(xds_t, str(out_zarr), out_format="zarr", overwrite=True)
            first_write = False
        else:
            logger.info(f"[append] {out_zarr}")
            xds_t.to_zarr(str(out_zarr), mode="a", append_dim="time")

        if progress_callback:
            progress_callback(
                "converting",
                idx + 1,
                total_time_steps,
                f"Completed time step {idx + 1}/{total_time_steps}",
            )

    logger.info(f"[done] All times appended into: {out_zarr}")
    return out_zarr
