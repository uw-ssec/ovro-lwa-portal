"""xradio-powered FITS → Zarr conversion for OVRO-LWA.

This module converts per-time, per-subb and FITS images into a single LM-only Zarr store.
It uses `xradio` for FITS I/O and Zarr writing, and enforces deterministic ordering by
sorting on frequency and time. It also materializes FITS scaling (BSCALE/BZERO) and adds
a minimal set of header keywords so `xradio` can parse images reliably.

Library usage
-------------
    from ovro_lwa_portal.ingest.fit_to_zarr_xradio import convert_fits_dir_to_zarr
    out = convert_fits_dir_to_zarr(
        input_dir="/path/to/fits",
        out_dir="zarr_out",
        zarr_name="ovro_lwa_full_lm_only.zarr",
        fixed_dir="fixed_fits",
        chunk_lm=1024,
        rebuild=False,
    )

Notes
-----
* The code assumes filenames of the form:
  YYYYMMDD_HHMMSS_<SB>MHz_averaged_* -I-image[ _fixed].fits
* LM grids must match across time steps; a mismatch raises a RuntimeError.
* On append, the existing Zarr is read and re-written with the appended time step.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from astropy.io import fits
from numpy.typing import NDArray
from xradio.image import read_image, write_image

__all__ = ["convert_fits_dir_to_zarr"]

logger = logging.getLogger(__name__)

# Match: 20240524_050019_41MHz_averaged_...-I-image(.fits|_fixed.fits)
PAT = re.compile(
    r"^(?P<date>\d{8})_(?P<hms>\d{6})_(?P<sb>\d+)MHz_averaged_.*-I-image(?:_fixed)?\.fits$"
)
MHZ_RE = re.compile(r"_(\d+)MHz_")


def _mhz_from_name(p: Path) -> int:
    """Extract the subband MHz from a filename; return a large sentinel if absent."""
    m = MHZ_RE.search(p.name)
    return int(m.group(1)) if m else 10**9


def _fix_headers(path_in: Path, path_out: Path) -> None:
    """Write a *_fixed.fits with BSCALE/BZERO applied and minimal WCS/spectral keys.

    Adds/ensures:
      RESTFREQ/RESTFRQ, SPECSYS=LSRK, TIMESYS=UTC, RADESYS=FK5, LATPOLE=90,
      identity PC matrix for LM, nominal beam (BMAJ/BMIN=6 arcmin), BUNIT=Jy/beam.
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


def _load_for_combine(fp: Path, *, chunk_lm: int = 1024) -> xr.Dataset:
    """Load a fixed FITS image via xradio in LM-only mode and clean metadata."""
    xds = read_image(str(fp), do_sky_coords=False, compute_mask=False)

    # Drop sky coords if they slipped in
    drop = [k for k in ("right_ascension", "declination", "velocity") if k in xds.coords]
    if drop:
        xds = xds.reset_coords(drop, drop=True)

    # Hygiene
    xds.attrs = {}
    for v in xds.data_vars:
        xds[v].encoding = {}

    # Optional LM chunking
    if "l" in xds.dims and "m" in xds.dims and chunk_lm:
        xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})

    return xds


def _discover_groups(in_dir: Path) -> Dict[str, List[Path]]:
    """Group input FITS by time key 'YYYYMMDD_HHMMSS' using PAT."""
    by_time: Dict[str, List[Path]] = {}
    for f in sorted(in_dir.glob("*.fits")):
        m = PAT.match(f.name)
        if not m:
            continue
        key = f"{m.group('date')}_{m.group('hms')}"
        by_time.setdefault(key, []).append(f)
    return by_time


def _combine_time_step(
    files: List[Path], fixed_dir: Path, *, chunk_lm: int
) -> Tuple[xr.Dataset, List[float]]:
    """Create a single-time dataset by combining frequency slices from subbands."""
    fixed_paths: List[Path] = []
    for f in sorted(files, key=_mhz_from_name):
        if f.name.endswith("_fixed.fits"):
            fixed_paths.append(f)
        else:
            fixed = fixed_dir / (f.stem + "_fixed.fits")
            if not fixed.exists():
                _fix_headers(f, fixed)
            fixed_paths.append(fixed)

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
    """Ensure the LM grids match across time steps."""
    same_l = np.allclose(reference[0], current[0])
    same_m = np.allclose(reference[1], current[1])
    if not (same_l and same_m):
        raise RuntimeError("l/m grids differ across times; aborting to avoid misalignment.")


def _write_or_append_zarr(xds_t: xr.Dataset, out_zarr: Path, first_write: bool) -> None:
    """
    Safe write/append:
      - First write: write directly.
      - Append: read existing lazily, build combined, write to a TEMP path,
        then atomically swap into place so we never delete the source store
        while still reading from it.
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
    if "time" in existing.coords:    existing = existing.sortby("time")
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
    logger.info("Discovered %d FITS across %d time step(s).", total_files, len(by_time))
    for k, v in by_time.items():
        mhz_sorted = sorted(_mhz_from_name(p) for p in v)
        logger.info("  time %s: %d file(s), subbands (MHz): %s", k, len(v), mhz_sorted)

    if not by_time:
        raise FileNotFoundError(f"No matching FITS found in {input_dir}")

    # Decide whether we write a fresh store or append to an existing one
    first_write = not (out_zarr.exists() and not rebuild)
    lm_reference: Tuple[NDArray[np.floating], NDArray[np.floating]] | None = None

    for tkey in sorted(by_time.keys()):
        files = by_time[tkey]
        logger.info("[read/combine] time %s", tkey)
        xds_t, freqs = _combine_time_step(files, fixed_dir, chunk_lm=chunk_lm)
        logger.info("  combined dims: %s", dict(xds_t.dims))
        logger.info("  combined freqs (Hz): %s%s",
                    freqs[:8], " ..." if len(freqs) > 8 else "")

        lm_current = (xds_t["l"].values, xds_t["m"].values)
        if lm_reference is None:
            lm_reference = (lm_current[0].copy(), lm_current[1].copy())
            logger.info("  stored l/m grid as reference")
        else:
            _assert_same_lm(lm_reference, lm_current)
            logger.info("  l/m grid matches reference")

        logger.info("[%s] %s", "write new" if first_write else "append", out_zarr)
        _write_or_append_zarr(xds_t, out_zarr, first_write=first_write)
        first_write = False

    logger.info("[done] All times appended into: %s", out_zarr)
    return out_zarr
