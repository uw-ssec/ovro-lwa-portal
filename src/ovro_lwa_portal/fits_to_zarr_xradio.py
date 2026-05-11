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
* Discovery groups files by observation time and by a **10~kHz binned** frequency key
  from FITS headers (so Hz-level jitter does not create extra ``frequency`` planes for
  one subband). Multiple files in the same (time, bin) without a duplicate resolver
  keep the first and skip the rest (with a warning). Filename parsing is only used as
  a fallback when header metadata is missing.
* LM grids must match across time steps after global and per-step mixed-resolution normalization;
  a mismatch raises a RuntimeError.
* Within a single time step, mixed LM shapes are regridded onto the reference grid before combine.
* Before Zarr write, ``l``/``m`` are rechunked to uniform sizes so the store does not hit
  Dask/Zarr constraints on irregular spatial chunk boundaries after ``combine``/``concat``.
* On append, only the pending time step is appended along the ``time`` dimension.
"""

from __future__ import annotations

import os
import logging
import re
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import xarray as xr
import zarr
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from numpy.typing import NDArray
from xradio.image import read_image, write_image

__all__ = [
    "convert_fits_dir_to_zarr",
    "fix_fits_headers",
    "validate_zarr_store",
    "repair_zarr_store",
]

logger = logging.getLogger(__name__)

# Grouping key for "same subband" when discovering FITS. Raw ``int(round(hz))`` can differ
# by 1--1000+ Hz between files for the same 55~MHz (etc.) product; that produced multiple
# ``frequency`` planes in the Zarr from one logical band. Bins of 10~kHz merge that jitter
# while keeping distinct LWA subbands (MHz-scale) separate.
_DISCOVERY_FREQ_BIN_HZ: float = 10_000.0

# Cache expensive sky-coordinate transforms keyed by celestial WCS header + LM shape.
# Keep this bounded so long-running conversions do not accumulate unlimited RA/Dec grids.
_SKY_COORD_CACHE_MAX_ENTRIES = 64
_SKY_COORD_CACHE: "OrderedDict[Tuple[int, int, str], Tuple[NDArray[np.floating], NDArray[np.floating], str]]" = OrderedDict()


def _sky_coord_cache_get(
    key: Tuple[int, int, str],
) -> Optional[Tuple[NDArray[np.floating], NDArray[np.floating], str]]:
    """Lookup in LRU sky-coordinate cache and refresh recency on hit."""
    cached = _SKY_COORD_CACHE.get(key)
    if cached is not None:
        _SKY_COORD_CACHE.move_to_end(key)
    return cached


def _sky_coord_cache_set(
    key: Tuple[int, int, str],
    value: Tuple[NDArray[np.floating], NDArray[np.floating], str],
) -> None:
    """Insert into LRU sky-coordinate cache and evict oldest entry when full."""
    _SKY_COORD_CACHE[key] = value
    _SKY_COORD_CACHE.move_to_end(key)
    if len(_SKY_COORD_CACHE) > _SKY_COORD_CACHE_MAX_ENTRIES:
        _SKY_COORD_CACHE.popitem(last=False)


MHZ_RE = re.compile(r"_(\d+)MHz_")
_IMAGE_TIME_RE = re.compile(r"-image-(\d{8})_(\d{6})")


def _time_key_from_name(p: Path) -> Optional[str]:
    """Extract a normalized ``YYYYMMDD_HHMMSS`` key from basename when possible."""
    m = PAT.match(p.name)
    if m:
        return f"{m.group('date')}_{m.group('hms')}"
    m_img = _IMAGE_TIME_RE.search(p.name)
    if not m_img:
        return None
    return f"{m_img.group(1)}_{m_img.group(2)}"


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

    This project requires ``DATE-OBS`` to be present and parseable.
    ``TIME-OBS`` is used only when ``DATE-OBS`` is date-only (no ``T``).

    Returns ``None`` when no usable ``DATE-OBS`` timestamp is found.
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

    return None


def _normalize_time_key(value: object) -> Optional[str]:
    """Normalize mixed time representations to ``YYYYMMDD_HHMMSS`` in UTC.

    This helper is used to compare discovery keys against time values loaded
    from an existing Zarr store.
    """
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        # Many existing OVRO-LWA Zarr stores encode time as MJD floats.
        try:
            return Time(float(value), format="mjd", scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            return None

    if isinstance(value, (bytes, np.bytes_)):
        text_value = value.decode("utf-8").strip()
    elif isinstance(value, str):
        text_value = value.strip()
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        dt64_s = value.astype("datetime64[s]")
        iso = np.datetime_as_string(dt64_s, unit="s", timezone="UTC")
        try:
            return Time(iso, format="isot", scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            return None
    else:
        text_value = str(value).strip()

    if not text_value:
        return None

    if re.match(r"^\d{8}_\d{6}$", text_value):
        return text_value

    for fmt in ("isot", "fits"):
        try:
            return Time(text_value, format=fmt, scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            continue

    try:
        dt64 = np.datetime64(text_value)
        if np.isnat(dt64):
            return None
        dt64_s = dt64.astype("datetime64[s]")
        iso = np.datetime_as_string(dt64_s, unit="s", timezone="UTC")
        return Time(iso, format="isot", scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
    except Exception:
        return None


def _existing_time_keys_from_zarr(out_zarr: Path) -> set[str]:
    """Read and normalize timestep keys from an existing Zarr store."""
    try:
        xds = xr.open_zarr(str(out_zarr), consolidated=False)
    except Exception as exc:
        try:
            zg = zarr.open_group(str(out_zarr), mode="r")
            time_arr = zg["time"][:]
            fallback_keys: set[str] = set()
            for raw_value in np.atleast_1d(time_arr):
                key = _normalize_time_key(raw_value)
                if key is None:
                    msg = (
                        f"Could not normalize time value {raw_value!r} in existing Zarr store {out_zarr}; "
                        "cannot resume safely."
                    )
                    raise RuntimeError(msg)
                fallback_keys.add(key)
            return fallback_keys
        except Exception:
            msg = f"Could not open existing Zarr store {out_zarr}: {exc}"
            raise RuntimeError(msg) from exc

    try:
        if "time" not in xds.coords:
            msg = f"Existing Zarr store {out_zarr} has no 'time' coordinate; cannot resume safely."
            raise RuntimeError(msg)

        keys: set[str] = set()
        for raw_value in np.atleast_1d(xds["time"].values):
            key = _normalize_time_key(raw_value)
            if key is None:
                msg = (
                    f"Could not normalize time value {raw_value!r} in existing Zarr store {out_zarr}; "
                    "cannot resume safely."
                )
                raise RuntimeError(msg)
            keys.add(key)
        return keys
    finally:
        xds.close()


def _txn_dir_for_store(out_zarr: Path) -> Path:
    """Return sidecar transaction directory for one output Zarr store."""
    return out_zarr.parent / f".{out_zarr.name}.append_txn"


def _committed_time_keys_from_txn(out_zarr: Path) -> set[str]:
    """Read committed time keys from sidecar transaction markers."""
    txn_dir = _txn_dir_for_store(out_zarr)
    if not txn_dir.exists():
        return set()
    committed: set[str] = set()
    for marker in txn_dir.glob("*.committed"):
        committed.add(marker.stem)
    return committed


def _mark_time_in_progress(out_zarr: Path, time_key: str) -> None:
    """Create an in-progress transaction marker for one time key."""
    txn_dir = _txn_dir_for_store(out_zarr)
    txn_dir.mkdir(parents=True, exist_ok=True)
    (txn_dir / f"{time_key}.inprogress").write_text("in_progress\n", encoding="utf-8")


def _mark_time_committed(out_zarr: Path, time_key: str) -> None:
    """Mark a time key as committed and clear its in-progress marker."""
    txn_dir = _txn_dir_for_store(out_zarr)
    txn_dir.mkdir(parents=True, exist_ok=True)
    in_progress = txn_dir / f"{time_key}.inprogress"
    committed = txn_dir / f"{time_key}.committed"
    if in_progress.exists():
        in_progress.unlink()
    committed.write_text("committed\n", encoding="utf-8")


def _validate_time_axis_consistency_zarr(out_zarr: Path) -> None:
    """Ensure all Zarr arrays with a ``time`` dimension share one length."""
    try:
        zg = zarr.open_group(str(out_zarr), mode="r")
    except Exception:
        # Path exists but is not yet a readable Zarr group (e.g., test scaffold dir).
        return
    buckets: Dict[int, List[str]] = {}
    for name in zg.array_keys():
        arr = zg[name]
        dims_attr = arr.attrs.get("_ARRAY_DIMENSIONS")
        if dims_attr is None:
            continue
        dims = [dims_attr] if isinstance(dims_attr, str) else [str(d) for d in dims_attr]
        if "time" not in dims:
            continue
        time_axis = dims.index("time")
        time_len = int(arr.shape[time_axis])
        buckets.setdefault(time_len, []).append(name)

    if len(buckets) <= 1:
        return

    details = "; ".join(f"time={k}: {sorted(v)}" for k, v in sorted(buckets.items()))
    msg = (
        f"Existing Zarr store {out_zarr} has inconsistent time-axis lengths across arrays ({details}). "
        "This usually indicates an interrupted append. Repair the store or rebuild before resuming."
    )
    raise RuntimeError(msg)


def _time_axis_length_buckets(out_zarr: Path) -> Dict[int, List[str]]:
    """Return mapping of time-axis length -> array names."""
    zg = zarr.open_group(str(out_zarr), mode="r")
    buckets: Dict[int, List[str]] = {}
    for name in sorted(zg.array_keys()):
        arr = zg[name]
        dims_attr = arr.attrs.get("_ARRAY_DIMENSIONS")
        if dims_attr is None:
            continue
        dims = [dims_attr] if isinstance(dims_attr, str) else [str(d) for d in dims_attr]
        if "time" not in dims:
            continue
        time_axis = dims.index("time")
        time_len = int(arr.shape[time_axis])
        buckets.setdefault(time_len, []).append(name)
    return buckets


def validate_zarr_store(out_zarr: str | Path) -> Dict[str, object]:
    """Validate time-axis consistency for a Zarr store.

    Returns a report dictionary with per-length buckets and consistency flag.
    """
    out_zarr = Path(out_zarr)
    if not out_zarr.exists():
        msg = f"Zarr store does not exist: {out_zarr}"
        raise FileNotFoundError(msg)

    buckets = _time_axis_length_buckets(out_zarr)
    consistent = len(buckets) <= 1
    report: Dict[str, object] = {
        "store": str(out_zarr),
        "consistent": consistent,
        "time_length_buckets": {k: sorted(v) for k, v in sorted(buckets.items())},
    }
    if not consistent:
        details = "; ".join(f"time={k}: {sorted(v)}" for k, v in sorted(buckets.items()))
        report["message"] = (
            f"Inconsistent time-axis lengths across arrays ({details}). "
            "This usually indicates an interrupted append."
        )
    return report


def repair_zarr_store(
    out_zarr: str | Path,
    *,
    fits_dir: str | Path | None = None,
    backup_suffix: str = ".backup-before-repair",
) -> Dict[str, object]:
    """Repair inconsistent time-axis lengths and optionally refresh WCS headers.

    The store is backed up before modification.
    """
    out_zarr = Path(out_zarr)
    if not out_zarr.exists():
        msg = f"Zarr store does not exist: {out_zarr}"
        raise FileNotFoundError(msg)

    backup_path = out_zarr.with_name(out_zarr.name + backup_suffix)
    if backup_path.exists():
        msg = f"Backup path already exists: {backup_path}"
        raise FileExistsError(msg)

    pre = validate_zarr_store(out_zarr)
    buckets = pre["time_length_buckets"]
    if not buckets:
        msg = f"No arrays with a time axis found in {out_zarr}"
        raise RuntimeError(msg)
    repaired_len = min(int(k) for k in buckets.keys())

    shutil.copytree(out_zarr, backup_path)
    zg = zarr.open_group(str(out_zarr), mode="a")

    truncated: List[str] = []
    for name in sorted(zg.array_keys()):
        arr = zg[name]
        dims_attr = arr.attrs.get("_ARRAY_DIMENSIONS")
        if dims_attr is None:
            continue
        dims = [dims_attr] if isinstance(dims_attr, str) else [str(d) for d in dims_attr]
        if "time" not in dims:
            continue
        time_axis = dims.index("time")
        if int(arr.shape[time_axis]) <= repaired_len:
            continue

        slicer = [slice(None)] * arr.ndim
        slicer[time_axis] = slice(0, repaired_len)
        data = arr[tuple(slicer)]
        attrs = dict(arr.attrs)
        chunks = arr.chunks
        dtype = arr.dtype
        del zg[name]
        new_arr = zg.create_dataset(
            name,
            data=data.astype(dtype, copy=False),
            chunks=chunks if len(chunks) == data.ndim else True,
            overwrite=True,
        )
        new_arr.attrs.update(attrs)
        truncated.append(name)

    rewritten_wcs_rows = 0
    if fits_dir is not None and "wcs_header_str" in zg and "time" in zg:
        fits_dir = Path(fits_dir)
        if fits_dir.exists():
            by_time = _discover_groups(fits_dir)
            z_time = np.atleast_1d(zg["time"][:])
            z_time_keys = [_normalize_time_key(v) for v in z_time[:repaired_len]]
            wcs_arr = zg["wcs_header_str"]
            n_freq = int(wcs_arr.shape[1]) if wcs_arr.ndim >= 2 else 0
            for ti, tkey in enumerate(z_time_keys):
                if tkey is None:
                    continue
                files = by_time.get(tkey, [])
                if not files:
                    continue
                files = sorted(files, key=_frequency_sort_tuple)
                row = wcs_arr[ti, :].copy()
                for fi, fp in enumerate(files[:n_freq]):
                    try:
                        with fits.open(str(fp), memmap=True) as hdul:
                            hdr = hdul[0].header
                        s = WCS(hdr).celestial.to_header().tostring(sep="\n")
                        row[fi] = np.bytes_(s.encode("utf-8"))
                    except Exception:
                        continue
                wcs_arr[ti, :] = row
                rewritten_wcs_rows += 1

    zarr.consolidate_metadata(str(out_zarr))
    post = validate_zarr_store(out_zarr)
    return {
        "store": str(out_zarr),
        "backup": str(backup_path),
        "repaired_len": repaired_len,
        "truncated_arrays": truncated,
        "rewritten_wcs_rows": rewritten_wcs_rows,
        "pre": pre,
        "post": post,
    }


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


def _extract_group_metadata(
    fp: Path,
    *,
    time_key_source: Literal["header", "filename"] = "header",
) -> Tuple[Optional[str], Optional[float], List[str]]:
    """Extract grouping metadata from headers with optional filename time preference.

    Returns
    -------
    Tuple[Optional[str], Optional[float], List[str]]
        Tuple of ``(time_key, frequency_hz, fallback_notes)`` where fallback_notes
        records when filename-based fallback was used.

    When *time_key_source* is ``"filename"``, a parseable basename (legacy averaged pattern
    or ``-image-YYYYMMDD_HHMMSS``) wins over header times so multi-band pipeline products
    group together when ``DATE-OBS`` differs across symlink targets.

    ``frequency_hz`` is read from headers when possible with a filename fallback for
    frequency only.
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
        frequency_hz = _frequency_hz_from_header(header)

    if time_key_source == "filename":
        tk_name = _time_key_from_name(fp)
        if tk_name is not None:
            time_key = tk_name
            notes.append("time-from-filename")
        elif header is not None:
            time_key = _time_key_from_header(header)
    else:
        if header is not None:
            time_key = _time_key_from_header(header)
        if time_key is None:
            tk_name = _time_key_from_name(fp)
            if tk_name is not None:
                time_key = tk_name
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


def _expected_frequencies_from_groups(by_time: Dict[str, List[Path]]) -> List[float]:
    """Build a stable expected frequency axis from discovered input groups.

    Frequencies are binned with the discovery bin width so small header jitter
    does not inflate the expected axis.
    """
    representative_by_bin: Dict[int, float] = {}
    for time_key in sorted(by_time.keys()):
        for fp in sorted(by_time[time_key], key=_frequency_sort_tuple):
            _, frequency_hz, _ = _extract_group_metadata(fp)
            if frequency_hz is None:
                continue
            freq_key = int(round(float(frequency_hz) / _DISCOVERY_FREQ_BIN_HZ))
            representative_by_bin.setdefault(freq_key, float(frequency_hz))
    return [representative_by_bin[k] for k in sorted(representative_by_bin.keys())]


def _reindex_time_step_to_expected_frequencies(
    xds_t: xr.Dataset,
    expected_frequencies_hz: List[float],
) -> xr.Dataset:
    """Ensure each time-step has the full expected frequency axis.

    Missing subbands are introduced as NaN values in data variables.
    """
    if "frequency" not in xds_t.coords or not expected_frequencies_hz:
        return xds_t

    expected = np.asarray(expected_frequencies_hz, dtype=float)
    observed = np.asarray(np.atleast_1d(xds_t["frequency"].values), dtype=float)
    if observed.size == 0:
        return xds_t.reindex({"frequency": expected}, fill_value=np.nan)

    # Snap observed frequencies to the nearest expected bin representative to
    # tolerate small Hz-level jitter while keeping one stable frequency axis.
    mapped = observed.copy()
    max_jitter_hz = _DISCOVERY_FREQ_BIN_HZ / 2.0
    for i, freq in enumerate(observed):
        nearest_idx = int(np.argmin(np.abs(expected - freq)))
        if abs(float(expected[nearest_idx] - freq)) <= max_jitter_hz:
            mapped[i] = expected[nearest_idx]

    xds_norm = xds_t.assign_coords(frequency=("frequency", mapped))

    # If snapping produced duplicate labels, keep first occurrence per frequency.
    _, first_indices = np.unique(mapped, return_index=True)
    if len(first_indices) != len(mapped):
        xds_norm = xds_norm.isel(frequency=np.sort(first_indices))

    xds_norm = xds_norm.sortby("frequency")
    return xds_norm.reindex({"frequency": expected}, fill_value=np.nan)


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

        # xradio expects a STOKES axis in image metadata. Some OVRO-LWA FITS files
        # are 3D cubes with only (RA, DEC, FREQ). Promote these to a 4D cube by
        # adding a singleton STOKES axis so FITS parsing is accepted.
        ctype_values = [
            str(hdr.get(f"CTYPE{i}", "")).strip().upper()
            for i in range(1, int(hdr.get("NAXIS", 0)) + 1)
        ]
        has_stokes = any("STOKES" in c for c in ctype_values)
        if data is not None and int(hdr.get("NAXIS", 0)) == 3 and not has_stokes:
            data = np.expand_dims(data, axis=0)
            hdr["NAXIS"] = 4
            hdr["CTYPE4"] = "STOKES"
            hdr["CRVAL4"] = 1.0
            hdr["CRPIX4"] = 1.0
            hdr["CDELT4"] = 1.0
            if "CUNIT4" not in hdr:
                hdr["CUNIT4"] = ""

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

        # Write via temporary file + atomic replace to avoid leaving a partial
        # output file when the underlying filesystem intermittently short-writes.
        # Retry short-write failures a few times because they can be transient.
        max_attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            tmp_out = path_out.with_name(f"{path_out.name}.tmp.{os.getpid()}.{attempt}")
            try:
                phdu.writeto(tmp_out, overwrite=True)
                os.replace(tmp_out, path_out)
                return
            except OSError as exc:
                last_error = exc
                msg = str(exc)
                short_write = "requested" in msg and "written" in msg
                try:
                    if tmp_out.exists():
                        tmp_out.unlink()
                except OSError:
                    pass
                if short_write and attempt < max_attempts:
                    logger.warning(
                        "Short write while fixing %s (attempt %d/%d): %s; retrying",
                        path_in.name,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    time.sleep(0.2 * attempt)
                    continue
                raise
            except Exception:
                try:
                    if tmp_out.exists():
                        tmp_out.unlink()
                except OSError:
                    pass
                raise

        if last_error is not None:
            raise last_error


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
    cel_hdr = w2d.to_header()
    hdr_str = cel_hdr.tostring(sep="\n")
    cache_key = (ny, nx, hdr_str)
    cached = _sky_coord_cache_get(cache_key)
    if cached is None:
        yy, xx = np.indices((ny, nx), dtype=float)
        ra2d, dec2d = w2d.all_pix2world(xx, yy, 0)  # degrees, pixel centers
        _sky_coord_cache_set(cache_key, (ra2d, dec2d, hdr_str))
    else:
        ra2d, dec2d, hdr_str = cached

    # 4) Attach coords exactly equal to FITS WCS
    xds = xds.assign_coords(
        right_ascension=(("m", "l"), ra2d),
        declination=(("m", "l"), dec2d),
    )
    xds["right_ascension"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})
    xds["declination"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})

    # 5) Persist the exact celestial WCS header so we can re-create WCSAxes later without FITS
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
    """Return LM shape ``(m, l)`` from FITS header without loading pixel data."""
    header = fits.getheader(fp, ext=0)
    naxis = int(header.get("NAXIS", 0))
    if naxis < 2:
        msg = f"FITS file {fp} has NAXIS={naxis}; expected at least 2 for LM dimensions."
        raise RuntimeError(msg)

    return int(header["NAXIS2"]), int(header["NAXIS1"])


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
        # Avoid eagerly fixing every FITS file just to inspect LM shape.
        # Header LM dimensions (NAXIS1/NAXIS2) are sufficient for choosing
        # the global reference grid and dramatically reduce temporary disk use.
        if fix_headers_on_demand:
            shape_paths = sorted(files, key=_frequency_sort_tuple)
        else:
            shape_paths = _get_fixed_paths(files, fixed_dir)
        for fp in shape_paths:
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
    if fix_headers_on_demand and not ref_fp.name.endswith("_fixed.fits"):
        ref_fp = fix_fits_headers([ref_fp], fixed_dir, skip_existing=True)[0]
    return _load_for_combine(ref_fp, chunk_lm=chunk_lm)


def _regrid_to_reference_lm(
    xds: xr.Dataset,
    ref: xr.Dataset,
    *,
    source_label: Optional[str] = None,
) -> xr.Dataset:
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
    source_label
        Optional filename or path for error messages when regridding fails.

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

    if "l" not in xds.coords or "m" not in xds.coords:
        who = f"{source_label}: " if source_label else ""
        msg = f"{who}cannot regrid: dataset is missing ``l`` and/or ``m`` coordinates."
        raise RuntimeError(msg)

    # Materialize for scipy-backed interp; reference coords may be lazy.
    xds = xds.load()
    target_l = ref["l"].load() if hasattr(ref["l"].data, "compute") else ref["l"]
    target_m = ref["m"].load() if hasattr(ref["m"].data, "compute") else ref["m"]

    try:
        regridded = xds.interp(l=target_l, m=target_m, method="linear")
    except Exception as exc:
        who = f"{source_label}: " if source_label else ""
        msg = f"{who}LM regridding onto reference grid failed: {exc}"
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
    *,
    freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ,
    time_key_source: Literal["header", "filename"] = "header",
) -> Dict[str, List[Path]]:
    """Group input FITS by observation time and frequency (headers first, filename fallback).

    Files are associated with a **coarse** frequency key (default 10~kHz bins) so small
    header differences in Hz (RESTFREQ, etc.) do not create extra ``frequency`` planes
    in the Zarr for the same physical subband. For multiple paths in the same
    (time, bin) without a ``duplicate_resolver``, the first file is kept and the rest
    are skipped (with a warning). Distinct subbands remain separate (e.g. 41~MHz vs 55~MHz).

    Parameters
    ----------
    in_dir : Path
        Directory containing input FITS files.
    duplicate_resolver
        Optional callback ``(time_key, frequency_hz, candidates) -> Path`` when multiple
        files share the same time and binned frequency group.
    freq_bin_hz
        Width in Hz for rounding header frequencies to a discovery key,
        ``int(round(frequency_hz / freq_bin_hz))``. Frequencies in the same bin are treated
        as one subband for grouping (up to ~``freq_bin_hz`` separation at bin edges).
    time_key_source
        ``"header"`` (default): prefer header time, then basename patterns when missing.
        ``"filename"``: prefer basename time when parseable (see :func:`_extract_group_metadata`).

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping time keys to lists of FITS file paths.
    """
    if freq_bin_hz <= 0.0:
        msg = f"freq_bin_hz must be positive, got {freq_bin_hz}"
        raise ValueError(msg)

    by_time: Dict[str, List[Path]] = {}
    by_time_freq: Dict[str, Dict[int, List[Path]]] = {}
    for f in sorted(in_dir.glob("*.fits")):
        time_key, frequency_hz, notes = _extract_group_metadata(f, time_key_source=time_key_source)
        if time_key is None:
            t_hint = (
                "-image-YYYYMMDD_HHMMSS in basename, legacy averaged name, or DATE-OBS"
                if time_key_source == "filename"
                else "DATE-OBS or filename time pattern"
            )
            logger.warning(f"Skipping {f.name}: missing usable observation time ({t_hint}).")
            continue
        if frequency_hz is None:
            logger.warning(
                f"Could not determine frequency for {f.name}; duplicate detection disabled for this file."
            )
            by_time.setdefault(time_key, []).append(f)
            continue

        freq_key = int(round(frequency_hz / freq_bin_hz))
        time_freq_map = by_time_freq.setdefault(time_key, {})
        candidates = time_freq_map.setdefault(freq_key, [])
        candidates.append(f)

        if len(candidates) > 1:
            duplicate_names = [p.name for p in candidates]
            if duplicate_resolver is None:
                # Same time + same binned subband: stacking would be wrong; keep the first
                # file and drop the rest (typical: symlink pairs or header jitter in Hz).
                kept = candidates[0]
                logger.warning(
                    "Multiple FITS share time=%s and the same %g Hz frequency bin "
                    "(binned key=%s, ~%.3f MHz): %s. Using only %s. "
                    "Remove extras or pass duplicate_resolver to select a file.",
                    time_key,
                    freq_bin_hz,
                    freq_key,
                    frequency_hz / 1e6,
                    duplicate_names,
                    kept.name,
                )
                time_freq_map[freq_key] = [kept]
                continue

            _, rep_hz, _ = _extract_group_metadata(candidates[0], time_key_source=time_key_source)
            resolver_hz = float(rep_hz) if rep_hz is not None else float(freq_key) * freq_bin_hz
            selected = duplicate_resolver(time_key, resolver_hz, candidates.copy())
            if selected not in candidates:
                msg = (
                    f"Duplicate resolver returned unknown file {selected} for "
                    f"time={time_key}, frequency_hz={resolver_hz}."
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
                resolver_hz,
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
) -> Tuple[xr.Dataset, List[float], List[Path]]:
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
    Tuple[xr.Dataset, List[float], List[Path]]
        Tuple of (combined dataset, sorted list of unique frequencies in Hz,
        newly-created ``*_fixed.fits`` paths for optional cleanup).
    """
    created_fixed_paths: List[Path] = []
    if fix_headers_on_demand:
        existed_before: Dict[Path, bool] = {}
        for f in sorted(files, key=_frequency_sort_tuple):
            if f.name.endswith("_fixed.fits"):
                continue
            candidate = fixed_dir / (f.stem + "_fixed.fits")
            existed_before[candidate] = candidate.exists()
        # Fix headers if needed (skips existing fixed files)
        fixed_paths = fix_fits_headers(files, fixed_dir, skip_existing=True)
        for f in sorted(files, key=_frequency_sort_tuple):
            if f.name.endswith("_fixed.fits"):
                continue
            candidate = fixed_dir / (f.stem + "_fixed.fits")
            if not existed_before.get(candidate, False) and candidate.exists():
                created_fixed_paths.append(candidate)
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
        xds_list[i] = _regrid_to_reference_lm(xds, ref_ds, source_label=str(fixed_paths[i]))

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

    xds_t = _rechunk_lm_for_zarr(xds_t, chunk_lm)

    return xds_t, sorted(set(freqs_seen)), created_fixed_paths


def _rechunk_nonuniform_aux_vars_for_zarr(xds: xr.Dataset) -> xr.Dataset:
    """Rechunk metadata-sized data vars so Dask chunks satisfy Zarr's uniformity rule.

    ``xr.concat`` / ``combine_by_coords`` can leave variables that only span
    ``frequency`` (or similar) with *irregular* Dask chunks (e.g. ``(2, 1, 1)``
    along one dimension). ``xarray``'s Zarr writer requires all non-final
    chunks to share the same size along each dimension; otherwise it raises
    (``wcs_header_str`` is a known case).

    Parameters
    ----------
    xds
        Dataset possibly containing small dask-backed aux variables.

    Returns
    -------
    xarray.Dataset
        Dataset with offending variables rechunks to one chunk per dimension.
    """
    out = xds
    for name in list(out.data_vars):
        v = out[name]
        da_ = v.data
        if not hasattr(da_, "chunks") or not da_.chunks:
            continue
        bad = False
        for dim_chunks in da_.chunks:
            if len(dim_chunks) > 1 and len(set(dim_chunks[:-1])) > 1:
                bad = True
                break
        if bad:
            chunk_arg = {d: -1 for d in v.dims}
            out = out.assign(**{name: v.chunk(chunk_arg)})
    return out


def _rechunk_nonuniform_coords_for_zarr(xds: xr.Dataset) -> xr.Dataset:
    """Rechunk coordinate arrays that have non-uniform Dask chunks.

    Coordinates like ``right_ascension``/``declination`` can gain a leading
    ``time`` dimension during append. If that axis chunks as ``(2, 1, 1)``,
    xarray's Zarr writer rejects it as non-uniform. Rechunk only the offending
    dimensions (e.g. ``time``) and keep already-uniform spatial chunks unchanged.
    """
    out = xds
    for name in list(out.coords):
        c = out[name]
        da_ = c.data
        if not hasattr(da_, "chunks") or not da_.chunks:
            continue
        bad_dims: list[str] = []
        for dim_name, dim_chunks in zip(c.dims, da_.chunks, strict=True):
            if len(dim_chunks) > 1 and len(set(dim_chunks[:-1])) > 1:
                bad_dims.append(dim_name)
        if bad_dims:
            chunk_arg = {d: -1 for d in bad_dims}
            out = out.assign_coords({name: c.chunk(chunk_arg)})
    return out


def _strip_encodings_for_zarr_write(xds: xr.Dataset) -> xr.Dataset:
    """Clear encodings on coordinates and data variables before Zarr write.

    After ``Dataset.chunk`` and ``concat``, xarray may attach ``encoding['chunks']``
    to coordinates (e.g. ``right_ascension``, ``declination``). If those encodings
    do not align with the current Dask chunk boundaries, ``to_zarr`` raises
    *Specified Zarr chunks … would overlap multiple Dask chunks*. Stripping
    encodings matches the pattern already used for data variables after
    :func:`_combine_time_step` and lets the writer derive chunks from the arrays.
    """
    for name in xds.coords:
        xds[name].encoding = {}
    for name in xds.data_vars:
        xds[name].encoding = {}
    return xds


def _rechunk_lm_for_zarr(xds: xr.Dataset, chunk_lm: int) -> xr.Dataset:
    """Rechunk ``l`` and ``m`` so Dask-backed arrays use uniform spatial chunk sizes.

    ``combine_by_coords`` / ``concat`` can fuse slices into irregular chunk
    boundaries along ``l``/``m``. Zarr encoding (via xradio) requires uniform
    chunk sizes per dimension except possibly the final chunk.

    Parameters
    ----------
    xds
        Dataset whose spatial dimensions are named ``l`` and ``m``.
    chunk_lm
        Target chunk length for each of ``l`` and ``m``. If zero, each spatial
        axis is stored as a single chunk (still uniform).
    """
    if {"l", "m"} <= set(xds.dims):
        if chunk_lm and chunk_lm > 0:
            xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})
        else:
            xds = xds.chunk({"l": -1, "m": -1})
    xds = _rechunk_nonuniform_aux_vars_for_zarr(xds)
    xds = _rechunk_nonuniform_coords_for_zarr(xds)
    xds = _strip_encodings_for_zarr_write(xds)
    return xds


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
        raise RuntimeError(
            "l/m grids differ across times after normalization; aborting to avoid misalignment."
        )


def _zarr_array_dims_and_lengths(
    var_name: str,
    *,
    zgroup: zarr.Group | None = None,
    store_path: Path | None = None,
) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    """Return dimension order and axis lengths for one Zarr array using xarray conventions.

    Reads ``_ARRAY_DIMENSIONS`` from array metadata (xarray/zarr convention).
    """
    zg = zgroup if zgroup is not None else zarr.open_group(str(store_path), mode="r")
    node = zg[var_name]
    if not hasattr(node, "shape"):
        msg = f"Zarr node {var_name!r} in {store_path} is not an array; cannot read schema."
        raise RuntimeError(msg)
    arr = node
    dims_attr = arr.attrs.get("_ARRAY_DIMENSIONS")
    if dims_attr is None:
        msg = (
            f"Zarr array {var_name!r} has no _ARRAY_DIMENSIONS metadata; "
            "cannot align append schema without opening the store as xarray."
        )
        raise RuntimeError(msg)
    if isinstance(dims_attr, str):
        dims = (dims_attr,)
    else:
        dims = tuple(str(d) for d in dims_attr)
    shape = tuple(int(s) for s in arr.shape)
    if len(dims) != len(shape):
        msg = f"Zarr array {var_name!r} dims {dims} do not match shape {shape}."
        raise RuntimeError(msg)
    return dims, dict(zip(dims, shape, strict=True))


def _align_incoming_with_zarr_schema(
    incoming: xr.Dataset,
    existing_path: Path,
) -> xr.Dataset:
    """Broadcast shared variables to match on-disk Zarr dimension order (no full Dataset open).

    Used when ``xr.open_zarr`` fails due to inconsistent lengths across arrays in the store.
    """
    aligned = incoming
    zg = zarr.open_group(str(existing_path), mode="r")
    existing_names = {k for k in zg.array_keys()}
    incoming_names = set(aligned.variables.keys())
    shared = sorted(existing_names & incoming_names)
    for name in shared:
        e_dims, e_lengths = _zarr_array_dims_and_lengths(name, zgroup=zg)
        i_dims = tuple(aligned[name].dims)
        if e_dims == i_dims:
            continue
        if set(i_dims).issubset(set(e_dims)):
            missing_dims = [d for d in e_dims if d not in i_dims]
            expand_kwargs = {}
            for d in missing_dims:
                if d in aligned.sizes:
                    expand_kwargs[d] = int(aligned.sizes[d])
                elif d in e_lengths:
                    expand_kwargs[d] = int(e_lengths[d])
                else:
                    msg = (
                        f"Cannot infer size for missing dimension {d!r} when aligning {name!r} "
                        f"to on-disk dims {e_dims}."
                    )
                    raise RuntimeError(msg)
            expanded = aligned[name].expand_dims(expand_kwargs)
            if name in aligned.coords:
                aligned = aligned.assign_coords({name: expanded})
            else:
                aligned = aligned.assign(**{name: expanded})
            aligned[name] = aligned[name].transpose(*e_dims)
    return aligned


def _write_or_append_zarr(
    xds_t: xr.Dataset,
    out_zarr: Path,
    first_write: bool,
    *,
    chunk_lm: int,
) -> None:
    """Write first time-step, then append incrementally on ``time``.

    Parameters
    ----------
    xds_t : xr.Dataset
        Dataset to write or append.
    out_zarr : Path
        Path to output Zarr store.
    first_write : bool
        If True, write a new Zarr store; if False, append this dataset
        along the ``time`` dimension.
    chunk_lm
        Spatial chunk size passed to :func:`_rechunk_lm_for_zarr` so appended
        stores match uniform Zarr chunking requirements.
    """
    if first_write or (not out_zarr.exists()):
        to_write = _rechunk_lm_for_zarr(xds_t, chunk_lm)
        write_image(
            to_write,
            str(out_zarr),
            out_format="zarr",
            overwrite=True,
        )
        zarr.consolidate_metadata(str(out_zarr))
        return

    def _align_with_existing_schema(incoming: xr.Dataset, existing_path: Path) -> xr.Dataset:
        """Broadcast append variables to match existing Zarr variable dimensions.

        Some existing stores contain metadata variables (e.g., velocity, wcs_header_str)
        with dimensions including ``time`` due to historical concat/write behavior.
        Incremental append must preserve variable dimension names exactly.
        """
        try:
            existing_schema = xr.open_zarr(str(existing_path), consolidated=False)
        except Exception as exc:
            return _align_incoming_with_zarr_schema(incoming, existing_path)
        try:
            aligned = incoming
            existing_var_names = set(existing_schema.variables.keys())
            incoming_var_names = set(aligned.variables.keys())
            shared = sorted(existing_var_names & incoming_var_names)
            for name in shared:
                e_dims = tuple(existing_schema[name].dims)
                i_dims = tuple(aligned[name].dims)
                if e_dims == i_dims:
                    continue
                # If existing dims are a superset of incoming dims, broadcast missing dims.
                if set(i_dims).issubset(set(e_dims)):
                    missing_dims = [d for d in e_dims if d not in i_dims]
                    expand_kwargs = {
                        d: int(aligned.sizes[d]) if d in aligned.sizes else int(existing_schema.sizes[d])
                        for d in missing_dims
                    }
                    expanded = aligned[name].expand_dims(expand_kwargs)
                    if name in aligned.coords:
                        aligned = aligned.assign_coords({name: expanded})
                    else:
                        aligned = aligned.assign(**{name: expanded})
                    aligned[name] = aligned[name].transpose(*e_dims)
            return aligned
        finally:
            existing_schema.close()

    to_append = _rechunk_lm_for_zarr(xds_t, chunk_lm)
    to_append = _align_with_existing_schema(to_append, out_zarr)
    if "frequency" in to_append.coords:
        to_append = to_append.sortby("frequency")
    if "time" in to_append.coords:
        to_append = to_append.sortby("time")
    if "wcs_header_str" in to_append.data_vars:
        # Keep metadata variable chunking compatible with existing Zarr chunk shape.
        # The target store typically has frequency as a single chunk (e.g. 15), and
        # split chunks like (14, 1) can overlap that target chunk during append.
        chunk_args: Dict[str, int] = {}
        if "time" in to_append["wcs_header_str"].dims:
            chunk_args["time"] = 1
        if "frequency" in to_append["wcs_header_str"].dims:
            chunk_args["frequency"] = -1
        if chunk_args:
            to_append = to_append.assign(
                wcs_header_str=to_append["wcs_header_str"].chunk(chunk_args)
            )

    # True incremental append: write only new time samples.
    to_append.to_zarr(str(out_zarr), mode="a", append_dim="time")
    zarr.consolidate_metadata(str(out_zarr))



def convert_fits_dir_to_zarr(
    input_dir: str | Path,
    out_dir: str | Path,
    zarr_name: str = "ovro_lwa_full_lm_only.zarr",
    fixed_dir: str | Path = "fixed_fits",
    chunk_lm: int = 1024,
    rebuild: bool = False,
    resume: bool = False,
    fix_headers_on_demand: bool = True,
    cleanup_fixed_fits: bool = False,
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
    resume
        If True and an existing output Zarr is present, skip input time steps
        that already exist in the Zarr ``time`` coordinate.
    fix_headers_on_demand
        If True, fix FITS headers on-demand during conversion if they don't exist.
        If False, assume headers are already fixed using :func:`fix_fits_headers`.
        Default is True.
    cleanup_fixed_fits
        If True (and ``fix_headers_on_demand`` is enabled), delete temporary
        ``*_fixed.fits`` files created during each time-step after writing that
        step to Zarr. Use this to reduce peak disk usage.
    progress_callback
        Optional callback function for progress reporting. Should accept
        (stage: str, current: int, total: int, message: str).
    duplicate_resolver
        Optional callback to resolve duplicate files that map to the same
        time/frequency group. Signature: ``(time_key, frequency_hz, candidates) -> selected_path``.

    Mixed-resolution inputs (different ``l``/``m`` pixel shapes) are supported: the
    largest LM grid among all selected files becomes the conversion-wide reference,
    and smaller images are linearly interpolated onto that grid before combine. The
    same reference is used for every time step so output Zarr has one consistent
    sky pixel grid.

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

    all_time_keys = sorted(by_time.keys())
    expected_frequencies_hz = _expected_frequencies_from_groups(by_time)
    pending_time_keys = all_time_keys
    if resume and out_zarr.exists() and not rebuild:
        committed_time_keys = _committed_time_keys_from_txn(out_zarr)
        if committed_time_keys:
            existing_time_keys = committed_time_keys
            logger.info(
                "Resume mode: using %d committed time marker(s) from %s",
                len(existing_time_keys),
                _txn_dir_for_store(out_zarr),
            )
        else:
            existing_time_keys = _existing_time_keys_from_zarr(out_zarr)
        pending_time_keys = [k for k in all_time_keys if k not in existing_time_keys]
        skipped = len(all_time_keys) - len(pending_time_keys)
        logger.info(
            "Resume mode: %d/%d time step(s) already present in output Zarr; %d pending.",
            skipped,
            len(all_time_keys),
            len(pending_time_keys),
        )

    if not pending_time_keys:
        logger.info("No pending time steps to process; output already up to date: %s", out_zarr)
        return out_zarr

    total_time_steps = len(pending_time_keys)
    for idx, tkey in enumerate(pending_time_keys):
        files = by_time[tkey]

        logger.info(f"[read/combine] time {tkey}")
        xds_t, freqs, created_fixed_paths = _combine_time_step(
            files,
            fixed_dir,
            chunk_lm=chunk_lm,
            fix_headers_on_demand=fix_headers_on_demand,
            lm_reference_ds=lm_ref_ds,
        )
        xds_t = _reindex_time_step_to_expected_frequencies(xds_t, expected_frequencies_hz)
        logger.info(f"  combined dims: {dict(xds_t.sizes)}")
        logger.info(f"  combined freqs (Hz): {freqs[:8]}{' ...' if len(freqs) > 8 else ''}")

        lm_current = (xds_t["l"].values, xds_t["m"].values)
        _assert_same_lm(lm_reference, lm_current)
        logger.info("  l/m grid matches global reference")

        logger.info(f"[{'write new' if first_write else 'append'}] {out_zarr}")
        if out_zarr.exists() and not first_write:
            _validate_time_axis_consistency_zarr(out_zarr)
        _mark_time_in_progress(out_zarr, tkey)
        try:
            _write_or_append_zarr(xds_t, out_zarr, first_write=first_write, chunk_lm=chunk_lm)
            _mark_time_committed(out_zarr, tkey)
            first_write = False
        except KeyboardInterrupt:
            if out_zarr.exists():
                try:
                    zarr.consolidate_metadata(str(out_zarr))
                except Exception:
                    pass
            raise
        if cleanup_fixed_fits and fix_headers_on_demand and created_fixed_paths:
            removed = 0
            for fixed_path in created_fixed_paths:
                try:
                    fixed_path.unlink(missing_ok=True)
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not remove temporary fixed FITS %s: %s", fixed_path, exc)
            logger.info("Cleaned up %d temporary fixed FITS file(s) for time %s", removed, tkey)

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
