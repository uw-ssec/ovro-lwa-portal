"""Dewarp (``image_plane_correction.flow.flow_cascade73MHz``) staging before FITS→Zarr."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ovro_lwa_portal.fits_to_zarr_xradio import _discover_groups

__all__ = [
    "collect_cascade_fits",
    "import_flow_cascade73mhz",
    "run_cascade_per_time_group",
]

logger = logging.getLogger(__name__)


def import_flow_cascade73mhz() -> Callable[..., Any]:
    """Import and return ``image_plane_correction.flow.flow_cascade73MHz``.

    Raises
    ------
    ImportError
        If ``image_plane_correction`` is not installed, or ``flow`` has no
        ``flow_cascade73MHz``.
    """
    try:
        from image_plane_correction import flow as flow_mod
    except ImportError as e:
        msg = (
            "Could not import `image_plane_correction.flow` (install the "
            "`image-plane-correction` / `image_plane_correction` package in the "
            "same Python environment you use to run `ovro-ingest`)."
        )
        raise ImportError(msg) from e
    fn = getattr(flow_mod, "flow_cascade73MHz", None)
    if fn is None:
        msg = (
            "`image_plane_correction.flow` has no attribute `flow_cascade73MHz`. "
            "Check your image_plane_correction version."
        )
        raise ImportError(msg)
    return fn


def collect_cascade_fits(outroot: Path) -> list[Path]:
    """Return sorted ``*.fits`` paths under *outroot* (non-recursive, then recursive)."""
    direct = sorted(outroot.glob("*.fits"))
    if direct:
        return direct
    return sorted(outroot.rglob("*.fits"))


def _link_or_copy(src: Path, dest: Path) -> None:
    """Symlink *src* to *dest* when possible; otherwise copy."""
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def run_cascade_per_time_group(
    input_dir: Path,
    cascade_parent: Path,
    staging_dir: Path,
    *,
    discovery_freq_bin_hz: float,
    duplicate_resolver: Callable[[str, float, list[Path]], Path] | None = None,
    cascade_fn: Callable[..., Any] | None = None,
    cleaned: bool = True,
    qa: bool = True,
    use_best_pb_model: bool = True,
    bright_source_flux_qa: bool = True,
    write: bool = True,
    target_size: int | None = None,
) -> tuple[int, list[str]]:
    """Run ``flow_cascade73MHz`` once per observation-time group and stage outputs.

    FITS under *input_dir* are grouped like :func:`convert_fits_dir_to_zarr`, except the
    time key prefers the basename ``-image-YYYYMMDD_HHMMSS`` stamp when present (so
    multi-band images that share a pipeline image id stay in one group even if
    ``DATE-OBS`` differs between symlink targets). If that pattern is missing, grouping
    falls back to ``DATE-OBS`` like the Zarr ingest path.

    For each time key, all subband files in that group are passed as ``image_filenames`` to
    ``image_plane_correction.flow.flow_cascade73MHz`` with ``outroot=cascade_parent / time_key``.

    Produced ``*.fits`` (immediate children of *outroot*, else recursive) are linked
    or copied into *staging_dir* as ``{time_key}__{basename}`` so the flat directory
    can be consumed by :meth:`ovro_lwa_portal.ingest.core.FITSToZarrConverter.convert`.

    Parameters
    ----------
    input_dir, cascade_parent, staging_dir
        Input FITS directory, per-time cascade output parent, and flat staging dir.
    discovery_freq_bin_hz, duplicate_resolver
        Passed to :func:`ovro_lwa_portal.fits_to_zarr_xradio._discover_groups` (with
        ``time_key_source="filename"`` so basename image times drive groups).
    cascade_fn
        Callable with the same keyword interface as
        ``image_plane_correction.flow.flow_cascade73MHz``.
        Defaults to importing that function.
    cleaned, qa, use_best_pb_model, bright_source_flux_qa, write, target_size
        Forwarded to *cascade_fn* together with ``image_filenames`` and ``outroot``.
        *target_size* matches ``image_plane_correction`` dewarp entry points
        (e.g. ``calcflow``, ``flow_cascade73MHz``): side length in pixels for the
        output raster, or ``None`` for the library default.

    Returns
    -------
    n_staged : int
        Number of FITS linked or copied into *staging_dir*.
    time_keys : list of str
        Sorted time keys that were processed.

    Raises
    ------
    FileNotFoundError
        If discovery finds no groupable FITS.
    RuntimeError
        If a cascade step writes no ``*.fits`` under the per-time *outroot*.
    """
    cascade_fn = cascade_fn or import_flow_cascade73mhz()
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    cascade_parent.mkdir(parents=True, exist_ok=True)

    by_time = _discover_groups(
        input_dir,
        duplicate_resolver=duplicate_resolver,
        freq_bin_hz=discovery_freq_bin_hz,
        time_key_source="filename",
    )
    if not by_time:
        msg = f"No groupable FITS files found in {input_dir}"
        raise FileNotFoundError(msg)

    time_keys_sorted = sorted(by_time.keys())
    n_staged = 0
    for tkey in time_keys_sorted:
        files = list(by_time[tkey])
        outroot = cascade_parent / tkey
        if outroot.exists():
            shutil.rmtree(outroot)
        outroot.mkdir(parents=True, exist_ok=True)
        fns: Sequence[str] = [str(p.resolve()) for p in files]
        logger.info(
            "Running flow_cascade73MHz for time_key=%s (%d files) → %s",
            tkey,
            len(fns),
            outroot,
        )
        cascade_fn(
            image_filenames=fns,
            cleaned=cleaned,
            qa=qa,
            use_best_pb_model=use_best_pb_model,
            bright_source_flux_qa=bright_source_flux_qa,
            write=write,
            outroot=str(outroot),
            target_size=target_size,
        )
        produced = collect_cascade_fits(outroot)
        if not produced:
            msg = (
                f"flow_cascade73MHz produced no *.fits under {outroot}. "
                "Check the flow package output layout or enable write=True."
            )
            raise RuntimeError(msg)
        for p in produced:
            dest = staging_dir / f"{tkey}__{p.name}"
            if dest.exists():
                dest.unlink()
            _link_or_copy(p, dest)
            n_staged += 1
    return n_staged, time_keys_sorted
