"""Tests for dewarp → staging helpers."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits

from tests.conftest import skip_github_ci_without_image_plane_correction

from ovro_lwa_portal.ingest.dewarp_convert import (
    collect_cascade_fits,
    run_cascade_per_time_group,
)


def _write_minimal_fits(path: Path, *, restfreq_hz: float, date_obs: str) -> None:
    data = np.zeros((4, 4), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["DATE-OBS"] = date_obs
    hdu.header["RESTFREQ"] = restfreq_hz
    hdu.writeto(path, overwrite=True)


def test_collect_cascade_fits_prefers_flat(tmp_path: Path) -> None:
    """Non-recursive *.fits in outroot are returned first."""
    (tmp_path / "a.fits").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.fits").touch()
    found = collect_cascade_fits(tmp_path)
    assert [p.name for p in found] == ["a.fits"]


def test_collect_cascade_fits_recursive(tmp_path: Path) -> None:
    """When no top-level fits, rglob finds nested files."""
    sub = tmp_path / "nested"
    sub.mkdir(parents=True)
    (sub / "c.fits").touch()
    found = collect_cascade_fits(tmp_path)
    assert len(found) == 1
    assert found[0].name == "c.fits"


@skip_github_ci_without_image_plane_correction
@pytest.mark.dewarp
def test_run_cascade_per_time_group_fake_cascade(tmp_path: Path) -> None:
    """Staging receives one link per cascade output per time group."""

    def fake_cascade(
        image_filenames: list[str],
        outroot: str,
        write: bool = True,
        **_kwargs: Any,
    ) -> None:
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(image_filenames):
            shutil.copy2(fp, out / f"dewarped_{i}.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    cascade_parent = tmp_path / "cascade"
    staging = tmp_path / "staging"
    _write_minimal_fits(
        raw / "a.fits", restfreq_hz=70e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "b.fits", restfreq_hz=74e6, date_obs="2024-06-01T12:00:00.0"
    )

    n, keys = run_cascade_per_time_group(
        raw,
        cascade_parent,
        staging,
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )
    assert len(keys) == 1
    assert n == 2
    staged = sorted(staging.glob("*.fits"))
    assert len(staged) == 2
    assert all("__dewarped_" in p.name for p in staged)


@skip_github_ci_without_image_plane_correction
@pytest.mark.dewarp
def test_run_cascade_per_time_group_raises_if_no_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "only.fits", restfreq_hz=70e6, date_obs="2024-06-02T12:00:00.0"
    )

    def noop_cascade(**_kwargs: Any) -> None:
        return None

    with pytest.raises(RuntimeError, match="produced no"):
        run_cascade_per_time_group(
            raw,
            tmp_path / "cascade",
            tmp_path / "staging",
            discovery_freq_bin_hz=23e3,
            cascade_fn=noop_cascade,
        )


def test_run_cascade_per_time_group_passes_absolute_paths_and_outroot(tmp_path: Path) -> None:
    """Cascade receives absolute ``image_filenames`` and ``outroot`` under cascade_parent/time_key."""
    recorded: list[dict[str, Any]] = []

    def recording_cascade(
        *,
        image_filenames: Sequence[str],
        outroot: str,
        cleaned: bool,
        qa: bool,
        use_best_pb_model: bool,
        bright_source_flux_qa: bool,
        write: bool,
        target_size: int | None = None,
    ) -> None:
        recorded.append(
            {
                "image_filenames": list(image_filenames),
                "outroot": outroot,
                "cleaned": cleaned,
                "qa": qa,
                "use_best_pb_model": use_best_pb_model,
                "bright_source_flux_qa": bright_source_flux_qa,
                "write": write,
                "target_size": target_size,
            }
        )
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(image_filenames):
            shutil.copy2(fp, out / f"out_{i}.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    cascade_parent = tmp_path / "cascade"
    staging = tmp_path / "staging"
    _write_minimal_fits(
        raw / "low.fits", restfreq_hz=70e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "high.fits", restfreq_hz=74e6, date_obs="2024-06-01T12:00:00.0"
    )

    n, keys = run_cascade_per_time_group(
        raw,
        cascade_parent,
        staging,
        discovery_freq_bin_hz=23e3,
        cascade_fn=recording_cascade,
        cleaned=False,
        qa=False,
        use_best_pb_model=False,
        bright_source_flux_qa=False,
        write=False,
    )

    assert keys == ["20240601_120000"]
    assert n == 2
    assert len(recorded) == 1
    call = recorded[0]
    assert call["cleaned"] is False
    assert call["qa"] is False
    assert call["use_best_pb_model"] is False
    assert call["bright_source_flux_qa"] is False
    assert call["write"] is False
    assert call["target_size"] is None

    expected_outroot = cascade_parent / "20240601_120000"
    assert Path(call["outroot"]) == expected_outroot
    assert Path(call["outroot"]).is_dir()

    # Frequency order: 70 MHz before 74 MHz (same as _discover_groups sort).
    paths = [Path(p) for p in call["image_filenames"]]
    assert all(p.is_absolute() for p in paths)
    assert [p.name for p in paths] == ["low.fits", "high.fits"]
    assert paths[0].resolve() == (raw / "low.fits").resolve()
    assert paths[1].resolve() == (raw / "high.fits").resolve()

    staged = sorted(staging.glob("*.fits"))
    assert [p.name for p in staged] == [
        "20240601_120000__out_0.fits",
        "20240601_120000__out_1.fits",
    ]


def test_run_cascade_per_time_group_multiple_time_keys(tmp_path: Path) -> None:
    """Each observation time gets its own outroot and staging name prefix."""
    calls: list[str] = []

    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        calls.append(outroot)
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        tkey = Path(outroot).name
        shutil.copy2(image_filenames[0], out / f"{tkey}_band.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "day1.fits", restfreq_hz=70e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "day2.fits", restfreq_hz=70e6, date_obs="2024-06-02T12:00:00.0"
    )

    n, keys = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )

    assert keys == ["20240601_120000", "20240602_120000"]
    assert n == 2
    assert calls == [
        str(tmp_path / "cascade" / "20240601_120000"),
        str(tmp_path / "cascade" / "20240602_120000"),
    ]
    staged = sorted((tmp_path / "staging").glob("*.fits"))
    assert {p.name for p in staged} == {
        "20240601_120000__20240601_120000_band.fits",
        "20240602_120000__20240602_120000_band.fits",
    }


def test_run_cascade_per_time_group_clears_staging_and_per_time_outroot(tmp_path: Path) -> None:
    """Existing staging tree and per-time cascade dirs are removed before a run."""
    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_filenames[0], out / "new.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "one.fits", restfreq_hz=70e6, date_obs="2024-06-03T12:00:00.0"
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "stale.fits").write_text("old")
    sub = staging / "nested"
    sub.mkdir()
    (sub / "more.fits").touch()

    tkey = "20240603_120000"
    outroot = tmp_path / "cascade" / tkey
    outroot.mkdir(parents=True)
    (outroot / "leftover.txt").write_text("stale")

    run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        staging,
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )

    assert not (staging / "stale.fits").exists()
    assert not (staging / "nested").exists()
    assert not (outroot / "leftover.txt").exists()
    assert (staging / f"{tkey}__new.fits").is_file()


def test_run_cascade_groups_by_basename_image_time_not_header(tmp_path: Path) -> None:
    """Dewarp discovery co-locates bands that share ``-image-…`` even when DATE-OBS disagrees."""
    calls: list[tuple[str, list[str]]] = []

    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        calls.append((Path(outroot).name, sorted(Path(p).name for p in image_filenames)))
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(image_filenames):
            shutil.copy2(fp, out / f"band_{i}.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits",
        restfreq_hz=18e6,
        date_obs="2024-12-21T01:00:00.0",
    )
    _write_minimal_fits(
        raw / "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits",
        restfreq_hz=73e6,
        date_obs="2024-12-22T23:00:00.0",
    )

    n, keys = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )

    assert keys == ["20241221_102109"]
    assert n == 2
    assert len(calls) == 1
    assert calls[0][0] == "20241221_102109"
    assert set(calls[0][1]) == {
        "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits",
        "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits",
    }


def test_run_cascade_per_time_group_forwards_target_size(tmp_path: Path) -> None:
    recorded: list[int | None] = []

    def cascade_kw(
        *,
        image_filenames: Sequence[str],
        outroot: str,
        target_size: int | None = None,
        **_kw: Any,
    ) -> None:
        recorded.append(target_size)
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_filenames[0], out / "one.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "one.fits", restfreq_hz=70e6, date_obs="2024-06-01T12:00:00.0"
    )

    n, keys = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=cascade_kw,
        target_size=3122,
    )
    assert n == 1
    assert keys == ["20240601_120000"]
    assert recorded == [3122]
