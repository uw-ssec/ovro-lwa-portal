"""Tests for dewarp → staging helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy.io import fits

<<<<<<< HEAD
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


||||||| c766338
=======
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
>>>>>>> 32d638a8d16398de7ed583b22349b04b1d6a5048
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
