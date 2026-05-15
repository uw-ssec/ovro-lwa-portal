"""Tests for subband metadata audit."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from ovro_lwa_portal.ingest.cli import app
from ovro_lwa_portal.ingest.metadata_audit import audit_time_group_files

runner = CliRunner()
_CI_PLAIN_ENV = {"NO_COLOR": "1", "FORCE_COLOR": "0"}


def _write_subband(
    path: Path,
    *,
    date_obs: str,
    mhz: int,
    image_time: str,
    naxis: int = 64,
) -> None:
    name = f"{mhz}MHz-I-Deep-Taper-Robust-0-image-{image_time}_x.fits"
    fpath = path / name
    data = np.ones((naxis, naxis), dtype=np.float32)
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis
    hdr["NAXIS2"] = naxis
    hdr["DATE-OBS"] = date_obs
    hdr["RESTFREQ"] = float(mhz) * 1e6
    hdr["CRVAL3"] = float(mhz) * 1e6 + 1.5e6
    hdr["BMAJ"] = 0.1 + mhz * 1e-4
    hdr["BMIN"] = 0.08 + mhz * 1e-4
    fits.PrimaryHDU(data=data, header=hdr).writeto(fpath, overwrite=True)


def test_audit_consistent_subbands_no_issues(tmp_path: Path) -> None:
    """Matching DATE-OBS across subbands should not raise subband consistency issues."""
    image_time = "20250106_051855"
    date_obs = "2025-01-06T04:49:04.5"
    for mhz in (41, 55, 73):
        _write_subband(tmp_path, date_obs=date_obs, mhz=mhz, image_time=image_time)

    paths = sorted(tmp_path.glob("*.fits"))
    report = audit_time_group_files(paths, image_time, label="test")

    assert report.n_files == 3
    assert not report.has_issues
    assert report.time_keys is not None
    assert report.time_keys.header_mjd_spread_s == 0.0
    assert "DATE-OBS" in {fs.name for fs in report.field_summaries}


def test_audit_inconsistent_date_obs_reports_issue(tmp_path: Path) -> None:
    """Different DATE-OBS across subbands must be flagged."""
    image_time = "20250106_051855"
    _write_subband(tmp_path, date_obs="2025-01-06T04:49:04.5", mhz=41, image_time=image_time)
    _write_subband(tmp_path, date_obs="2025-01-06T05:00:00.0", mhz=73, image_time=image_time)

    report = audit_time_group_files(sorted(tmp_path.glob("*.fits")), image_time)

    assert report.has_issues
    assert any("DATE-OBS" in issue for issue in report.issues)


def test_audit_filename_header_mismatch_warns(tmp_path: Path) -> None:
    """Filename -image- time and DATE-OBS time should produce a warning when they differ."""
    image_time = "20250106_051855"
    _write_subband(tmp_path, date_obs="2025-01-06T04:49:04.5", mhz=73, image_time=image_time)

    report = audit_time_group_files(sorted(tmp_path.glob("*.fits")), image_time)

    assert report.time_keys is not None
    assert not report.time_keys.filename_header_agree
    assert any("filename time key" in w for w in report.warnings)


def test_cli_audit_metadata_lists_time_keys(tmp_path: Path) -> None:
    """Default invocation lists discovered groups without --time-key."""
    image_time = "20250106_051855"
    _write_subband(tmp_path, date_obs="2025-01-06T04:49:04.5", mhz=73, image_time=image_time)

    result = runner.invoke(
        app,
        ["audit-metadata", str(tmp_path)],
        color=False,
        env=_CI_PLAIN_ENV,
    )
    assert result.exit_code == 0
    plain = click.unstyle(result.stdout)
    assert image_time in plain
    assert "time-key" in plain


def test_cli_audit_metadata_single_group(tmp_path: Path) -> None:
    """Auditing one time group should succeed without --strict issues."""
    image_time = "20250106_051855"
    date_obs = "2025-01-06T04:49:04.5"
    for mhz in (41, 73):
        _write_subband(tmp_path, date_obs=date_obs, mhz=mhz, image_time=image_time)

    result = runner.invoke(
        app,
        ["audit-metadata", str(tmp_path), "--time-key", image_time, "--strict"],
    )
    assert result.exit_code == 0
    assert "No subband metadata issues" in result.stdout


def test_cli_audit_metadata_help() -> None:
    """audit-metadata --help documents staging and combine probe flags."""
    result = runner.invoke(
        app,
        ["audit-metadata", "--help"],
        color=False,
        terminal_width=120,
        env=_CI_PLAIN_ENV,
    )
    assert result.exit_code == 0
    plain = click.unstyle(result.stdout)
    assert "probe-combine" in plain
    assert "staging-dir" in plain
