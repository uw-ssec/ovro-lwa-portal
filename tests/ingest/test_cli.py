"""Tests for the CLI module."""

from __future__ import annotations

import click
import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from ovro_lwa_portal.ingest.cli import app


runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self) -> None:
        """Test CLI help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Convert OVRO-LWA FITS files to Zarr format" in result.stdout

    def test_version(self) -> None:
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "ovro-ingest version" in result.stdout

    def test_version_flag(self) -> None:
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "ovro-ingest version" in result.stdout

    def test_convert_help(self) -> None:
        """Test convert command help."""
        result = runner.invoke(app, ["convert", "--help"])
        plain_output = click.unstyle(result.stdout)
        assert result.exit_code == 0
        assert "Convert OVRO-LWA FITS files to a single Zarr store" in plain_output
        assert "largest grid" in plain_output
        assert "--resume" in plain_output

    def test_validate_help(self) -> None:
        """Test validate command help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate time-axis consistency" in result.stdout

    def test_repair_help(self) -> None:
        """Test repair command help."""
        result = runner.invoke(app, ["repair", "--help"])
        assert result.exit_code == 0
        assert "Repair interrupted-append time-axis inconsistencies" in result.stdout

    def test_validate_consistent_store(self, tmp_path) -> None:
        """Validate command should pass for consistent test Zarr."""
        store = tmp_path / "ok.zarr"
        zg = zarr.open_group(str(store), mode="w")
        t = zg.create_dataset("time", data=np.arange(2), chunks=(2,))
        t.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        v = zg.create_dataset("velocity", data=np.zeros((2, 3), dtype=np.float32), chunks=(1, 3))
        v.attrs["_ARRAY_DIMENSIONS"] = ["time", "frequency"]
        zarr.consolidate_metadata(str(store))

        result = runner.invoke(app, ["validate", str(store)])
        assert result.exit_code == 0
        assert "Store is time-axis consistent" in result.stdout

    def test_dewarp_convert_help(self) -> None:
        """dewarp-convert --help documents cascade + Zarr pipeline."""
        result = runner.invoke(
            app,
            ["dewarp-convert", "--help"],
            color=False,
            terminal_width=120,
        )
        assert result.exit_code == 0
        plain = click.unstyle(result.stdout)
        assert "flow_cascade73MHz" in plain
        assert "image_plane_correction" in plain
        assert "--cascade-parent" in plain

    def test_dewarp_convert_help(self) -> None:
        """dewarp-convert --help documents cascade + Zarr pipeline."""
        result = runner.invoke(app, ["dewarp-convert", "--help"])
        assert result.exit_code == 0
        assert "flow_cascade73MHz" in result.stdout
        assert "image_plane_correction" in result.stdout
        assert "--cascade-parent" in result.stdout

    def test_convert_missing_args(self) -> None:
        """Test convert command with missing arguments."""
        result = runner.invoke(app, ["convert"])
        assert result.exit_code != 0
        # Typer outputs error messages, check for either stdout or that exit code is non-zero
        assert result.exit_code == 2  # Typer returns 2 for usage errors

    def test_convert_nonexistent_input(self, tmp_path) -> None:
        """Test convert command with nonexistent input directory."""
        nonexistent = tmp_path / "nonexistent"
        output = tmp_path / "output"

        result = runner.invoke(app, ["convert", str(nonexistent), str(output)])
        assert result.exit_code != 0
