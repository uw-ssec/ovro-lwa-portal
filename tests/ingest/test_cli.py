"""Tests for the CLI module."""

from __future__ import annotations

import pytest
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
        assert result.exit_code == 0
        assert "Convert OVRO-LWA FITS files to a single Zarr store" in result.stdout

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
