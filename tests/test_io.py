"""Tests for the io module (open_dataset function)."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import xarray as xr

from ovro_lwa_portal.io import (
    DataSourceError,
    _detect_source_type,
    _is_doi,
    _normalize_doi,
    _validate_dataset,
    open_dataset,
)


class TestDOIDetection:
    """Tests for DOI detection and normalization."""

    def test_is_doi_with_prefix(self) -> None:
        """Test DOI detection with 'doi:' prefix."""
        assert _is_doi("doi:10.5281/zenodo.1234567")
        assert _is_doi("DOI:10.5281/zenodo.1234567")

    def test_is_doi_without_prefix(self) -> None:
        """Test DOI detection without prefix."""
        assert _is_doi("10.5281/zenodo.1234567")
        assert _is_doi("10.1234/abcd.efgh")

    def test_is_not_doi(self) -> None:
        """Test that non-DOI strings are not detected as DOIs."""
        assert not _is_doi("https://example.com")
        assert not _is_doi("/path/to/file")
        assert not _is_doi("s3://bucket/key")
        assert not _is_doi("not-a-doi")

    def test_normalize_doi_with_prefix(self) -> None:
        """Test DOI normalization removes prefix."""
        assert _normalize_doi("doi:10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"
        assert _normalize_doi("DOI:10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"

    def test_normalize_doi_without_prefix(self) -> None:
        """Test DOI normalization preserves DOI without prefix."""
        assert _normalize_doi("10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"


class TestSourceTypeDetection:
    """Tests for source type detection."""

    def test_detect_local_path(self) -> None:
        """Test detection of local file paths."""
        source_type, normalized = _detect_source_type("/path/to/data.zarr")
        assert source_type == "local"
        assert normalized == "/path/to/data.zarr"

        source_type, normalized = _detect_source_type(Path("relative/path/data.zarr"))
        assert source_type == "local"

    def test_detect_remote_http(self) -> None:
        """Test detection of HTTP/HTTPS URLs."""
        source_type, normalized = _detect_source_type("https://example.com/data.zarr")
        assert source_type == "remote"
        assert normalized == "https://example.com/data.zarr"

        source_type, normalized = _detect_source_type("http://example.com/data.zarr")
        assert source_type == "remote"

    def test_detect_remote_s3(self) -> None:
        """Test detection of S3 URLs."""
        source_type, normalized = _detect_source_type("s3://bucket/data.zarr")
        assert source_type == "remote"
        assert normalized == "s3://bucket/data.zarr"

    def test_detect_remote_gcs(self) -> None:
        """Test detection of Google Cloud Storage URLs."""
        source_type, normalized = _detect_source_type("gs://bucket/data.zarr")
        assert source_type == "remote"

        source_type, normalized = _detect_source_type("gcs://bucket/data.zarr")
        assert source_type == "remote"

    def test_detect_doi(self) -> None:
        """Test detection of DOI identifiers."""
        source_type, normalized = _detect_source_type("doi:10.5281/zenodo.1234567")
        assert source_type == "doi"
        assert normalized == "10.5281/zenodo.1234567"

        source_type, normalized = _detect_source_type("10.5281/zenodo.1234567")
        assert source_type == "doi"
        assert normalized == "10.5281/zenodo.1234567"


class TestDatasetValidation:
    """Tests for dataset validation."""

    def test_validate_valid_ovro_dataset(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test validation of a valid OVRO-LWA dataset."""
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(2, 3, 10, 10)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
                "l": np.arange(10),
                "m": np.arange(10),
            },
        )

        # Should not raise
        _validate_dataset(ds)

        # Should log info about dimensions and variables
        assert "dimensions" in caplog.text.lower()

    def test_validate_missing_dimensions(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test validation warns about missing expected dimensions."""
        ds = xr.Dataset(
            {
                "data": (["x", "y"], np.random.rand(10, 10)),
            },
            coords={
                "x": np.arange(10),
                "y": np.arange(10),
            },
        )

        # Should not raise but should warn
        _validate_dataset(ds)
        assert "may not be OVRO-LWA format" in caplog.text

    def test_validate_missing_variables(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test validation warns about missing expected variables."""
        ds = xr.Dataset(
            {
                "other_var": (["time", "frequency"], np.random.rand(2, 3)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
            },
        )

        # Should not raise but should warn
        _validate_dataset(ds)
        assert "may not be OVRO-LWA format" in caplog.text


class TestOpenDataset:
    """Tests for open_dataset function."""

    def test_open_local_zarr(self, tmp_path: Path) -> None:
        """Test opening a local zarr store."""
        # Create a test zarr store
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(2, 3, 10, 10)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
                "l": np.arange(10),
                "m": np.arange(10),
            },
        )
        ds.to_zarr(zarr_path)

        # Load it back
        loaded_ds = open_dataset(zarr_path, validate=False)

        assert isinstance(loaded_ds, xr.Dataset)
        assert "SKY" in loaded_ds.data_vars
        assert set(loaded_ds.sizes.keys()) == {"time", "frequency", "l", "m"}

    def test_open_local_zarr_with_validation(self, tmp_path: Path) -> None:
        """Test opening a local zarr store with validation."""
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(2, 3, 10, 10)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
                "l": np.arange(10),
                "m": np.arange(10),
            },
        )
        ds.to_zarr(zarr_path)

        # Load with validation (default)
        loaded_ds = open_dataset(zarr_path)

        assert isinstance(loaded_ds, xr.Dataset)

    def test_open_nonexistent_local_path(self, tmp_path: Path) -> None:
        """Test opening a nonexistent local path raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.zarr"

        with pytest.raises(FileNotFoundError, match="Local path does not exist"):
            open_dataset(nonexistent)

    def test_open_with_custom_chunks(self, tmp_path: Path) -> None:
        """Test opening with custom chunk specification."""
        import warnings
        
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(10, 20, 100, 100)),
            },
            coords={
                "time": np.arange(10),
                "frequency": np.arange(20),
                "l": np.arange(100),
                "m": np.arange(100),
            },
        )
        ds.to_zarr(zarr_path)

        # Load with custom chunks (may trigger performance warning from xarray)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded_ds = open_dataset(
                zarr_path,
                chunks={"time": 5, "frequency": 10},
                validate=False,
            )

        assert isinstance(loaded_ds, xr.Dataset)
        # Check that data is chunked (dask array)
        assert hasattr(loaded_ds["SKY"].data, "chunks")

    def test_open_with_no_chunks(self, tmp_path: Path) -> None:
        """Test opening without chunking (load into memory)."""
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency"], np.random.rand(2, 3)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
            },
        )
        ds.to_zarr(zarr_path)

        # Load without chunks
        loaded_ds = open_dataset(zarr_path, chunks=None, validate=False)

        assert isinstance(loaded_ds, xr.Dataset)
        # Data should be numpy array, not dask
        assert isinstance(loaded_ds["SKY"].data, np.ndarray)

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_remote_http(self, mock_open_zarr: Mock) -> None:
        """Test opening a remote HTTP URL."""
        mock_ds = xr.Dataset(
            {
                "SKY": (["time", "frequency"], np.random.rand(2, 3)),
            }
        )
        mock_open_zarr.return_value = mock_ds

        url = "https://example.com/data.zarr"
        loaded_ds = open_dataset(url, validate=False)

        mock_open_zarr.assert_called_once()
        assert mock_open_zarr.call_args[0][0] == url
        assert isinstance(loaded_ds, xr.Dataset)

    def test_open_remote_s3_requires_s3fs(self) -> None:
        """Test that S3 URL detection works (actual loading requires s3fs)."""
        # We can't easily test S3 loading without s3fs installed
        # This test just verifies source type detection
        source_type, normalized = _detect_source_type("s3://bucket/data.zarr")
        assert source_type == "remote"
        assert normalized == "s3://bucket/data.zarr"

    @patch("ovro_lwa_portal.io._resolve_doi")
    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_doi(self, mock_open_zarr: Mock, mock_resolve_doi: Mock) -> None:
        """Test opening a dataset via DOI."""
        mock_resolve_doi.return_value = "https://example.com/data.zarr"
        mock_ds = xr.Dataset(
            {
                "SKY": (["time", "frequency"], np.random.rand(2, 3)),
            }
        )
        mock_open_zarr.return_value = mock_ds

        doi = "doi:10.5281/zenodo.1234567"
        loaded_ds = open_dataset(doi, validate=False)

        mock_resolve_doi.assert_called_once_with("10.5281/zenodo.1234567")
        mock_open_zarr.assert_called_once()
        assert isinstance(loaded_ds, xr.Dataset)

    @patch("ovro_lwa_portal.io._resolve_doi")
    def test_open_doi_resolution_fails(self, mock_resolve_doi: Mock) -> None:
        """Test opening DOI when resolution fails."""
        mock_resolve_doi.side_effect = Exception("Resolution failed")

        doi = "doi:10.5281/zenodo.1234567"

        with pytest.raises(DataSourceError, match="Failed to resolve DOI"):
            open_dataset(doi, validate=False)

    def test_open_unsupported_engine(self, tmp_path: Path) -> None:
        """Test opening with unsupported engine raises error."""
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset({"data": (["x"], np.arange(10))})
        ds.to_zarr(zarr_path)

        with pytest.raises(DataSourceError, match="Unsupported engine"):
            open_dataset(zarr_path, engine="netcdf", validate=False)

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_dataset_load_failure(self, mock_open_zarr: Mock, tmp_path: Path) -> None:
        """Test that load failures are properly wrapped."""
        mock_open_zarr.side_effect = Exception("Load failed")

        zarr_path = tmp_path / "test.zarr"
        zarr_path.mkdir()

        with pytest.raises(DataSourceError, match="Failed to load dataset"):
            open_dataset(zarr_path, validate=False)


class TestDOIResolution:
    """Tests for DOI resolution functionality."""

    def test_doi_resolution_requires_dependencies(self) -> None:
        """Test that DOI resolution requires requests library."""
        # DOI resolution requires optional dependencies
        # This test just verifies DOI detection works
        assert _is_doi("doi:10.5281/zenodo.1234567")
        assert _is_doi("10.5281/zenodo.1234567")
        
        # Verify normalization
        assert _normalize_doi("doi:10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"
