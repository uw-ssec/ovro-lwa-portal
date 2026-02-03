"""Tests for the io module (open_dataset function)."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from fsspec.mapping import FSMap

import numpy as np
import pytest
import xarray as xr

from ovro_lwa_portal.io import (
    DATACITE_API_PRODUCTION,
    DATACITE_API_TEST,
    DataSourceError,
    _detect_source_type,
    _is_doi,
    _normalize_doi,
    _resolve_doi,
    _resolve_doi_from_metadata,
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
        loaded_ds = open_dataset(zarr_path)

        assert isinstance(loaded_ds, xr.Dataset)
        assert "SKY" in loaded_ds.data_vars
        assert set(loaded_ds.sizes.keys()) == {"time", "frequency", "l", "m"}

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
        loaded_ds = open_dataset(zarr_path, chunks=None)

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
        loaded_ds = open_dataset(url)

        mock_open_zarr.assert_called_once()
        store_arg = mock_open_zarr.call_args[0][0]
        # We now expect a Zarr store (FSMap), not a bare URL
        assert isinstance(store_arg, FSMap)
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
        loaded_ds = open_dataset(doi)

        mock_resolve_doi.assert_called_once_with(
            "10.5281/zenodo.1234567", production=True
        )
        mock_open_zarr.assert_called_once()
        assert isinstance(loaded_ds, xr.Dataset)

    @patch("ovro_lwa_portal.io._resolve_doi")
    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_doi_test_api(
        self, mock_open_zarr: Mock, mock_resolve_doi: Mock
    ) -> None:
        """Test opening a dataset via DOI using the test DataCite API."""
        mock_resolve_doi.return_value = "https://example.com/data.zarr"
        mock_ds = xr.Dataset(
            {
                "SKY": (["time", "frequency"], np.random.rand(2, 3)),
            }
        )
        mock_open_zarr.return_value = mock_ds

        loaded_ds = open_dataset("10.33569/9wsys-h7b71", production=False)

        mock_resolve_doi.assert_called_once_with(
            "10.33569/9wsys-h7b71", production=False
        )
        assert isinstance(loaded_ds, xr.Dataset)

    @patch("ovro_lwa_portal.io._resolve_doi")
    def test_open_doi_resolution_fails(self, mock_resolve_doi: Mock) -> None:
        """Test opening DOI when resolution fails."""
        mock_resolve_doi.side_effect = Exception("Resolution failed")

        doi = "doi:10.5281/zenodo.1234567"

        with pytest.raises(DataSourceError, match="Failed to resolve DOI"):
            open_dataset(doi)

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_dataset_load_failure(self, mock_open_zarr: Mock, tmp_path: Path) -> None:
        """Test that load failures are properly wrapped."""
        mock_open_zarr.side_effect = Exception("Load failed")

        zarr_path = tmp_path / "test.zarr"
        zarr_path.mkdir()

        with pytest.raises(DataSourceError, match="Failed to load dataset"):
            open_dataset(zarr_path)


class TestStorageOptions:
    """Tests for storage_options handling."""

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    @patch("upath.UPath")
    def test_storage_options_passed_to_upath(
        self, mock_upath_cls: Mock, mock_open_zarr: Mock
    ) -> None:
        """Test that storage_options are passed to UPath constructor."""
        mock_ds = xr.Dataset(
            {"SKY": (["time", "frequency"], np.random.rand(2, 3))}
        )
        mock_open_zarr.return_value = mock_ds

        mock_path = MagicMock()
        mock_path.protocol = "s3"
        mock_path.fs = MagicMock()
        mock_path.fs.get_mapper.return_value = MagicMock(spec=FSMap)
        mock_path.path = "bucket/data.zarr"
        mock_upath_cls.return_value = mock_path

        open_dataset(
            "s3://bucket/data.zarr",
            storage_options={"key": "AK", "secret": "SK"},
        )

        mock_upath_cls.assert_called_once_with(
            "s3://bucket/data.zarr", key="AK", secret="SK"
        )

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    @patch("upath.UPath")
    def test_no_storage_options(
        self, mock_upath_cls: Mock, mock_open_zarr: Mock
    ) -> None:
        """Test that UPath is called without extras when no storage_options."""
        mock_ds = xr.Dataset(
            {"SKY": (["time", "frequency"], np.random.rand(2, 3))}
        )
        mock_open_zarr.return_value = mock_ds

        mock_path = MagicMock()
        mock_path.protocol = "https"
        mock_path.fs = MagicMock()
        mock_path.fs.get_mapper.return_value = MagicMock(spec=FSMap)
        mock_path.path = "example.com/data.zarr"
        mock_upath_cls.return_value = mock_path

        open_dataset("https://example.com/data.zarr")

        mock_upath_cls.assert_called_once_with("https://example.com/data.zarr")


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

    @patch("requests.get")
    def test_resolve_doi_production(self, mock_get: Mock) -> None:
        """Test DOI resolution uses production API by default."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/zarr",
                        "url": "https://store.example.com/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        url = _resolve_doi("10.5281/zenodo.1234567", production=True)

        mock_get.assert_called_once_with(
            f"{DATACITE_API_PRODUCTION}/10.5281/zenodo.1234567/media", timeout=30
        )
        assert url == "https://store.example.com/data.zarr"

    @patch("requests.get")
    def test_resolve_doi_test_api(self, mock_get: Mock) -> None:
        """Test DOI resolution uses test API when production=False."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/zarr",
                        "url": "https://store.example.com/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        url = _resolve_doi("10.33569/9wsys-h7b71", production=False)

        mock_get.assert_called_once_with(
            f"{DATACITE_API_TEST}/10.33569/9wsys-h7b71/media", timeout=30
        )
        assert url == "https://store.example.com/data.zarr"

    @patch("requests.get")
    def test_resolve_doi_404_falls_back_to_metadata(self, mock_get: Mock) -> None:
        """Test that a 404 on the media endpoint falls back to DOI metadata."""
        import requests as real_requests

        # First call (media endpoint) returns 404
        mock_404 = Mock()
        mock_404.status_code = 404
        mock_404.raise_for_status.side_effect = real_requests.exceptions.HTTPError(
            response=mock_404
        )

        # Second call (metadata endpoint) returns a URL
        mock_metadata = Mock()
        mock_metadata.raise_for_status.return_value = None
        mock_metadata.json.return_value = {
            "data": {
                "attributes": {
                    "url": "https://store.example.com/fallback.zarr",
                }
            }
        }

        mock_get.side_effect = [mock_404, mock_metadata]

        url = _resolve_doi("10.33569/test-doi", production=False)

        assert url == "https://store.example.com/fallback.zarr"
        assert mock_get.call_count == 2
        mock_get.assert_any_call(
            f"{DATACITE_API_TEST}/10.33569/test-doi/media", timeout=30
        )
        mock_get.assert_any_call(
            f"{DATACITE_API_TEST}/10.33569/test-doi", timeout=30
        )

    @patch("requests.get")
    def test_resolve_doi_from_metadata_no_url(self, mock_get: Mock) -> None:
        """Test that metadata fallback raises when no URL found."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": {"attributes": {}}}
        mock_get.return_value = mock_response

        with pytest.raises(DataSourceError, match="No download URL found"):
            _resolve_doi_from_metadata("10.33569/no-url", DATACITE_API_TEST)
