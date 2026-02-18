"""Tests for the io module (open_dataset function)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import xarray as xr

from ovro_lwa_portal.io import (
    DataSourceError,
    _convert_osn_https_to_s3,
    _detect_source_type,
    _is_doi,
    _normalize_doi,
    _resolve_doi,
    _resolve_doi_from_metadata,
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

    def test_validate_valid_ovro_dataset(self) -> None:
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

        # Should not raise or warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            _validate_dataset(ds)  # Should not raise

    def test_validate_missing_dimensions(self) -> None:
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
        with pytest.warns(UserWarning, match="may not be OVRO-LWA format"):
            _validate_dataset(ds)

    def test_validate_missing_variables(self) -> None:
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
        with pytest.warns(UserWarning, match="may not be OVRO-LWA format"):
            _validate_dataset(ds)


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
        # The implementation passes an FSMap object, not the URL string directly
        call_arg = mock_open_zarr.call_args[0][0]
        # Verify it's an fsspec mapper (FSMap or similar)
        assert hasattr(call_arg, "fs") or hasattr(call_arg, "__getitem__")
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

        mock_resolve_doi.assert_called_once_with("10.5281/zenodo.1234567", production=True)
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

    @patch("requests.get")
    def test_resolve_doi_application_zarr(self, mock_get: Mock) -> None:
        """Test that application/zarr media type is matched."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/zarr",
                        "url": "https://example.com/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/data.zarr"

    @patch("requests.get")
    def test_resolve_doi_application_x_zarr(self, mock_get: Mock) -> None:
        """Test that application/x-zarr media type is matched."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/x-zarr",
                        "url": "https://uma1.osn.mghpcc.org/bucket/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        result = _resolve_doi("10.33569/test", production=False)
        assert result == "https://uma1.osn.mghpcc.org/bucket/data.zarr"

    @patch("requests.get")
    def test_resolve_doi_unknown_media_type_uses_first(self, mock_get: Mock) -> None:
        """Test that unknown media type falls back to first entry."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/octet-stream",
                        "url": "https://example.com/first.zarr",
                    }
                },
                {
                    "attributes": {
                        "mediaType": "text/plain",
                        "url": "https://example.com/second.txt",
                    }
                },
            ]
        }
        mock_get.return_value = mock_response

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/first.zarr"

    @patch("ovro_lwa_portal.io._resolve_doi_from_metadata")
    @patch("requests.get")
    def test_resolve_doi_empty_media_falls_back_to_metadata(
        self, mock_get: Mock, mock_metadata: Mock
    ) -> None:
        """Test that empty media list triggers metadata fallback."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response
        mock_metadata.return_value = "https://example.com/fallback.zarr"

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/fallback.zarr"
        mock_metadata.assert_called_once()

    @patch("ovro_lwa_portal.io._resolve_doi_from_metadata")
    @patch("requests.get")
    def test_resolve_doi_404_falls_back_to_metadata(
        self, mock_get: Mock, mock_metadata: Mock
    ) -> None:
        """Test that 404 from media endpoint triggers metadata fallback."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value.raise_for_status.side_effect = http_error
        mock_metadata.return_value = "https://example.com/fallback.zarr"

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/fallback.zarr"
        mock_metadata.assert_called_once()

    @patch("requests.get")
    def test_resolve_doi_uses_test_api(self, mock_get: Mock) -> None:
        """Test that production=False uses the test DataCite API."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/x-zarr",
                        "url": "https://example.com/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        _resolve_doi("10.33569/test", production=False)
        call_url = mock_get.call_args[0][0]
        assert "api.test.datacite.org" in call_url
        assert "api.datacite.org/dois" not in call_url


class TestDOIMetadataFallback:
    """Tests for _resolve_doi_from_metadata fallback."""

    @patch("requests.get")
    def test_resolve_from_metadata_success(self, mock_get: Mock) -> None:
        """Test successful URL extraction from DOI metadata."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": {
                "attributes": {
                    "url": "https://example.com/metadata-url.zarr",
                }
            }
        }
        mock_get.return_value = mock_response

        result = _resolve_doi_from_metadata(
            "10.5281/test", "https://api.datacite.org/dois"
        )
        assert result == "https://example.com/metadata-url.zarr"

    @patch("requests.get")
    def test_resolve_from_metadata_no_url(self, mock_get: Mock) -> None:
        """Test that missing URL in metadata raises DataSourceError."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": {"attributes": {"url": None}}
        }
        mock_get.return_value = mock_response

        with pytest.raises(DataSourceError, match="No download URL found"):
            _resolve_doi_from_metadata(
                "10.5281/test", "https://api.datacite.org/dois"
            )

    @patch("requests.get")
    def test_resolve_from_metadata_request_error(self, mock_get: Mock) -> None:
        """Test that request failure raises DataSourceError."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(DataSourceError, match="Failed to resolve DOI"):
            _resolve_doi_from_metadata(
                "10.5281/test", "https://api.datacite.org/dois"
            )


class TestOSNConversion:
    """Tests for _convert_osn_https_to_s3 URL conversion."""

    def test_convert_uma1_endpoint(self) -> None:
        """Test conversion of uma1 OSN path-style URL."""
        url = "https://uma1.osn.mghpcc.org/caltech-drill-core-temp/ovro-temp/test.zarr"
        opts = {"key": "ACCESS_KEY", "secret": "SECRET_KEY"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://caltech-drill-core-temp/ovro-temp/test.zarr"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://uma1.osn.mghpcc.org"
        assert updated_opts["key"] == "ACCESS_KEY"
        assert updated_opts["secret"] == "SECRET_KEY"

    def test_convert_caltech1_endpoint(self) -> None:
        """Test conversion of caltech1 OSN URL (the bug from issue #88)."""
        url = "https://caltech1.osn.mghpcc.org/10.25800/all_subbands_2024-05-24_first10.zarr"
        opts = {"key": "ACCESS_KEY", "secret": "SECRET_KEY"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://10.25800/all_subbands_2024-05-24_first10.zarr"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://caltech1.osn.mghpcc.org"

    def test_convert_unknown_endpoint(self) -> None:
        """Test that any *.osn.mghpcc.org URL converts correctly (path-style)."""
        url = "https://newsite99.osn.mghpcc.org/mybucket/path/to/data.zarr"
        opts = {"key": "K", "secret": "S"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://mybucket/path/to/data.zarr"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://newsite99.osn.mghpcc.org"

    def test_non_osn_url_unchanged(self) -> None:
        """Test that non-OSN HTTPS URLs are returned unchanged."""
        url = "https://example.com/data.zarr"
        opts = {"key": "K", "secret": "S"}

        result_url, result_opts = _convert_osn_https_to_s3(url, opts)

        assert result_url == url
        assert result_opts is opts  # Same object, not modified

    def test_s3_url_unchanged(self) -> None:
        """Test that S3 URLs are returned unchanged."""
        url = "s3://bucket/data.zarr"
        opts = {"key": "K", "secret": "S"}

        result_url, result_opts = _convert_osn_https_to_s3(url, opts)

        assert result_url == url
        assert result_opts is opts

    def test_storage_options_preserved(self) -> None:
        """Test that existing storage_options keys are preserved."""
        url = "https://uma1.osn.mghpcc.org/bucket/data.zarr"
        opts = {
            "key": "ACCESS_KEY",
            "secret": "SECRET_KEY",
            "client_kwargs": {"region_name": "us-east-1"},
        }

        _, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert updated_opts["key"] == "ACCESS_KEY"
        assert updated_opts["secret"] == "SECRET_KEY"
        assert updated_opts["client_kwargs"]["region_name"] == "us-east-1"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://uma1.osn.mghpcc.org"
        # Original should not be mutated
        assert "endpoint_url" not in opts.get("client_kwargs", {})

    def test_bucket_only_no_path(self) -> None:
        """Test conversion when URL has only a bucket, no further path."""
        url = "https://uma1.osn.mghpcc.org/bucket"
        opts = {"key": "K", "secret": "S"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://bucket"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://uma1.osn.mghpcc.org"
