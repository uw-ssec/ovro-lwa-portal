"""Tests for CI helper scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add .ci-helpers to path to import the module
ci_helpers_path = Path(__file__).parent.parent / ".ci-helpers"
sys.path.insert(0, str(ci_helpers_path))

import download_test_fits  # noqa: E402


def test_validate_environment_success():
    """Test that validate_environment succeeds with all required vars set."""
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        result = download_test_fits.validate_environment()
        assert result == env_vars


def test_validate_environment_missing_vars():
    """Test that validate_environment raises error when vars are missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            download_test_fits.validate_environment()

        error_msg = str(exc_info.value)
        assert "Missing required environment variables" in error_msg
        assert "CALTECH_KEY" in error_msg
        assert "CALTECH_SECRET" in error_msg


def test_validate_environment_partial_vars():
    """Test that validate_environment raises error when some vars are missing."""
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            download_test_fits.validate_environment()

        error_msg = str(exc_info.value)
        assert "CALTECH_ENDPOINT_URL" in error_msg
        assert "CALTECH_DEV_S3_BUCKET" in error_msg


def test_download_fits_files_missing_s3fs(tmp_path):
    """Test that download_fits_files raises ImportError when s3fs is not installed."""
    with patch.dict("sys.modules", {"s3fs": None}):
        with pytest.raises(ImportError) as exc_info:
            download_test_fits.download_fits_files(tmp_path)

        error_msg = str(exc_info.value)
        assert "s3fs package is required" in error_msg


def test_download_fits_files_success(tmp_path):
    """Test successful download of FITS files."""
    # Setup mock S3 filesystem
    mock_fs = MagicMock()
    mock_s3fs_class = MagicMock(return_value=mock_fs)

    # Mock glob to return some test files
    mock_fs.glob.return_value = [
        "test-bucket/ovro-temp/fits/file1.fits",
        "test-bucket/ovro-temp/fits/file2.fits",
    ]

    # Setup environment variables
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("s3fs.S3FileSystem", mock_s3fs_class):
            download_test_fits.download_fits_files(tmp_path)

            # Verify S3FileSystem was called with correct credentials
            mock_s3fs_class.assert_called_once_with(
                key="test_key",
                secret="test_secret",
                endpoint_url="https://test.endpoint.com",
            )

            # Verify glob was called
            mock_fs.glob.assert_called_once()

            # Verify get was called for each file
            assert mock_fs.get.call_count == 2


def test_download_fits_files_no_files_found(tmp_path):
    """Test behavior when no files match the pattern."""
    # Setup mock S3 filesystem
    mock_fs = MagicMock()
    mock_s3fs_class = MagicMock(return_value=mock_fs)

    # Mock glob to return empty list
    mock_fs.glob.return_value = []

    # Setup environment variables
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("s3fs.S3FileSystem", mock_s3fs_class):
            # Should not raise error, just log warning
            download_test_fits.download_fits_files(tmp_path)

            # Verify get was not called
            mock_fs.get.assert_not_called()


def test_download_fits_files_with_subdir(tmp_path):
    """Test download with subdirectory specified."""
    # Setup mock S3 filesystem
    mock_fs = MagicMock()
    mock_s3fs_class = MagicMock(return_value=mock_fs)

    # Mock glob to return test files
    mock_fs.glob.return_value = ["test-bucket/ovro-temp/fits/subdir/file1.fits"]

    # Setup environment variables
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("s3fs.S3FileSystem", mock_s3fs_class):
            download_test_fits.download_fits_files(tmp_path, remote_subdir="subdir")

            # Verify glob was called with subdirectory in path
            call_args = mock_fs.glob.call_args[0][0]
            assert "subdir" in call_args


def test_setup_logging():
    """Test logging setup."""
    # Test default (INFO) level
    download_test_fits.setup_logging(verbose=False)
    # Just verify it doesn't raise an exception

    # Test verbose (DEBUG) level
    download_test_fits.setup_logging(verbose=True)
    # Just verify it doesn't raise an exception


def test_main_missing_environment(capsys):
    """Test main function with missing environment variables."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("sys.argv", ["download_test_fits.py"]):
            exit_code = download_test_fits.main()
            assert exit_code == 1


@patch("download_test_fits.download_fits_files")
def test_main_success(mock_download):
    """Test main function with successful download."""
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("sys.argv", ["download_test_fits.py"]):
            exit_code = download_test_fits.main()
            assert exit_code == 0
            mock_download.assert_called_once()
