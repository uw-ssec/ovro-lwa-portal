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


def test_download_fits_files_missing_tqdm(tmp_path):
    """Test that download_fits_files raises ImportError when tqdm is not installed."""
    # Temporarily set tqdm to None to simulate it not being installed
    original_tqdm = download_test_fits.tqdm
    try:
        download_test_fits.tqdm = None
        with pytest.raises(ImportError) as exc_info:
            download_test_fits.download_fits_files(tmp_path)

        error_msg = str(exc_info.value)
        assert "tqdm package is required" in error_msg
    finally:
        download_test_fits.tqdm = original_tqdm


def test_download_fits_files_success(tmp_path):
    """Test successful download and extraction of ZIP file."""
    import io
    import zipfile as zf

    # Setup mock S3 filesystem
    mock_fs = MagicMock()
    mock_s3fs_class = MagicMock(return_value=mock_fs)

    # Mock exists to return True
    mock_fs.exists.return_value = True

    # Mock info to return file size
    mock_fs.info.return_value = {"size": 1024}

    # Create a mock ZIP file in memory
    zip_buffer = io.BytesIO()
    with zf.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("file1.fits", "mock fits data 1")
        zip_file.writestr("file2.fits", "mock fits data 2")
    zip_data = zip_buffer.getvalue()
    zip_buffer.seek(0)

    # Mock open to return a file-like object
    mock_fs.open.return_value.__enter__.return_value = io.BytesIO(zip_data)

    # Mock tqdm
    mock_tqdm = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = MagicMock()

    # Setup environment variables
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("s3fs.S3FileSystem", mock_s3fs_class):
            with patch("download_test_fits.tqdm", mock_tqdm):
                download_test_fits.download_fits_files(tmp_path)

                # Verify S3FileSystem was called with correct credentials
                mock_s3fs_class.assert_called_once_with(
                    key="test_key",
                    secret="test_secret",
                    endpoint_url="https://test.endpoint.com",
                )

                # Verify exists was called
                mock_fs.exists.assert_called_once()

                # Verify info was called to get file size
                mock_fs.info.assert_called_once()

                # Verify open was called to read the file
                mock_fs.open.assert_called_once()

                # Verify files were extracted
                assert (tmp_path / "file1.fits").exists()
                assert (tmp_path / "file2.fits").exists()

                # Verify ZIP file was removed after extraction
                assert not (tmp_path / "fits_files.zip").exists()


def test_download_fits_files_not_found(tmp_path):
    """Test behavior when ZIP file doesn't exist."""
    # Setup mock S3 filesystem
    mock_fs = MagicMock()
    mock_s3fs_class = MagicMock(return_value=mock_fs)

    # Mock exists to return False
    mock_fs.exists.return_value = False

    # Setup environment variables
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("s3fs.S3FileSystem", mock_s3fs_class):
            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                download_test_fits.download_fits_files(tmp_path)

            error_msg = str(exc_info.value)
            assert "ZIP file not found" in error_msg

            # Verify open was not called
            mock_fs.open.assert_not_called()


def test_download_fits_files_custom_zip(tmp_path):
    """Test download with custom ZIP filename."""
    import io
    import zipfile as zf

    # Setup mock S3 filesystem
    mock_fs = MagicMock()
    mock_s3fs_class = MagicMock(return_value=mock_fs)

    # Mock exists to return True
    mock_fs.exists.return_value = True

    # Mock info to return file size
    mock_fs.info.return_value = {"size": 512}

    # Create a mock ZIP file in memory
    zip_buffer = io.BytesIO()
    with zf.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("custom.fits", "custom fits data")
    zip_data = zip_buffer.getvalue()
    zip_buffer.seek(0)

    # Mock open to return a file-like object
    mock_fs.open.return_value.__enter__.return_value = io.BytesIO(zip_data)

    # Mock tqdm
    mock_tqdm = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = MagicMock()

    # Setup environment variables
    env_vars = {
        "CALTECH_KEY": "test_key",
        "CALTECH_SECRET": "test_secret",
        "CALTECH_ENDPOINT_URL": "https://test.endpoint.com",
        "CALTECH_DEV_S3_BUCKET": "test-bucket",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        with patch("s3fs.S3FileSystem", mock_s3fs_class):
            with patch("download_test_fits.tqdm", mock_tqdm):
                download_test_fits.download_fits_files(tmp_path, zip_filename="custom.zip")

                # Verify exists was called with custom filename
                call_args = mock_fs.exists.call_args[0][0]
                assert "custom.zip" in call_args


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
