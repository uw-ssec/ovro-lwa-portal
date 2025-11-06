#!/usr/bin/env python
"""Download test FITS files from Caltech S3 bucket.

This script downloads a ZIP file containing test FITS files from the
Caltech S3 bucket and extracts them for development and testing purposes.
It can be used in codespaces, GitHub Actions, or local development environments.

Environment Variables
--------------------
CALTECH_KEY : str
    AWS access key ID for Caltech S3 bucket
CALTECH_SECRET : str
    AWS secret access key for Caltech S3 bucket
CALTECH_ENDPOINT_URL : str
    S3 endpoint URL for Caltech storage
CALTECH_DEV_S3_BUCKET : str
    S3 bucket name containing test data

The script can load environment variables from a .env file using the --env-file option.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment,misc]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script.

    Parameters
    ----------
    verbose : bool, optional
        If True, set logging level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_env_file(env_file: Optional[Path] = None) -> None:
    """Load environment variables from a .env file.

    Parameters
    ----------
    env_file : Path, optional
        Path to the .env file. If None, tries to find .env in current directory

    Raises
    ------
    ImportError
        If python-dotenv package is not installed
    FileNotFoundError
        If specified env_file does not exist
    """
    if load_dotenv is None:
        msg = "python-dotenv package is required. Install it with: pip install python-dotenv"
        raise ImportError(msg)

    if env_file is not None:
        if not env_file.exists():
            msg = f".env file not found: {env_file}"
            raise FileNotFoundError(msg)
        logger.info(f"Loading environment variables from: {env_file}")
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        # Try to find .env in current directory
        default_env = Path(".env")
        if default_env.exists():
            logger.info(f"Loading environment variables from: {default_env}")
            load_dotenv(dotenv_path=default_env, override=False)
        else:
            logger.debug("No .env file found in current directory")


def validate_environment() -> dict[str, str]:
    """Validate that required environment variables are set.

    Returns
    -------
    dict[str, str]
        Dictionary containing validated environment variables

    Raises
    ------
    EnvironmentError
        If any required environment variable is missing
    """
    required_vars = [
        "CALTECH_KEY",
        "CALTECH_SECRET",
        "CALTECH_ENDPOINT_URL",
        "CALTECH_DEV_S3_BUCKET",
    ]

    env_vars = {}
    missing_vars = []

    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var] = value

    if missing_vars:
        msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        raise EnvironmentError(msg)

    logger.debug("Environment variables validated successfully")
    return env_vars


def download_fits_files(
    output_dir: Path,
    zip_filename: str = "fits_files.zip",
) -> None:
    """Download and extract FITS files ZIP from S3 bucket to local directory.

    Parameters
    ----------
    output_dir : Path
        Local directory where FITS files will be extracted
    zip_filename : str, optional
        Name of the ZIP file to download, by default "fits_files.zip"

    Raises
    ------
    EnvironmentError
        If required environment variables are not set
    ImportError
        If s3fs or tqdm package is not installed
    """
    try:
        import s3fs
    except ImportError as e:
        msg = "s3fs package is required. Install it with: pip install s3fs"
        raise ImportError(msg) from e

    if tqdm is None:
        msg = "tqdm package is required. Install it with: pip install tqdm"
        raise ImportError(msg)

    # Validate environment variables
    env_vars = validate_environment()

    # Setup the S3 filesystem
    logger.info("Connecting to S3 bucket...")
    fs = s3fs.S3FileSystem(
        key=env_vars["CALTECH_KEY"],
        secret=env_vars["CALTECH_SECRET"],
        endpoint_url=env_vars["CALTECH_ENDPOINT_URL"],
    )

    # Setup path to the S3 ZIP file
    s3_bucket = env_vars["CALTECH_DEV_S3_BUCKET"]
    remote_zip_path = str(PurePosixPath(s3_bucket) / "ovro-temp" / zip_filename)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Local path for the downloaded ZIP file
    local_zip_path = output_dir / zip_filename

    try:
        # Check if ZIP file exists on S3
        if not fs.exists(remote_zip_path):
            msg = f"ZIP file not found: {remote_zip_path}"
            raise FileNotFoundError(msg)

        # Get file size information
        file_info = fs.info(remote_zip_path)
        file_size_bytes = file_info.get("size", 0)
        file_size_mb = file_size_bytes / (1024 * 1024)

        logger.info(f"Downloading ZIP file from: {remote_zip_path}")
        logger.info(f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

        # Download the ZIP file with progress bar
        with (
            fs.open(remote_zip_path, "rb") as remote_file,
            local_zip_path.open("wb") as local_file,
            tqdm(
                total=file_size_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as pbar,
        ):
            # Download in chunks
            chunk_size = 8192  # 8 KB chunks
            while True:
                chunk = remote_file.read(chunk_size)
                if not chunk:
                    break
                local_file.write(chunk)
                pbar.update(len(chunk))

        logger.info(f"Downloaded ZIP file to: {local_zip_path}")

        # Extract the ZIP file
        logger.info(f"Extracting ZIP file to: {output_dir}")
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files = zip_ref.namelist()
            logger.info(f"Extracted {len(extracted_files)} file(s)")

        # Remove the ZIP file after extraction
        local_zip_path.unlink()
        logger.debug(f"Removed ZIP file: {local_zip_path}")

        logger.info(f"Successfully extracted FITS files to {output_dir}")

    except Exception as e:
        msg = f"Error downloading or extracting files: {e}"
        logger.error(msg)
        # Clean up ZIP file if it exists
        if local_zip_path.exists():
            local_zip_path.unlink()
            logger.debug(f"Cleaned up ZIP file: {local_zip_path}")
        raise


def main() -> int:
    """Main entry point for the script.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Download and extract test FITS files ZIP from Caltech S3 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  CALTECH_KEY              AWS access key ID for Caltech S3 bucket
  CALTECH_SECRET           AWS secret access key for Caltech S3 bucket
  CALTECH_ENDPOINT_URL     S3 endpoint URL for Caltech storage
  CALTECH_DEV_S3_BUCKET    S3 bucket name containing test data

Examples:
  # Download and extract FITS files to default directory
  python download_test_fits.py

  # Load environment variables from a .env file
  python download_test_fits.py --env-file .env

  # Download to specific directory with verbose logging
  python download_test_fits.py -o ./data -v

  # Download a different ZIP file
  python download_test_fits.py -z custom_fits.zip
        """,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("test_fits_files"),
        help="Output directory for extracted FITS files (default: test_fits_files)",
    )

    parser.add_argument(
        "-z",
        "--zip-filename",
        type=str,
        default="fits_files.zip",
        help="Name of the ZIP file to download (default: fits_files.zip)",
    )

    parser.add_argument(
        "-e",
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file to load environment variables (default: looks for .env in current directory)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Load environment variables from .env file if specified or found
    try:
        load_env_file(env_file=args.env_file)
    except ImportError as e:
        logger.warning(f"Could not load .env file: {e}")
        logger.info("Continuing with existing environment variables...")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    try:
        download_fits_files(
            output_dir=args.output_dir,
            zip_filename=args.zip_filename,
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to download and extract files: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
