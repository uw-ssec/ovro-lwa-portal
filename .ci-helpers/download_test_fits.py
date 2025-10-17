#!/usr/bin/env python
"""Download test FITS files from Caltech S3 bucket.

This script downloads test FITS files from the Caltech S3 bucket for
development and testing purposes. It can be used in codespaces,
GitHub Actions, or local development environments.

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
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Optional

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
    pattern: str = "*.fits",
    remote_subdir: Optional[str] = None,
) -> None:
    """Download FITS files from S3 bucket to local directory.

    Parameters
    ----------
    output_dir : Path
        Local directory where FITS files will be downloaded
    pattern : str, optional
        Glob pattern for files to download, by default "*.fits"
    remote_subdir : str, optional
        Subdirectory within the remote test fits directory, by default None

    Raises
    ------
    EnvironmentError
        If required environment variables are not set
    ImportError
        If s3fs package is not installed
    """
    try:
        import s3fs
    except ImportError as e:
        msg = "s3fs package is required. Install it with: pip install s3fs"
        raise ImportError(msg) from e

    # Validate environment variables
    env_vars = validate_environment()

    # Setup the S3 filesystem
    logger.info("Connecting to S3 bucket...")
    fs = s3fs.S3FileSystem(
        key=env_vars["CALTECH_KEY"],
        secret=env_vars["CALTECH_SECRET"],
        endpoint_url=env_vars["CALTECH_ENDPOINT_URL"],
    )

    # Setup path to the S3 remote directory to the test fits data
    s3_bucket = env_vars["CALTECH_DEV_S3_BUCKET"]
    remote_test_fits_dir = PurePosixPath(s3_bucket) / "ovro-temp" / "fits"

    # Add subdirectory if specified
    if remote_subdir:
        remote_test_fits_dir = remote_test_fits_dir / remote_subdir

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Construct the full remote path with pattern
    remote_path = str(remote_test_fits_dir / pattern)
    logger.info(f"Searching for files matching: {remote_path}")

    try:
        # List files matching the pattern
        files = fs.glob(remote_path)
        if not files:
            logger.warning(f"No files found matching pattern: {remote_path}")
            return

        logger.info(f"Found {len(files)} file(s) to download")

        # Download each file
        for file_path in files:
            file_name = PurePosixPath(file_path).name
            local_path = output_dir / file_name
            logger.info(f"Downloading: {file_name}")
            fs.get(file_path, str(local_path))
            logger.debug(f"Downloaded to: {local_path}")

        logger.info(f"Successfully downloaded {len(files)} file(s) to {output_dir}")

    except Exception as e:
        msg = f"Error downloading files: {e}"
        logger.error(msg)
        raise


def main() -> int:
    """Main entry point for the script.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Download test FITS files from Caltech S3 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  CALTECH_KEY              AWS access key ID for Caltech S3 bucket
  CALTECH_SECRET           AWS secret access key for Caltech S3 bucket
  CALTECH_ENDPOINT_URL     S3 endpoint URL for Caltech storage
  CALTECH_DEV_S3_BUCKET    S3 bucket name containing test data

Examples:
  # Download all FITS files to default directory
  python download_test_fits.py

  # Download to specific directory with pattern
  python download_test_fits.py -o /path/to/output -p "20240101_*.fits"

  # Download from subdirectory with verbose logging
  python download_test_fits.py -o ./data -s subdir -v
        """,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("test_fits_files"),
        help="Output directory for downloaded FITS files (default: test_fits_files)",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.fits",
        help="Glob pattern for files to download (default: *.fits)",
    )

    parser.add_argument(
        "-s",
        "--subdir",
        type=str,
        default=None,
        help="Subdirectory within the remote test fits directory",
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

    try:
        download_fits_files(
            output_dir=args.output_dir,
            pattern=args.pattern,
            remote_subdir=args.subdir,
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to download files: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
