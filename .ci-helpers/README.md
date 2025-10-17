# CI Helper Scripts

This directory contains helper scripts for continuous integration (CI) and
development workflows.

## Scripts

### download_test_fits.py

Downloads test FITS files from the Caltech S3 bucket for development and
testing purposes.

#### Requirements

- Python 3.12+
- s3fs package (install with: `pip install s3fs`)

#### Environment Variables

The script requires the following environment variables to be set:

- `CALTECH_KEY`: AWS access key ID for Caltech S3 bucket
- `CALTECH_SECRET`: AWS secret access key for Caltech S3 bucket
- `CALTECH_ENDPOINT_URL`: S3 endpoint URL for Caltech storage
- `CALTECH_DEV_S3_BUCKET`: S3 bucket name containing test data

#### Usage

```bash
# Download all FITS files to default directory (test_fits_files)
python .ci-helpers/download_test_fits.py

# Download to specific directory
python .ci-helpers/download_test_fits.py -o /path/to/output

# Download files matching a specific pattern
python .ci-helpers/download_test_fits.py -p "20240101_*.fits"

# Download from a subdirectory within the remote test fits directory
python .ci-helpers/download_test_fits.py -s subdir

# Enable verbose logging
python .ci-helpers/download_test_fits.py -v

# Combine options
python .ci-helpers/download_test_fits.py -o ./data -p "*.fits" -s subdir -v
```

#### Options

- `-o, --output-dir PATH`: Output directory for downloaded FITS files (default:
  test_fits_files)
- `-p, --pattern PATTERN`: Glob pattern for files to download (default:
  \*.fits)
- `-s, --subdir DIR`: Subdirectory within the remote test fits directory
- `-v, --verbose`: Enable verbose logging (DEBUG level)
- `-h, --help`: Show help message and exit

#### Example in GitHub Actions

```yaml
- name: Set up environment
  env:
    CALTECH_KEY: ${{ secrets.CALTECH_KEY }}
    CALTECH_SECRET: ${{ secrets.CALTECH_SECRET }}
    CALTECH_ENDPOINT_URL: ${{ secrets.CALTECH_ENDPOINT_URL }}
    CALTECH_DEV_S3_BUCKET: ${{ secrets.CALTECH_DEV_S3_BUCKET }}
  run: |
    pip install s3fs
    python .ci-helpers/download_test_fits.py -v
```
