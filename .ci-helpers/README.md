# CI Helper Scripts

This directory contains helper scripts for continuous integration (CI) and
development workflows.

## Scripts

### download_test_fits.py

Downloads and extracts a ZIP file containing test FITS files from the Caltech
S3 bucket for development and testing purposes.

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
# Download and extract FITS files to default directory (test_fits_files)
python .ci-helpers/download_test_fits.py

# Download to specific directory
python .ci-helpers/download_test_fits.py -o /path/to/output

# Download a different ZIP file
python .ci-helpers/download_test_fits.py -z custom_fits.zip

# Enable verbose logging
python .ci-helpers/download_test_fits.py -v

# Combine options
python .ci-helpers/download_test_fits.py -o ./data -z fits_files.zip -v
```

#### Options

- `-o, --output-dir PATH`: Output directory for extracted FITS files (default:
  test_fits_files)
- `-z, --zip-filename NAME`: Name of the ZIP file to download (default:
  fits_files.zip)
- `-v, --verbose`: Enable verbose logging (DEBUG level)
- `-h, --help`: Show help message and exit

#### How It Works

1. Connects to the Caltech S3 bucket using provided credentials
2. Downloads the ZIP file from `s3://{bucket}/ovro-temp/{zip_filename}`
3. Extracts all FITS files to the specified output directory
4. Removes the ZIP file after successful extraction

#### Example in GitHub Actions

```yaml
- name: Download test data
  env:
    CALTECH_KEY: ${{ secrets.CALTECH_KEY }}
    CALTECH_SECRET: ${{ secrets.CALTECH_SECRET }}
    CALTECH_ENDPOINT_URL: ${{ secrets.CALTECH_ENDPOINT_URL }}
    CALTECH_DEV_S3_BUCKET: ${{ secrets.CALTECH_DEV_S3_BUCKET }}
  run: |
    pip install s3fs
    python .ci-helpers/download_test_fits.py -v
```
