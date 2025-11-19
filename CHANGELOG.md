# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`open_dataset()` function** (#72): Unified interface for loading OVRO-LWA data from multiple sources
  - Load from local file paths
  - Load from remote URLs (HTTP/HTTPS, S3, Google Cloud Storage)
  - Load via DOI identifiers with automatic resolution
  - Automatic source type detection
  - Configurable chunking strategies for large datasets
  - Data validation to ensure OVRO-LWA schema compliance
  - Comprehensive error handling with clear messages
  - Support for lazy loading with dask
  - Optional dependencies for remote access (`ovro_lwa_portal[remote]`)
- New `ovro_lwa_portal.io` module with data loading utilities
- Documentation for `open_dataset()` in `docs/open_dataset.md`
- Example notebook demonstrating `open_dataset()` usage
- Comprehensive test suite for data loading functionality
- Integration tests with realistic OVRO-LWA data structures

### Changed

- Updated main README to highlight new data loading capabilities
- Added `remote` optional dependency group to `pyproject.toml`
- Exported `open_dataset` from main package namespace

## [Previous Versions]

See git history for changes in previous versions.
