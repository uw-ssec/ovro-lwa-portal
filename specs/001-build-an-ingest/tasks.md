# Implementation Tasks: FITS to Zarr Ingest Package

**Feature**: 001-build-an-ingest **Date**: 2025-11-05 **Status**: Partially
Implemented **Last Updated**: November 6, 2025

**Implementation Status**: See
[IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) for detailed report.

## Overview

This document outlined a comprehensive, ordered list of 79 tasks for
implementing the FITS to Zarr ingestion pipeline following Test-Driven
Development (TDD) principles with a Pydantic-first approach.

**IMPORTANT NOTE**: The actual implementation took a **different, simplified
approach**:

- **Wrapper-based architecture** instead of full modular rewrite
- **Simple dataclasses** instead of Pydantic models
- **~23 of 79 tasks completed** (~29%), but **core functionality is 100%
  complete**
- Tasks marked with 🟢 were completed, 🔵 were completed differently, ⚪ were
  skipped

**Total Planned Tasks**: 79 (75 required + 4 optional Prefect tasks) **Total
Completed**: ~23 tasks (core functionality complete) **Estimated Time Planned**:
22-65 hours **Actual Time**: Significantly less due to simplified approach

## Task Organization

### Task Types

- `setup`: Environment and configuration setup
- `pydantic-model`: Pydantic model definition with validators
- `pydantic-settings`: Configuration management with pydantic-settings
- `test`: Test implementation (written before code)
- `implement`: Code implementation (after tests)
- `refactor`: Code refactoring (preserve behavior)
- `integration`: End-to-end integration test
- `docs`: Documentation

### Task Properties

- **Layer**: Architectural layer (1-8)
- **Dependencies**: Tasks that must complete first
- **Parallel**: `[P]` marker indicates can run concurrently with other `[P]`
  tasks
- **Contract**: Reference to contract specification
- **Acceptance**: Clear completion criteria
- **Time**: Rough time estimate

---

## Layer 0: Setup & Configuration

### T001: Update pyproject.toml with new dependencies

**Type**: setup **Layer**: 0 **Dependencies**: None **Parallel**: No **Time**:
10 min

**Description**: Add required dependencies to pyproject.toml

**Tasks**:

1. Add `pydantic>=2.0` to `[project.dependencies]`
2. Add `pydantic-settings>=2.0` to `[project.dependencies]`
3. Add `typer>=0.12.0` to `[project.dependencies]`
4. Add `rich>=13.0.0` to `[project.dependencies]`
5. Add `portalocker>=2.8.0` to `[project.dependencies]`
6. Add `prefect>=2.14.0` to `[project.optional-dependencies]` under `prefect`
   key
7. Add CLI entry point: `ovro-lwa-ingest = "ovro_lwa_portal.ingest.cli:app"`

**Acceptance**:

- [ ] All dependencies added to pyproject.toml
- [ ] CLI entry point registered
- [ ] `pixi install` succeeds without errors

---

### T002: Create ingest package structure

**Type**: setup **Layer**: 0 **Dependencies**: None **Parallel**: No **Time**: 5
min

**Description**: Create directory structure for ingest subpackage

**Tasks**:

1. Create `src/ovro_lwa_portal/ingest/` directory
2. Create `src/ovro_lwa_portal/ingest/__init__.py`
3. Create `tests/ingest/` directory
4. Create `tests/ingest/__init__.py`
5. Create `tests/ingest/conftest.py`

**Acceptance**:

- [ ] Directory structure exists
- [ ] Python can import `ovro_lwa_portal.ingest`

---

## Layer 1: Pydantic Models (Foundation)

### T003: Write Pydantic model validation tests (FITSImageFile)

**Type**: test **Layer**: 1 **Dependencies**: T002 **Parallel**: No
**Contract**: data-model.md (FITSImageFile) **Time**: 15 min

**Description**: Write validation tests for FITSImageFile Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid FITSImageFile creation
2. Test path validation (path must exist)
3. Test observation_date validation (8 digits, YYYYMMDD format)
4. Test observation_time validation (6 digits, HHMMSS format)
5. Test subband_mhz validation (positive integer)
6. Test time_step_key property
7. Test frozen model (immutability)
8. Test serialization (`model_dump()`, `model_dump_json()`)

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)
- [ ] ≥85% coverage for FITSImageFile when implemented

---

### T004: Implement FITSImageFile Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T003 **Parallel**: No
**Contract**: data-model.md (FITSImageFile) **Time**: 20 min

**Description**: Implement FITSImageFile as frozen Pydantic BaseModel

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `FITSImageFile(BaseModel)` with frozen config
2. Add fields: `path: Path`, `observation_date: str`, `observation_time: str`,
   `subband_mhz: int`, `is_fixed: bool`
3. Add `@field_validator` for path existence
4. Add `@field_validator` for date format (8 digits, valid date)
5. Add `@field_validator` for time format (6 digits, valid time)
6. Add `@field_validator` for positive frequency
7. Implement `time_step_key` as `@property`
8. Add Field descriptions for documentation

**Acceptance**:

- [ ] All validation tests (T003) pass
- [ ] Model is frozen (immutable)
- [ ] Serialization works correctly

---

### T005: Write Pydantic model validation tests (FixedFITSFile)

**Type**: test **Layer**: 1 **Dependencies**: T004 **Parallel**: [P]
**Contract**: data-model.md (FixedFITSFile) **Time**: 15 min

**Description**: Write validation tests for FixedFITSFile Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid FixedFITSFile creation
2. Test source_file relationship with FITSImageFile
3. Test fixed_path validation (must end with \_fixed.fits)
4. Test corrections_applied validation (non-empty list)
5. Test created_at timestamp
6. Test frozen model (immutability)
7. Test serialization

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)

---

### T006: Implement FixedFITSFile Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T005 **Parallel**: [P]
**Contract**: data-model.md (FixedFITSFile) **Time**: 20 min

**Description**: Implement FixedFITSFile as frozen Pydantic BaseModel

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `FixedFITSFile(BaseModel)` with frozen config
2. Add fields: `source_file: FITSImageFile`, `fixed_path: Path`,
   `corrections_applied: list[str]`, `created_at: datetime`
3. Add `@field_validator` for fixed_path suffix validation
4. Add `@field_validator` for non-empty corrections list
5. Implement `exists` property to check file existence
6. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T005) pass
- [ ] Model is frozen

---

### T007: Write Pydantic model validation tests (SpatialGrid)

**Type**: test **Layer**: 1 **Dependencies**: T004 **Parallel**: [P]
**Contract**: data-model.md (SpatialGrid) **Time**: 15 min

**Description**: Write validation tests for SpatialGrid Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid SpatialGrid creation
2. Test l_coords and m_coords arrays
3. Test shape property
4. Test checksum computation (MD5)
5. Test matches() method for grid comparison
6. Test frozen model
7. Test serialization with numpy arrays

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)

---

### T008: Implement SpatialGrid Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T007 **Parallel**: [P]
**Contract**: data-model.md (SpatialGrid) **Time**: 25 min

**Description**: Implement SpatialGrid as frozen Pydantic BaseModel

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `SpatialGrid(BaseModel)` with frozen config
2. Add fields: `l_coords: np.ndarray`, `m_coords: np.ndarray`
3. Configure Pydantic to handle numpy arrays (use
   `arbitrary_types_allowed=True`)
4. Implement `shape` property
5. Implement `l_checksum` and `m_checksum` properties with MD5
6. Implement `matches(other)` method using checksums
7. Add custom serialization for numpy arrays

**Acceptance**:

- [ ] All validation tests (T007) pass
- [ ] Checksums are deterministic

---

### T009: Write Pydantic model validation tests (TimeStepGroup)

**Type**: test **Layer**: 1 **Dependencies**: T004 **Parallel**: [P]
**Contract**: data-model.md (TimeStepGroup) **Time**: 15 min

**Description**: Write validation tests for TimeStepGroup Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid TimeStepGroup creation
2. Test auto-sort files by frequency
3. Test time_step_key consistency across files
4. Test frequency_range_mhz calculation
5. Test file_count property
6. Test has_frequency_gaps detection
7. Test serialization

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)

---

### T010: Implement TimeStepGroup Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T009 **Parallel**: [P]
**Contract**: data-model.md (TimeStepGroup) **Time**: 25 min

**Description**: Implement TimeStepGroup as Pydantic BaseModel

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `TimeStepGroup(BaseModel)`
2. Add fields: `time_step_key: str`, `files: list[FITSImageFile]`,
   `frequency_range_mhz: tuple[int, int]`
3. Add `@model_validator(mode='after')` to auto-sort files by frequency
4. Add `@field_validator` to validate all files have same time_step_key
5. Implement `file_count` property
6. Implement `has_frequency_gaps` detection logic
7. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T009) pass
- [ ] Files are always sorted by frequency

---

### T011: Write Pydantic model validation tests (ZarrStore)

**Type**: test **Layer**: 1 **Dependencies**: T004 **Parallel**: [P]
**Contract**: data-model.md (ZarrStore) **Time**: 15 min

**Description**: Write validation tests for ZarrStore Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid ZarrStore creation
2. Test path validation (.zarr suffix)
3. Test dimensions and chunk_sizes
4. Test lock_file_path property
5. Test state_file_path property
6. Test is_locked() method
7. Test serialization

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)

---

### T012: Implement ZarrStore Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T011 **Parallel**: [P]
**Contract**: data-model.md (ZarrStore) **Time**: 25 min

**Description**: Implement ZarrStore as Pydantic BaseModel

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `ZarrStore(BaseModel)`
2. Add fields: `path: Path`, `dimensions: dict[str, int]`,
   `chunk_sizes: dict[str, int]`, `exists: bool`, `locked: bool = False`
3. Add `@field_validator` for .zarr suffix
4. Add `@field_validator` for valid chunk sizes (positive)
5. Implement `lock_file_path` property
6. Implement `state_file_path` property
7. Implement `is_locked()` method
8. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T011) pass
- [ ] Lock and state file paths are correct

---

### T013: Write Pydantic model validation tests (PipelineState)

**Type**: test **Layer**: 1 **Dependencies**: T004, T012 **Parallel**: [P]
**Contract**: data-model.md (PipelineState) **Time**: 20 min

**Description**: Write validation tests for PipelineState Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid PipelineState creation
2. Test PipelineStatus enum
3. Test state transitions (NOT_STARTED → IN_PROGRESS → COMPLETED)
4. Test serialization with `model_dump_json()`
5. Test deserialization with `model_validate_json()`
6. Test atomic save() with temp file + rename
7. Test load() from JSON file

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)

---

### T014: Implement PipelineState Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T013 **Parallel**: [P]
**Contract**: data-model.md (PipelineState) **Time**: 30 min

**Description**: Implement PipelineState as Pydantic BaseModel with JSON
persistence

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `PipelineStatus(str, Enum)` with states
2. Define `PipelineState(BaseModel)`
3. Add fields: `version: str`, `zarr_path: Path`, `start_time: datetime`,
   `last_update: datetime`, `status: PipelineStatus`,
   `completed_time_steps: list[str]`, `current_time_step: Optional[str]`,
   `spatial_grid_reference: Optional[dict]`, `total_files_processed: int`,
   `error_message: Optional[str]`
4. Implement `save(path: Path)` with atomic write (temp + rename)
5. Implement `load(path: Path)` classmethod using `model_validate_json()`
6. Configure datetime serialization
7. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T013) pass
- [ ] Atomic writes prevent corruption

---

### T015: Write Pydantic model validation tests (ConversionMetadata)

**Type**: test **Layer**: 1 **Dependencies**: T008, T010 **Parallel**: [P]
**Contract**: data-model.md (ConversionMetadata) **Time**: 20 min

**Description**: Write validation tests for ConversionMetadata Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid ConversionMetadata creation
2. Test add_time_step() method
3. Test set_reference_grid() method
4. Test validate_grid() with matching grids
5. Test validate_grid() with mismatched grids (should raise ValidationError)
6. Test total_files property
7. Test frequency_range_mhz property
8. Test serialization

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)

---

### T016: Implement ConversionMetadata Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T015 **Parallel**: [P]
**Contract**: data-model.md (ConversionMetadata) **Time**: 30 min

**Description**: Implement ConversionMetadata as Pydantic BaseModel

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `ConversionMetadata(BaseModel)`
2. Add fields: `discovered_time_steps: list[TimeStepGroup]`,
   `spatial_grid_reference: Optional[SpatialGrid]`, `files_processed: int`,
   `errors_encountered: list[str]`, `warnings: list[str]`,
   `start_time: datetime`
3. Implement `add_time_step(group)` method
4. Implement `set_reference_grid(grid)` method
5. Implement `validate_grid(grid)` method with ValidationError
6. Implement `total_files` property
7. Implement `frequency_range_mhz` property
8. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T015) pass
- [ ] Grid validation raises clear errors

---

### T016a: Write Pydantic model validation tests (WCSHeader)

**Type**: test **Layer**: 1 **Dependencies**: T008 **Parallel**: [P]
**Contract**: data-model.md (WCSHeader), FR-019a-f **Time**: 20 min

**Description**: Write validation tests for WCSHeader Pydantic model

**Test File**: `tests/ingest/test_models.py`

**Test Cases**:

1. Test valid WCSHeader creation
2. Test header_string validation (valid FITS WCS format)
3. Test ra_coords and dec_coords arrays (2D, matching shape)
4. Test wcs property (reconstructs astropy WCS)
5. Test shape property
6. Test to_dict() serialization
7. Test from_fits() classmethod extracts WCS from FITS file
8. Test from_fits() computes RA/Dec at pixel centers (origin=0)
9. Test frozen model (immutability)

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (model not implemented yet)

---

### T016b: Implement WCSHeader Pydantic model

**Type**: pydantic-model **Layer**: 1 **Dependencies**: T016a **Parallel**: [P]
**Contract**: data-model.md (WCSHeader), FR-019a-f **Time**: 35 min

**Description**: Implement WCSHeader as frozen Pydantic BaseModel with WCS
extraction

**File**: `src/ovro_lwa_portal/ingest/models.py`

**Implementation**:

1. Define `WCSHeader(BaseModel)` with frozen config
2. Add fields: `header_string: str`, `ra_coords: np.ndarray`,
   `dec_coords: np.ndarray`, `frame: str = "fk5"`, `equinox: str = "J2000"`
3. Configure Pydantic for numpy arrays (arbitrary_types_allowed=True)
4. Implement `wcs` property to reconstruct astropy WCS
5. Implement `shape` property
6. Implement `to_dict()` method for serialization
7. Implement `from_fits(fits_path, spatial_grid)` classmethod:
   - Open FITS file
   - Extract 2D celestial WCS
   - Compute RA/Dec at pixel centers (origin=0)
   - Format header string
8. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T016a) pass
- [ ] WCS can be reconstructed from header_string
- [ ] RA/Dec coordinates computed correctly

---

## Layer 2: Pydantic Settings (Configuration)

### T017: Write Pydantic settings validation tests (ConversionOptions)

**Type**: test **Layer**: 2 **Dependencies**: T016 **Parallel**: No
**Contract**: data-model.md (ConversionOptions) **Time**: 15 min

**Description**: Write validation tests for ConversionOptions Pydantic settings

**Test File**: `tests/ingest/test_config.py`

**Test Cases**:

1. Test valid ConversionOptions creation from parameters
2. Test frozen model (immutability)
3. Test input_dir validation (must exist)
4. Test chunk_lm validation (must be positive)
5. Test default values (zarr_name, chunk_lm, rebuild, etc.)
6. Test serialization

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (settings not implemented yet)

---

### T018: Implement ConversionOptions Pydantic settings

**Type**: pydantic-settings **Layer**: 2 **Dependencies**: T017 **Parallel**:
[P] **Contract**: data-model.md (ConversionOptions) **Time**: 20 min

**Description**: Implement ConversionOptions as frozen Pydantic BaseModel

**File**: `src/ovro_lwa_portal/ingest/config.py`

**Implementation**:

1. Define `ConversionOptions(BaseModel)` with frozen config
2. Add fields: `input_dir: Path`, `output_dir: Path`, `zarr_name: str`,
   `fixed_dir: Path`, `chunk_lm: int`, `rebuild: bool`, `use_prefect: bool`,
   `log_level: str`
3. Set defaults using Field(default=...)
4. Add `@field_validator` for input_dir existence
5. Add `@field_validator` for positive chunk_lm
6. Add `@field_validator` for valid log_level
7. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T017) pass
- [ ] Model is frozen

---

### T019: Write Pydantic settings tests with environment variables

**Type**: test **Layer**: 2 **Dependencies**: T018 **Parallel**: [P]
**Contract**: research.md (pydantic-settings) **Time**: 15 min

**Description**: Write tests for AppConfig loading from environment variables

**Test File**: `tests/ingest/test_config.py`

**Test Cases**:

1. Test AppConfig loading with default values
2. Test loading from environment variables (OVRO_LWA_LOG_LEVEL, etc.)
3. Test precedence: env vars > defaults
4. Test validation of environment variable values
5. Test serialization

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (AppConfig not implemented yet)

---

### T020: Implement AppConfig Pydantic settings

**Type**: pydantic-settings **Layer**: 2 **Dependencies**: T019 **Parallel**:
[P] **Contract**: data-model.md (AppConfig from plan.md) **Time**: 25 min

**Description**: Implement AppConfig using pydantic-settings BaseSettings

**File**: `src/ovro_lwa_portal/ingest/config.py`

**Implementation**:

1. Define `AppConfig(BaseSettings)`
2. Add fields: `log_level: str`, `log_file: Optional[Path]`,
   `enable_prefect: bool`, `lock_timeout: int`, `state_version: str`
3. Set defaults using Field(default=...)
4. Configure `SettingsConfigDict` with env*prefix="OVRO_LWA*"
5. Add `@field_validator` for valid log_level
6. Add Field descriptions

**Acceptance**:

- [ ] All validation tests (T019) pass
- [ ] Environment variables are loaded correctly

---

## Layer 3: Core Utilities

### T021: Write file locking tests

**Type**: test **Layer**: 3 **Dependencies**: T012 **Parallel**: No
**Contract**: research.md (portalocker) **Time**: 20 min

**Description**: Write tests for file locking utilities

**Test File**: `tests/ingest/test_locking.py`

**Test Cases**:

1. Test lock acquisition succeeds when no lock exists
2. Test lock acquisition fails when lock already held
3. Test LockAcquisitionError with PID information
4. Test context manager auto-releases lock
5. Test lock timeout behavior
6. Test concurrent access detection (multiprocessing)

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (locking not implemented yet)

---

### T022: Implement file locking utilities

**Type**: implement **Layer**: 3 **Dependencies**: T021 **Parallel**: [P]
**Contract**: FR-019 **Time**: 30 min

**Description**: Implement cross-platform file locking with portalocker

**File**: `src/ovro_lwa_portal/ingest/locking.py`

**Implementation**:

1. Define `LockAcquisitionError(Exception)` with PID info
2. Implement `acquire_lock(lock_path: Path, timeout: int)` context manager
3. Use portalocker for cross-platform locking
4. Write PID to lock file
5. Non-blocking lock attempts
6. Auto-release on context exit
7. Add logging for lock operations

**Acceptance**:

- [ ] All validation tests (T021) pass
- [ ] Works on macOS and Linux

---

### T023: Write state management tests

**Type**: test **Layer**: 3 **Dependencies**: T014 **Parallel**: [P]
**Contract**: data-model.md (PipelineState) **Time**: 20 min

**Description**: Write tests for pipeline state persistence

**Test File**: `tests/ingest/test_state.py`

**Test Cases**:

1. Test state file creation with atomic write
2. Test state loading from JSON
3. Test state updates (completed_time_steps)
4. Test state transitions (status changes)
5. Test corruption recovery (temp file cleanup)
6. Test state file locking during writes

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (state management not implemented yet)

---

### T024: Implement state management utilities

**Type**: implement **Layer**: 3 **Dependencies**: T023 **Parallel**: [P]
**Contract**: FR-027, NFR-002 **Time**: 25 min

**Description**: Implement pipeline state persistence utilities

**File**: `src/ovro_lwa_portal/ingest/state.py`

**Implementation**:

1. Implement `save_state(state: PipelineState, path: Path)` with atomic write
2. Implement `load_state(path: Path) -> PipelineState`
3. Handle missing files gracefully
4. Validate state schema version
5. Implement state migration if needed
6. Add logging for state operations

**Acceptance**:

- [ ] All validation tests (T023) pass
- [ ] Atomic writes prevent corruption

---

### T025: Write logging configuration tests

**Type**: test **Layer**: 3 **Dependencies**: T020 **Parallel**: [P]
**Contract**: FR-028, FR-029 **Time**: 15 min

**Description**: Write tests for configurable logging

**Test File**: `tests/ingest/test_logging.py`

**Test Cases**:

1. Test logging setup with different levels (DEBUG, INFO, ERROR)
2. Test log file output
3. Test console output formatting with rich
4. Test structured logging (JSON format)
5. Test log filtering by module

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (logging not implemented yet)

---

### T026: Implement logging configuration

**Type**: implement **Layer**: 3 **Dependencies**: T025 **Parallel**: [P]
**Contract**: FR-028, FR-029 **Time**: 20 min

**Description**: Implement configurable logging with rich formatting

**File**: `src/ovro_lwa_portal/ingest/logging_config.py`

**Implementation**:

1. Define `configure_logging(level: str, log_file: Optional[Path])`
2. Set up console handler with rich formatting
3. Set up file handler if log_file provided
4. Configure log levels (DEBUG, INFO, ERROR)
5. Add structured logging for verbose mode
6. Integrate with existing logger in fits_to_zarr_xradio.py

**Acceptance**:

- [ ] All validation tests (T025) pass
- [ ] Logging is configurable via CLI flags

---

## Layer 4: Conversion Logic (Core Functions)

### T027: Write discovery contract tests

**Type**: test **Layer**: 4 **Dependencies**: T004, T010 **Parallel**: No
**Contract**: contracts/discovery_api.md **Time**: 25 min

**Description**: Write contract tests for FITS discovery functions

**Test File**: `tests/ingest/test_discovery.py`

**Test Cases**:

1. Test discover_fits_files() with valid FITS files
2. Test pattern matching (YYYYMMDD*HHMMSS*\_MHz\_\_-I-image.fits)
3. Test handling of \_fixed.fits files
4. Test error on empty directory (ValueError)
5. Test group_by_time_step() creates correct groups
6. Test files sorted by frequency within groups
7. Test groups sorted by time_step_key
8. Test frequency gap detection

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (functions not implemented yet)

---

### T028: Implement FITS discovery functions

**Type**: implement **Layer**: 4 **Dependencies**: T027 **Parallel**: No
**Contract**: contracts/discovery_api.md **Time**: 35 min

**Description**: Implement discover_fits_files() and group_by_time_step()

**File**: `src/ovro_lwa_portal/ingest/discovery.py`

**Implementation**:

1. Implement `discover_fits_files(input_dir: Path) -> list[FITSImageFile]`
2. Use regex to match OVRO-LWA naming pattern
3. Extract metadata from filename
4. Return list of FITSImageFile Pydantic models
5. Raise ValueError if no files found
6. Implement
   `group_by_time_step(files: list[FITSImageFile]) -> list[TimeStepGroup]`
7. Group by time_step_key
8. Sort files by frequency within groups
9. Sort groups by time_step_key
10. Detect and log frequency gaps
11. Return list of TimeStepGroup Pydantic models

**Acceptance**:

- [ ] All contract tests (T027) pass
- [ ] Functions use Pydantic models

---

### T029: Write validation contract tests

**Type**: test **Layer**: 4 **Dependencies**: T008, T016 **Parallel**: [P]
**Contract**: FR-009 **Time**: 20 min

**Description**: Write contract tests for spatial grid validation

**Test File**: `tests/ingest/test_validation.py`

**Test Cases**:

1. Test validate_spatial_grid() with matching grids
2. Test validate_spatial_grid() with mismatched grids (raises ValidationError)
3. Test error message includes grid details
4. Test first grid becomes reference
5. Test subsequent grids validated against reference

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (function not implemented yet)

---

### T030: Implement spatial grid validation

**Type**: implement **Layer**: 4 **Dependencies**: T029 **Parallel**: [P]
**Contract**: FR-009 **Time**: 25 min

**Description**: Implement validate_spatial_grid() function

**File**: `src/ovro_lwa_portal/ingest/validation.py`

**Implementation**:

1. Implement
   `validate_spatial_grid(grid: SpatialGrid, reference: Optional[SpatialGrid]) -> SpatialGrid`
2. If no reference, set grid as reference
3. If reference exists, use SpatialGrid.matches() method
4. Raise ValidationError with details if mismatch
5. Return reference grid

**Acceptance**:

- [ ] All contract tests (T029) pass
- [ ] Error messages are actionable

---

### T031: Write FITS header correction contract tests

**Type**: test **Layer**: 4 **Dependencies**: T004, T006 **Parallel**: [P]
**Contract**: contracts/core_api.md (fix_fits_headers) **Time**: 25 min

**Description**: Write contract tests for fix_fits_headers()

**Test File**: `tests/ingest/test_core.py`

**Test Cases**:

1. Test fix_fits_headers() creates \_fixed.fits file
2. Test BSCALE/BZERO applied to data
3. Test mandatory keywords added (RESTFREQ, SPECSYS, TIMESYS, RADESYS)
4. Test PC matrix identity set
5. Test beam parameters added (BMAJ, BMIN)
6. Test BUNIT set to Jy/beam
7. Test reuse of existing fixed file (FR-013)
8. Test returns FixedFITSFile Pydantic model

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (function not implemented yet)

---

### T032: Extract and refactor fix_fits_headers() from existing code

**Type**: refactor **Layer**: 4 **Dependencies**: T031 **Parallel**: [P]
**Contract**: contracts/core_api.md (fix_fits_headers) **Time**: 35 min

**Description**: Extract \_fix_headers() logic from fits_to_zarr_xradio.py

**File**: `src/ovro_lwa_portal/ingest/core.py`

**Implementation**:

1. Copy \_fix_headers() logic from fits_to_zarr_xradio.py
2. Rename to
   `fix_fits_headers(fits_file: FITSImageFile, output_dir: Path) -> FixedFITSFile`
3. Update signature to use Pydantic models
4. Check if fixed file already exists (FR-013)
5. Apply all corrections from FR-010, FR-011
6. Return FixedFITSFile Pydantic model
7. Add logging for corrections applied

**Acceptance**:

- [ ] All contract tests (T031) pass
- [ ] Backward compatible with existing code

---

### T033: Write load_fits_for_zarr() contract tests

**Type**: test **Layer**: 4 **Dependencies**: T006, T016b **Parallel**: [P]
**Contract**: contracts/core_api.md (load_fits_for_zarr), FR-019a-f **Time**: 25
min

**Description**: Write contract tests for load_fits_for_zarr() with WCS support

**Test File**: `tests/ingest/test_core.py`

**Test Cases**:

1. Test load_fits_for_zarr() returns tuple (xarray Dataset, WCSHeader)
2. Test uses xradio.read_image() with do_sky_coords=False
3. Test computes and attaches RA/Dec coordinates (2D, dims=(m,l))
4. Test RA/Dec coordinates at pixel centers (origin=0)
5. Test WCS header preserved in 4 redundant locations:
   - Dataset global attrs
   - 0-D variable `wcs_header_str` (np.bytes\_)
   - Per-variable attrs
   - Coordinate attrs (right_ascension, declination)
6. Test LM chunking applied
7. Test WCSHeader object returned
8. Test accepts FixedFITSFile Pydantic model

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (function not implemented yet)
- [ ] WCS preservation tests included

---

### T034: Extract and refactor load_fits_for_zarr() with WCS from existing code

**Type**: refactor **Layer**: 4 **Dependencies**: T033 **Parallel**: [P]
**Contract**: contracts/core_api.md (load_fits_for_zarr), FR-019a-f **Time**: 40
min

**Description**: Extract \_load_for_combine() logic from fits_to_zarr_xradio.py
with WCS handling

**File**: `src/ovro_lwa_portal/ingest/core.py`

**Implementation**:

1. Copy \_load_for_combine() logic from fits_to_zarr_xradio.py
2. Rename to
   `load_fits_for_zarr(fixed_file: FixedFITSFile, chunk_lm: int = 1024) -> tuple[xr.Dataset, WCSHeader]`
3. Update signature to use Pydantic models
4. Use xradio.read_image() with do_sky_coords=False
5. Extract 2D celestial WCS from FITS header
6. Compute RA/Dec at pixel centers (origin=0) using astropy WCS
7. Attach 2D coordinates: right_ascension (m,l), declination (m,l)
8. Preserve WCS header redundantly in 4 locations:
   - xds.attrs['fits_wcs_header']
   - 0-D variable wcs*header_str (use np.bytes* for NumPy 2.0)
   - Per-variable attrs
   - Coordinate attrs
9. Create WCSHeader Pydantic model from extracted WCS
10. Apply LM chunking
11. Add logging

**Acceptance**:

- [ ] All contract tests (T033) pass
- [ ] WCS coordinates attached correctly
- [ ] WCS header preserved in all 4 locations
- [ ] Returns tuple (Dataset, WCSHeader)

---

### T035: Write combine_frequency_subbands() contract tests

**Type**: test **Layer**: 4 **Dependencies**: T010, T016b **Parallel**: [P]
**Contract**: contracts/core_api.md (combine_frequency_subbands), FR-019b-c
**Time**: 30 min

**Description**: Write contract tests for combine_frequency_subbands() with WCS
preservation

**Test File**: `tests/ingest/test_core.py`

**Test Cases**:

1. Test combine_frequency_subbands() returns tuple (Dataset, SpatialGrid,
   WCSHeader)
2. Test files sorted by frequency (FR-006)
3. Test frequency dimension created
4. Test returns SpatialGrid for validation
5. Test returns WCSHeader from first file
6. Test WCS coordinates preserved through combine/concat
7. Test WCS header remains in all 4 locations after combine
8. Test warnings for frequency gaps
9. Test accepts TimeStepGroup Pydantic model

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (function not implemented yet)
- [ ] WCS preservation tests included

---

### T036: Extract and refactor combine_frequency_subbands() with WCS from existing code

**Type**: refactor **Layer**: 4 **Dependencies**: T035 **Parallel**: [P]
**Contract**: contracts/core_api.md (combine_frequency_subbands), FR-019b-c
**Time**: 40 min

**Description**: Extract combine logic from fits_to_zarr_xradio.py with WCS
preservation

**File**: `src/ovro_lwa_portal/ingest/core.py`

**Implementation**:

1. Extract frequency subband combining logic from fits_to_zarr_xradio.py
2. Define
   `combine_frequency_subbands(time_step: TimeStepGroup, fixed_dir: Path, chunk_lm: int) -> tuple[xr.Dataset, SpatialGrid, WCSHeader]`
3. Update signature to use Pydantic models
4. Sort files by frequency (FR-006)
5. Load each file with load_fits_for_zarr() (returns Dataset + WCSHeader)
6. Extract WCSHeader from first file
7. Combine datasets with xr.combine_by_coords() or xr.concat()
8. Verify WCS coordinates preserved through combine operation
9. Verify WCS header remains in all 4 redundant locations
10. Extract SpatialGrid from first file
11. Detect and log frequency gaps
12. Return combined Dataset, SpatialGrid, and WCSHeader

**Acceptance**:

- [ ] All contract tests (T035) pass
- [ ] Deterministic frequency ordering
- [ ] WCS coordinates preserved through combine
- [ ] Returns tuple with WCSHeader

---

### T037: Write write_zarr_store() contract tests

**Type**: test **Layer**: 4 **Dependencies**: T012, T022 **Parallel**: [P]
**Contract**: contracts/core_api.md (write_zarr_store) **Time**: 25 min

**Description**: Write contract tests for write_zarr_store()

**Test File**: `tests/ingest/test_core.py`

**Test Cases**:

1. Test write_zarr_store() creates new Zarr store
2. Test append mode combines with existing store
3. Test atomic writes with temp + rename (FR-017)
4. Test lock acquisition before write (FR-019)
5. Test LockAcquisitionError if locked
6. Test uses xradio.write_image() (FR-014)
7. Test accepts ZarrStore Pydantic model

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (function not implemented yet)

---

### T038: Implement write_zarr_store() function

**Type**: implement **Layer**: 4 **Dependencies**: T037 **Parallel**: [P]
**Contract**: contracts/core_api.md (write_zarr_store) **Time**: 40 min

**Description**: Implement safe Zarr writing with locking

**File**: `src/ovro_lwa_portal/ingest/core.py`

**Implementation**:

1. Define
   `write_zarr_store(dataset: xr.Dataset, zarr_store: ZarrStore, append: bool = False) -> None`
2. Acquire lock on zarr_store.lock_file_path
3. If append=False: Direct write with xradio.write_image()
4. If append=True:
   - Load existing Zarr (lazy)
   - Combine with new dataset
   - Write to temp path
   - Atomic rename
5. Release lock in finally block
6. Add logging for write operations

**Acceptance**:

- [ ] All contract tests (T037) pass
- [ ] Atomic writes prevent corruption
- [ ] Lock prevents concurrent access

---

### T039: Write convert_fits_to_zarr() contract tests

**Type**: test **Layer**: 4 **Dependencies**: T018, T016, T028, T032, T036, T038
**Parallel**: No **Contract**: contracts/core_api.md (convert_fits_to_zarr)
**Time**: 35 min

**Description**: Write contract tests for main conversion function with WCS
verification

**Test File**: `tests/ingest/test_core.py`

**Test Cases**:

1. Test convert_fits_to_zarr() orchestrates full conversion
2. Test returns zarr_path and ConversionMetadata
3. Test framework-independent (no Prefect dependency)
4. Test ValidationError on grid mismatch (FR-009)
5. Test LockAcquisitionError if locked (FR-019)
6. Test deterministic output (NFR-003)
7. Test accepts ConversionOptions Pydantic model
8. Test WCS coordinates present in final Zarr output
9. Test WCS header preserved throughout pipeline

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (function not implemented yet)
- [ ] WCS verification included

---

### T040: Implement convert_fits_to_zarr() main function

**Type**: implement **Layer**: 4 **Dependencies**: T039 **Parallel**: No
**Contract**: contracts/core_api.md (convert_fits_to_zarr) **Time**: 50 min

**Description**: Implement main conversion orchestration function with WCS
handling

**File**: `src/ovro_lwa_portal/ingest/core.py`

**Implementation**:

1. Define
   `convert_fits_to_zarr(options: ConversionOptions, metadata: Optional[ConversionMetadata] = None) -> tuple[Path, ConversionMetadata]`
2. Initialize ConversionMetadata if None
3. Call discover_fits_files()
4. Call group_by_time_step()
5. For each TimeStepGroup:
   - Call fix_fits_headers() for unfixed files
   - Call combine_frequency_subbands() (returns Dataset, SpatialGrid, WCSHeader)
   - Call validate_spatial_grid()
   - Verify WCS coordinates present in Dataset
   - Call write_zarr_store()
   - Update ConversionMetadata
6. Return zarr_path and ConversionMetadata
7. Handle all errors with actionable messages
8. Add logging for each step (include WCS extraction)

**Acceptance**:

- [ ] All contract tests (T039) pass
- [ ] Framework-independent (FR-021)
- [ ] All core functions integrated
- [ ] WCS coordinates preserved in output

---

### T041: Write backward compatibility test for fits_to_zarr_xradio.py

**Type**: test **Layer**: 4 **Dependencies**: T040 **Parallel**: No
**Contract**: Gate 10 (plan.md) **Time**: 25 min

**Description**: Ensure existing API still works after refactoring with WCS
preservation

**Test File**: `tests/ingest/test_backward_compat.py`

**Test Cases**:

1. Test convert_fits_dir_to_zarr() still exists
2. Test same inputs produce same outputs
3. Test existing tests still pass
4. Test WCS coordinates present in output Zarr (new requirement)
5. Test WCS reconstruction works with backward compatible API

**Acceptance**:

- [ ] All tests written
- [ ] Tests fail (wrapper not implemented yet)
- [ ] WCS verification included

---

### T042: Create backward compatibility wrapper

**Type**: refactor **Layer**: 4 **Dependencies**: T041 **Parallel**: No
**Contract**: Gate 10 (plan.md) **Time**: 25 min

**Description**: Update fits_to_zarr_xradio.py to call new modular functions
with WCS support

**File**: `src/ovro_lwa_portal/fits_to_zarr_xradio.py`

**Implementation**:

1. Keep convert_fits_dir_to_zarr() signature unchanged
2. Map old parameters to ConversionOptions Pydantic model
3. Call convert_fits_to_zarr() from core.py
4. Return same outputs as before
5. Ensure WCS coordinates are preserved (existing behavior maintained)
6. Add note in docstring about WCS coordinate support

**Acceptance**:

- [ ] All backward compatibility tests (T041) pass
- [ ] Existing tests pass
- [ ] No breaking changes
- [ ] WCS coordinates preserved

---

## Layer 5: CLI Interface

### T043: Write CLI contract tests (help and version)

**Type**: test **Layer**: 5 **Dependencies**: T001, T040 **Parallel**: No
**Contract**: contracts/cli_api.md **Time**: 15 min

**Description**: Write CLI contract tests for help and version commands

**Test File**: `tests/ingest/test_cli.py`

**Test Cases**:

1. Test `ovro-lwa-ingest --help` displays usage
2. Test `ovro-lwa-ingest version` displays version
3. Test exit codes (0=success, non-zero=error)

**Acceptance**:

- [ ] All tests written (using subprocess)
- [ ] Tests fail (CLI not implemented yet)

---

### T044: Write CLI contract tests (convert command)

**Type**: test **Layer**: 5 **Dependencies**: T001, T040 **Parallel**: No
**Contract**: contracts/cli_api.md **Time**: 30 min

**Description**: Write CLI contract tests for main convert command

**Test File**: `tests/ingest/test_cli.py`

**Test Cases**:

1. Test basic conversion with required arguments
2. Test all optional flags (--zarr-name, --fixed-dir, --chunk-lm, etc.)
3. Test --rebuild flag
4. Test --verbose and --quiet flags
5. Test --log-file output
6. Test error on missing required arguments
7. Test error on invalid paths
8. Test progress indicators appear in output
9. Test summary statistics in output

**Acceptance**:

- [ ] All tests written (using subprocess)
- [ ] Tests fail (CLI not implemented yet)

---

### T045: Implement basic CLI structure with Typer

**Type**: implement **Layer**: 5 **Dependencies**: T044 **Parallel**: No
**Contract**: contracts/cli_api.md **Time**: 30 min

**Description**: Create Typer CLI application structure

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. Define `app = typer.Typer()` with help text
2. Implement `@app.command() def convert(...)` with all parameters
3. Use Typer's Path validation (exists, readable, writable)
4. Implement `@app.command() def version()`
5. Add docstrings with examples
6. Configure typer with add_completion=True

**Acceptance**:

- [ ] CLI help tests (T043) pass
- [ ] CLI entry point registered
- [ ] `ovro-lwa-ingest --help` works

---

### T046: Implement logging configuration in CLI

**Type**: implement **Layer**: 5 **Dependencies**: T026, T045 **Parallel**: No
**Contract**: FR-028, FR-029 **Time**: 20 min

**Description**: Configure logging based on CLI flags

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. In convert() command, configure logging before any operations
2. Map --verbose to DEBUG level
3. Map --quiet to ERROR level
4. Default to INFO level
5. Configure log_file if provided
6. Use rich console for formatted output

**Acceptance**:

- [ ] Logging flags work correctly
- [ ] Verbose mode shows detailed logs
- [ ] Quiet mode shows only errors

---

### T047: Implement resume/rebuild prompt in CLI

**Type**: implement **Layer**: 5 **Dependencies**: T024, T045 **Parallel**: No
**Contract**: FR-027, NFR-002 **Time**: 30 min

**Description**: Implement interactive resume/rebuild prompt

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. Check for incomplete state (ZarrStore.has_incomplete_state())
2. Load PipelineState if exists
3. Use rich.Prompt.ask() for interactive prompt
4. Show state information (last time step, completed count)
5. Offer resume or rebuild options
6. Handle user cancellation (raise typer.Abort)
7. Skip prompt if --rebuild flag set

**Acceptance**:

- [ ] Interactive prompt appears on incomplete state
- [ ] Resume continues from last checkpoint
- [ ] Rebuild clears state and starts over

---

### T048: Implement progress indicators in CLI

**Type**: implement **Layer**: 5 **Dependencies**: T045 **Parallel**: No
**Contract**: NFR-001 **Time**: 25 min

**Description**: Add rich progress bars to CLI

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. Use rich.Progress for long-running operations
2. Show progress bar for time step processing
3. Update description with current time_step_key
4. Show percentage and ETA
5. Handle terminal resize
6. Suppress progress if --quiet flag

**Acceptance**:

- [ ] Progress bar appears during conversion
- [ ] Shows current time step
- [ ] Updates incrementally

---

### T049: Implement summary statistics in CLI

**Type**: implement **Layer**: 5 **Dependencies**: T045 **Parallel**: No
**Contract**: FR-024 **Time**: 20 min

**Description**: Display summary statistics after conversion

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. After conversion completes, print summary
2. Show total files processed
3. Show total time steps
4. Show frequency range
5. Show output path
6. Show total time elapsed
7. Use rich formatting for visual appeal

**Acceptance**:

- [ ] Summary displays after successful conversion
- [ ] All statistics are accurate

---

### T050: Implement error formatting in CLI

**Type**: implement **Layer**: 5 **Dependencies**: T045 **Parallel**: No
**Contract**: NFR-005 **Time**: 25 min

**Description**: Add actionable error messages with suggested fixes

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. Catch LockAcquisitionError and format with suggestions
2. Catch ValidationError and format with grid details
3. Catch FileNotFoundError and format with pattern example
4. Use rich for color-coded errors (red for errors, yellow for warnings)
5. Show suggested actions for common errors
6. Suppress stack traces for user-facing errors

**Acceptance**:

- [ ] Error messages are actionable
- [ ] Suggestions help users resolve issues
- [ ] No Python jargon in user-facing errors

---

### T051: Implement CLI to core integration

**Type**: implement **Layer**: 5 **Dependencies**: T040, T050 **Parallel**: No
**Contract**: contracts/cli_api.md **Time**: 30 min

**Description**: Connect CLI to core conversion function

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. Map CLI arguments to ConversionOptions Pydantic model
2. Call convert_fits_to_zarr() from core.py
3. Handle keyboard interrupt (SIGINT) gracefully
4. Update progress bar during conversion
5. Display summary statistics from ConversionMetadata
6. Handle all exceptions with formatted errors

**Acceptance**:

- [ ] All CLI contract tests (T044) pass
- [ ] Full conversion works end-to-end
- [ ] Errors are handled gracefully

---

## Layer 6: Optional Prefect Integration

### T052: Write Prefect flow contract tests

**Type**: test **Layer**: 6 **Dependencies**: T040 **Parallel**: No
**Contract**: FR-022 (optional) **Time**: 25 min

**Description**: Write contract tests for Prefect flows

**Test File**: `tests/ingest/test_prefect_flows.py`

**Test Cases**:

1. Test fits_to_zarr_flow() wraps core conversion
2. Test flow returns same outputs as core function
3. Test flow tasks are logged in Prefect
4. Test flow accepts ConversionOptions Pydantic model
5. Test flow works without Prefect installed (graceful degradation)

**Acceptance**:

- [ ] All tests written
- [ ] Tests skip if Prefect not installed
- [ ] Tests fail (flows not implemented yet)

---

### T053: Implement Prefect flows (optional)

**Type**: implement **Layer**: 6 **Dependencies**: T052 **Parallel**: No
**Contract**: FR-022 (optional) **Time**: 35 min

**Description**: Implement Prefect orchestration layer

**File**: `src/ovro_lwa_portal/ingest/prefect_flows.py`

**Implementation**:

1. Define `@flow` for fits_to_zarr_flow()
2. Define `@task` for each core function (discover, fix, combine, write)
3. Pass Pydantic models between tasks
4. Wrap core.convert_fits_to_zarr()
5. Add Prefect-specific logging
6. Handle Prefect not installed gracefully

**Acceptance**:

- [ ] All Prefect tests (T052) pass
- [ ] Flow shows up in Prefect UI
- [ ] Core logic unchanged

---

### T054: Integrate Prefect with CLI (optional)

**Type**: implement **Layer**: 6 **Dependencies**: T051, T053 **Parallel**: No
**Contract**: FR-022 (optional) **Time**: 20 min

**Description**: Add --use-prefect flag support to CLI

**File**: `src/ovro_lwa_portal/ingest/cli.py`

**Implementation**:

1. Check --use-prefect flag
2. If True, import and call fits_to_zarr_flow()
3. If False, call core.convert_fits_to_zarr() directly
4. Show warning if Prefect not installed
5. Pass ConversionOptions to flow

**Acceptance**:

- [ ] --use-prefect flag works
- [ ] Graceful degradation if Prefect missing
- [ ] Flow appears in Prefect UI

---

### T055: Write Prefect UI test (optional, manual)

**Type**: integration **Layer**: 6 **Dependencies**: T054 **Parallel**: No
**Contract**: FR-022 (optional) **Time**: 15 min

**Description**: Manual test of Prefect UI integration

**Test Steps**:

1. Start Prefect server: `prefect server start`
2. Run conversion with --use-prefect
3. Open Prefect UI
4. Verify flow appears
5. Verify task status
6. Verify logs are captured

**Acceptance**:

- [ ] Flow visible in Prefect UI
- [ ] Tasks show correct status
- [ ] Logs are captured

---

## Layer 7: Integration & Acceptance Testing

### T056: Write integration test (Scenario 1: Basic Conversion)

**Type**: integration **Layer**: 7 **Dependencies**: T051 **Parallel**: No
**Contract**: quickstart.md (Scenario 1) **Time**: 30 min

**Description**: End-to-end test for basic FITS to Zarr conversion

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Test full conversion pipeline with test FITS files
2. Verify Zarr store structure
3. Verify dimensions (time, frequency, l, m)
4. Verify data integrity with xarray
5. Verify summary statistics

**Acceptance**:

- [ ] Test written
- [ ] Test passes with real data
- [ ] Covers Scenario 1 from spec

---

### T057: Write integration test (Scenario 2: Append)

**Type**: integration **Layer**: 7 **Dependencies**: T051 **Parallel**: [P]
**Contract**: quickstart.md (Scenario 2) **Time**: 25 min

**Description**: End-to-end test for appending to existing Zarr

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Create initial Zarr store
2. Append new time steps
3. Verify time dimension increased
4. Verify existing data preserved
5. Verify no corruption

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers Scenario 2 from spec

---

### T058: Write integration test (Scenario 3: Header Correction)

**Type**: integration **Layer**: 7 **Dependencies**: T032 **Parallel**: [P]
**Contract**: quickstart.md (Scenario 3) **Time**: 25 min

**Description**: End-to-end test for automatic header correction

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Run conversion with uncorrected FITS files
2. Verify \_fixed.fits files created
3. Verify header corrections applied (RESTFREQ, SPECSYS, etc.)
4. Verify BSCALE/BZERO materialized
5. Verify fixed files reused on second run

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers Scenario 3 from spec

---

### T059: Write integration test (Scenario 4: Progress Indicators)

**Type**: integration **Layer**: 7 **Dependencies**: T048 **Parallel**: [P]
**Contract**: quickstart.md (Scenario 4) **Time**: 20 min

**Description**: Test progress indicators and verbose logging

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Run conversion with --verbose
2. Capture stdout
3. Verify progress indicators appear
4. Verify verbose logs include per-file details
5. Verify summary statistics displayed

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers Scenario 4 from spec

---

### T060: Write integration test (Scenario 5: Grid Mismatch)

**Type**: integration **Layer**: 7 **Dependencies**: T030, T050 **Parallel**:
[P] **Contract**: quickstart.md (Scenario 5) **Time**: 25 min

**Description**: Test grid mismatch detection and error handling

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Create synthetic FITS files with mismatched grids
2. Run conversion
3. Verify ValidationError raised
4. Verify error message includes grid details
5. Verify suggested actions displayed

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers Scenario 5 from spec

---

### T061: Write integration test (Scenario 6: Rebuild)

**Type**: integration **Layer**: 7 **Dependencies**: T051 **Parallel**: [P]
**Contract**: quickstart.md (Scenario 6) **Time**: 20 min

**Description**: Test rebuild from scratch with --rebuild flag

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Create initial Zarr store
2. Run conversion with --rebuild
3. Verify existing Zarr replaced
4. Verify no state file
5. Verify data is fresh

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers Scenario 6 from spec

---

### T062: Write integration test (Scenario 7: Help Documentation)

**Type**: integration **Layer**: 7 **Dependencies**: T043 **Parallel**: [P]
**Contract**: quickstart.md (Scenario 7) **Time**: 15 min

**Description**: Test CLI help and documentation

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Run ovro-lwa-ingest --help
2. Verify usage instructions clear
3. Verify all options documented
4. Verify examples included

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers Scenario 7 from spec

---

### T062a: Write integration test (Scenario 8: WCS Coordinate Preservation)

**Type**: integration **Layer**: 7 **Dependencies**: T034, T051 **Parallel**:
[P] **Contract**: quickstart.md (Scenario 8), FR-019a-f **Time**: 30 min

**Description**: Test WCS coordinate preservation and reconstruction

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Run full conversion with test FITS files
2. Verify RA/Dec coordinates exist in Zarr store
3. Verify coordinate dimensions (m, l) and units (deg)
4. Verify coordinate attributes (frame=fk5, equinox=J2000)
5. Verify WCS header in all 4 redundant locations:
   - Dataset global attrs
   - 0-D variable wcs_header_str
   - Per-variable attrs
   - Coordinate attrs
6. Test WCS reconstruction with get_wcs_from_zarr()
7. Test WCS plotting with matplotlib WCSAxes
8. Verify FITS-free analysis (no original FITS required)

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers Scenario 8 from spec
- [ ] All 4 WCS storage locations verified
- [ ] WCS reconstruction works

---

### T063: Write integration test (Edge Case: No FITS Files)

**Type**: integration **Layer**: 7 **Dependencies**: T028, T050 **Parallel**:
[P] **Contract**: quickstart.md (Edge Case) **Time**: 15 min

**Description**: Test error handling for empty input directory

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Create empty directory
2. Run conversion
3. Verify ValueError raised
4. Verify error message actionable
5. Verify pattern example shown

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers edge case from spec

---

### T064: Write integration test (Edge Case: Concurrent Access)

**Type**: integration **Layer**: 7 **Dependencies**: T022, T050 **Parallel**:
[P] **Contract**: quickstart.md (Edge Case) **Time**: 25 min

**Description**: Test concurrent write detection with file locking

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Start conversion in subprocess
2. Attempt second conversion to same output
3. Verify LockAcquisitionError raised
4. Verify PID shown in error
5. Verify suggested actions displayed

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers edge case from spec

---

### T065: Write integration test (Edge Case: Interrupted Pipeline)

**Type**: integration **Layer**: 7 **Dependencies**: T047 **Parallel**: [P]
**Contract**: quickstart.md (Edge Case) **Time**: 30 min

**Description**: Test resume/rebuild prompt after interruption

**Test File**: `tests/ingest/test_integration.py`

**Test Cases**:

1. Start conversion
2. Simulate interruption (kill process)
3. Restart conversion
4. Verify resume prompt appears
5. Test resume option
6. Test rebuild option

**Acceptance**:

- [ ] Test written
- [ ] Test passes
- [ ] Covers edge case from spec

---

### T066: Write performance test (Memory Usage)

**Type**: integration **Layer**: 7 **Dependencies**: T040 **Parallel**: [P]
**Contract**: NFR-004 **Time**: 30 min

**Description**: Test memory usage stays under 4GB

**Test File**: `tests/ingest/test_performance.py`

**Test Cases**:

1. Profile memory usage during conversion
2. Verify peak memory ≤4GB
3. Verify no memory leaks
4. Test with large datasets (multiple time steps)

**Acceptance**:

- [ ] Test written
- [ ] Memory usage under limit
- [ ] Documented in quickstart.md

---

## Layer 8: Documentation & Finalization

### T067: Write docstrings for all public functions

**Type**: docs **Layer**: 8 **Dependencies**: T040, T051 **Parallel**: No
**Contract**: Gate 8 (plan.md) **Time**: 45 min

**Description**: Add NumPy-style docstrings to all public functions

**Files**:

- `src/ovro_lwa_portal/ingest/core.py`
- `src/ovro_lwa_portal/ingest/discovery.py`
- `src/ovro_lwa_portal/ingest/validation.py`
- `src/ovro_lwa_portal/ingest/locking.py`
- `src/ovro_lwa_portal/ingest/state.py`
- `src/ovro_lwa_portal/ingest/cli.py`

**Requirements**:

1. NumPy docstring format
2. Include Parameters, Returns, Raises, Notes, Examples sections
3. Add module-level docstrings
4. Document all Pydantic models

**Acceptance**:

- [ ] All public functions have docstrings
- [ ] Examples included
- [ ] ruff docstring checks pass

---

### T068: Write docstrings for all Pydantic models

**Type**: docs **Layer**: 8 **Dependencies**: T016 **Parallel**: [P]
**Contract**: Gate 8 (plan.md) **Time**: 30 min

**Description**: Add docstrings and field descriptions to Pydantic models

**Files**:

- `src/ovro_lwa_portal/ingest/models.py`
- `src/ovro_lwa_portal/ingest/config.py`

**Requirements**:

1. Class docstrings explaining purpose
2. Field descriptions via `Field(description="...")`
3. Examples via `Field(examples=[...])`
4. Document validators

**Acceptance**:

- [ ] All models documented
- [ ] Field descriptions clear
- [ ] Examples helpful

---

### T069: Update package **init**.py with exports

**Type**: docs **Layer**: 8 **Dependencies**: T040, T051 **Parallel**: [P]
**Time**: 15 min

**Description**: Export public API from package **init**.py

**File**: `src/ovro_lwa_portal/ingest/__init__.py`

**Requirements**:

1. Export all public Pydantic models
2. Export core conversion function
3. Export ConversionOptions
4. Document public API
5. Use `__all__` for explicit exports

**Acceptance**:

- [ ] Public API exported
- [ ] Import paths clean
- [ ] `from ovro_lwa_portal.ingest import ...` works

---

### T070: Create README for ingest package

**Type**: docs **Layer**: 8 **Dependencies**: T069 **Parallel**: [P] **Time**:
30 min

**Description**: Create comprehensive README for ingest package

**File**: `src/ovro_lwa_portal/ingest/README.md`

**Contents**:

1. Overview of ingest package
2. Installation instructions
3. Quick start examples
4. CLI usage
5. Python API usage
6. Prefect integration (optional)
7. Configuration options
8. Troubleshooting

**Acceptance**:

- [ ] README created
- [ ] Examples work
- [ ] Clear and comprehensive

---

### T071: Update main project README

**Type**: docs **Layer**: 8 **Dependencies**: T070 **Parallel**: [P] **Time**:
15 min

**Description**: Update project README with ingest package information

**File**: `README.md`

**Requirements**:

1. Add ingest package section
2. Link to ingest/README.md
3. Update installation instructions
4. Add CLI commands

**Acceptance**:

- [ ] Project README updated
- [ ] Links work
- [ ] Information accurate

---

### T072: Run pre-commit hooks and fix issues

**Type**: docs **Layer**: 8 **Dependencies**: T067, T068 **Parallel**: No
**Contract**: Gate 9 (plan.md) **Time**: 30 min

**Description**: Ensure all code passes pre-commit checks

**Commands**:

```bash
pixi run pre-commit-all
pixi run lint
pixi run format
```

**Requirements**:

1. All ruff checks pass
2. All formatting applied
3. No linting errors

**Acceptance**:

- [ ] Pre-commit hooks pass
- [ ] Code formatted correctly
- [ ] No linting errors

---

### T073: Generate test coverage report

**Type**: docs **Layer**: 8 **Dependencies**: T066 **Parallel**: [P] **Time**:
15 min

**Description**: Generate and verify test coverage ≥85%

**Commands**:

```bash
pixi run test-cov
```

**Requirements**:

1. Run pytest with coverage
2. Generate HTML report
3. Verify ≥85% coverage
4. Document uncovered areas

**Acceptance**:

- [ ] Coverage report generated
- [ ] Coverage ≥85%
- [ ] Constitution gate passed

---

### T074: Update AGENTS.md with implementation notes

**Type**: docs **Layer**: 8 **Dependencies**: T073 **Parallel**: [P] **Time**:
20 min

**Description**: Document implementation notes for future maintenance

**File**: `.github/AGENTS.md`

**Requirements**:

1. Document Pydantic usage patterns
2. Document CLI structure
3. Document testing approach
4. Add lessons learned

**Acceptance**:

- [ ] AGENTS.md updated
- [ ] Information useful for maintainers
- [ ] Links to relevant files

---

### T075: Create GitHub issue for known limitations

**Type**: docs **Layer**: 8 **Dependencies**: T074 **Parallel**: [P] **Time**:
15 min

**Description**: Document known limitations and future enhancements

**Tasks**:

1. List known limitations
2. Create GitHub issues for enhancements
3. Label issues appropriately
4. Link issues in documentation

**Acceptance**:

- [ ] Issues created
- [ ] Limitations documented
- [ ] Future work clear

---

### T076: Final validation checklist

**Type**: integration **Layer**: 8 **Dependencies**: T056-T065, T072, T073
**Parallel**: No **Time**: 30 min

**Description**: Final comprehensive validation before merge

**Checklist**:

1. [ ] All 79 tasks completed (75 required + 4 optional Prefect)
2. [ ] All tests pass (`pixi run test`)
3. [ ] Coverage ≥85% (`pixi run test-cov`)
4. [ ] Pre-commit hooks pass (`pixi run pre-commit-all`)
5. [ ] CLI entry point works (`ovro-lwa-ingest --help`)
6. [ ] All 8 acceptance scenarios pass (quickstart.md, including WCS Scenario 8)
7. [ ] All edge cases handled
8. [ ] WCS coordinates preserved and reconstructable
9. [ ] Documentation complete
10. [ ] No regressions in existing code
11. [ ] Constitution gates all passed

**Acceptance**:

- [ ] All checklist items pass
- [ ] WCS functionality verified
- [ ] Ready for code review
- [ ] Ready for merge to main

---

## Task Dependency Graph

```
Layer 0 (Setup): T001 → T002
                    ↓
Layer 1 (Models): T003 → T004 → [T005,T007,T009,T011,T013,T015,T016a] → [T006,T008,T010,T012,T014,T016,T016b]
                    ↓
Layer 2 (Settings): T017 → T018 → T019 → T020
                    ↓
Layer 3 (Utils): T021 → T022, T023 → T024, T025 → T026
                    ↓
Layer 4 (Core): T027 → T028 → T031 → T032
                              → T033 → T034 (depends on T016b for WCSHeader)
                              → T035 → T036 (depends on T016b for WCSHeader)
                  T029 → T030
                  T037 → T038
                  T039 → T040 → T041 → T042
                    ↓
Layer 5 (CLI): T043 → T044 → T045 → [T046,T047,T048,T049,T050] → T051
                    ↓
Layer 6 (Prefect): T052 → T053 → T054 → T055 [OPTIONAL]
                    ↓
Layer 7 (Integration): [T056-T062,T062a,T063-T066] (all parallel)
                    ↓
Layer 8 (Docs): [T067,T068,T069,T070,T071,T072,T073,T074,T075] → T076
```

**Key WCS Dependencies**:

- T016a, T016b: WCSHeader Pydantic model (new)
- T033, T034: Depend on T016b for WCSHeader integration
- T035, T036: Depend on T016b for WCS preservation through combine
- T062a: WCS integration test (new)

## Parallelization Strategy

**Maximum Parallelization Points**:

1. **After T004 (FITSImageFile)**: T005, T007, T009, T011, T013, T015, T016a (7
   parallel) ⬆️ +1
2. **After models complete**: T006, T008, T010, T012, T014, T016, T016b (7
   parallel) ⬆️ +1
3. **After T018 (ConversionOptions)**: T019, T020 (2 parallel)
4. **After T021 (locking tests)**: T022, T023, T025 (3 parallel)
5. **After T028 (discovery) + T016b (WCSHeader)**: T029, T031, T033, T035, T037
   (5 parallel)
6. **After T045 (CLI structure)**: T046, T047, T048, T049, T050 (5 parallel)
7. **After T051 (CLI complete)**: T056-T062, T062a, T063-T066 (12 parallel) ⬆️
   +1
8. **After T066 (performance)**: T067, T068, T069, T070, T071, T073, T074, T075
   (8 parallel)

**Estimated Time with Full Parallelization**: ~22-32 hours (increased due to WCS
features)

---

## Success Criteria

- [ ] All 79 tasks completed (75 required + 4 optional Prefect)
- [ ] All tests pass with ≥85% coverage
- [ ] All 8 acceptance scenarios validated (including WCS Scenario 8)
- [ ] All 3 edge cases handled
- [ ] All 10 constitution gates passed
- [ ] WCS coordinate preservation working correctly
- [ ] WCS reconstruction from Zarr store without FITS files
- [ ] CLI functional and user-friendly
- [ ] Documentation complete and clear
- [ ] No breaking changes to existing code
- [ ] Ready for production use

---

## Notes

- Tasks marked `[P]` can run in parallel with other `[P]` tasks at the same
  layer
- Optional Prefect tasks (T052-T055) can be skipped if not needed
- Test tasks must be written before implementation tasks (TDD)
- Pydantic validation is integral to all models
- All contract references point to specification documents in
  `specs/001-build-an-ingest/`
