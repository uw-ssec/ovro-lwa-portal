# Research: FITS to Zarr Ingest Package with CLI

**Date**: October 20, 2025 **Feature**: 001-build-an-ingest

## Overview

This document consolidates research findings for building a production-ready
FITS to Zarr ingestion pipeline with CLI interface. All technical unknowns from
the specification have been resolved through analysis of existing codebase,
dependencies, and best practices.

## Research Areas

### 1. CLI Framework Selection

**Decision**: Typer v0.9+

**Rationale**:

- User requirement explicitly specified Typer
- Modern Python CLI framework built on Click with excellent type hint support
- Automatic help generation from docstrings and type annotations
- Native support for enums, optional parameters, and validation
- Rich console output integration for progress indicators (NFR-001)
- Excellent documentation and active maintenance

**Alternatives Considered**:

- **Click**: Lower-level, more boilerplate, lacks automatic type validation
- **argparse**: Standard library but verbose, poor developer experience
- **Fire**: Too magical, lacks explicit typing, harder to test

**Integration Points**:

- Entry point in `pyproject.toml`: `[project.scripts]` section
- CLI module at `src/ovro_lwa_portal/ingest/cli.py`
- Rich console for progress bars and formatted output
- Typer v0.12+ for native Pydantic model support

### 2. Data Validation and Configuration Management

**Decision**: Pydantic v2.0+ for data models and pydantic-settings for
configuration

**Rationale**:

- **Runtime validation**: Catch data errors immediately with automatic
  validation
- **Type safety**: Full Python 3.12+ type hints with runtime enforcement via
  Pydantic
- **Configuration management**: pydantic-settings loads config from environment
  variables, config files, and CLI arguments with proper typing
- **Serialization**: Built-in `model_dump()` and `model_dump_json()` for JSON
  state files (FR-017, NFR-002)
- **CLI integration**: Typer v0.12+ has native support for Pydantic models as
  CLI parameters
- **IDE support**: Excellent autocomplete and error detection in modern IDEs
- **Performance**: Pydantic v2 uses Rust core for fast validation with minimal
  overhead
- **Documentation**: Field-level descriptions via `Field(description="...")` for
  self-documenting models

**Alternatives Considered**:

- **dataclasses**: No runtime validation, manual serialization, no configuration
  management
- **attrs**: Runtime validation via validators but less ergonomic than Pydantic,
  no settings management
- **marshmallow**: Older, separate schema/object distinction, not integrated
  with type hints
- **Manual validation**: Error-prone, verbose, not DRY

**Integration Strategy**:

- **Core models**: `models.py` with Pydantic BaseModel classes for
  FITSImageFile, TimeStepGroup, ZarrStore, etc.
- **Configuration**: `config.py` with pydantic-settings BaseSettings for
  ConversionOptions and AppConfig
- **Field validators**: `@field_validator` for custom validation logic (e.g.,
  path existence, frequency validation)
- **Model validators**: `@model_validator` for cross-field validation (e.g.,
  grid consistency)
- **Frozen models**: `model_config = ConfigDict(frozen=True)` for immutable
  entities
- **CLI binding**: Typer automatically validates CLI arguments against Pydantic
  models
- **State persistence**: PipelineState model with `model_dump_json()` for atomic
  JSON writes

**Benefits for This Feature**:

- FR-009: Spatial grid validation with Pydantic validators automatically checks
  grid consistency
- FR-017, NFR-002: Pipeline state serialization via `model_dump_json()` with
  atomic writes
- FR-019: Lock file paths validated at model creation time
- FR-030, FR-031: CLI arguments validated before processing begins
- NFR-003: Deterministic serialization via Pydantic ensures consistent state
  files
- NFR-005: Pydantic ValidationError messages are automatically actionable with
  field names and constraints

### 3. Pipeline Orchestration (Prefect)

**Decision**: Optional integration as orchestration layer wrapping core logic

**Rationale**:

- Clarification specified: "Optional integration (core logic independent,
  Prefect as optional orchestration layer)" (FR-021, FR-022)
- Core conversion logic MUST NOT depend on Prefect
- Prefect provides: flow management, task retries, observability UI, state
  tracking
- Enables future scaling to distributed processing without core logic changes

**Alternatives Considered**:

- **No orchestration**: Simpler but lacks observability and retry mechanisms
- **Luigi**: Older, less active development, heavier dependencies
- **Airflow**: Overkill for this use case, complex setup requirements
- **Dask Delayed**: Already available via dask dependency, but less
  feature-complete

**Integration Strategy**:

- Core logic in `core.py`, `discovery.py`, `validation.py` (pure Python, no
  Prefect or Pydantic dependencies)
- Pydantic models passed between functions for type safety
- Optional Prefect flows in `prefect_flows.py` wrapping core functions
- Prefect as optional dependency:
  `[tool.pixi.feature.prefect.pypi-dependencies]`
- CLI flag `--use-prefect` to enable orchestration layer

### 4. File Locking Strategy

**Decision**: `fcntl` (POSIX) + `msvcrt` (Windows) via `portalocker` library

**Rationale**:

- Clarification specified: "System MUST detect concurrent writes and fail fast
  with clear error (file locking)" (FR-019)
- `portalocker` provides cross-platform file locking abstraction
- Non-blocking lock attempts with immediate failure on conflict
- Lock files placed alongside Zarr store: `{zarr_path}.lock`
- Automatic lock cleanup on process termination

**Alternatives Considered**:

- **PID files**: Race conditions, stale PID issues, requires manual cleanup
- **Database locks**: Overkill, adds dependency
- **Distributed locks (Redis/etcd)**: Not needed for single-machine scenario

**Implementation**:

- Lock acquisition in `locking.py` module
- Context manager pattern: `with acquire_lock(zarr_path): ...`
- Clear error messages including PID of lock holder

### 4. Pipeline State Management

**Decision**: JSON state file with atomic writes

**Rationale**:

- Clarification specified: "Prompt user interactively on restart (ask whether to
  resume or rebuild)" (FR-027, NFR-002)
- State file tracks: current time step, processed files, spatial grid reference
- Atomic writes via temp file + rename for crash safety
- State file location: `{zarr_path}.state.json`

**State Schema**:

```json
{
  "version": "1.0.0",
  "zarr_path": "/path/to/output.zarr",
  "start_time": "2025-10-20T10:30:00Z",
  "last_update": "2025-10-20T10:35:00Z",
  "completed_time_steps": ["20240524_050019", "20240524_051019"],
  "current_time_step": "20240524_052019",
  "spatial_grid_reference": {
    "l_shape": 2048,
    "m_shape": 2048,
    "l_checksum": "abc123...",
    "m_checksum": "def456..."
  },
  "total_files_processed": 1200,
  "status": "in_progress"
}
```

**Alternatives Considered**:

- **Prefect state**: Couples to orchestration framework (violates FR-021)
- **SQLite**: Overkill for simple state tracking
- **YAML**: No atomic write support, slower parsing

### 6. Logging Configuration

**Decision**: Python `logging` module with configurable levels via CLI

**Rationale**:

- Clarification specified: "Configurable (user-selectable verbosity levels via
  CLI flag), with verbose option including detailed per-file processing, timing
  metrics, and data validation steps" (FR-028, FR-029)
- Standard library, no additional dependencies
- Structured logging with JSON formatter for production use
- Integration with existing logger in `fits_to_zarr_xradio.py`

**Log Levels**:

- `--quiet`: ERROR only
- Default: INFO (progress, time steps, summaries)
- `--verbose`: DEBUG (per-file processing, timing, validation details)

**Implementation**:

- Configure in CLI entry point based on flags
- Structured format:
  `{"timestamp": "...", "level": "...", "message": "...", "context": {...}}`
- File output optional: `--log-file` flag

### 6. Existing Code Refactoring

**Decision**: Extract and modularize `fits_to_zarr_xradio.py` logic

**Rationale**:

- Existing module (`fits_to_zarr_xradio.py`) contains production-quality
  conversion logic
- Current structure: Monolithic function with all logic inline
- Target structure: Modular components (discovery, validation, conversion,
  writing)

**Refactoring Plan**:

1. **Preserve existing function**: Keep `convert_fits_dir_to_zarr()` as
   high-level API
2. **Extract to modules**:
   - `discovery.py`: `_discover_groups()`, `_mhz_from_name()` →
     `discover_fits_files()`, `group_by_time_step()`
   - `validation.py`: `_assert_same_lm()` → `validate_spatial_grid()`,
     `ValidationError`
   - `core.py`: `_fix_headers()`, `_load_for_combine()`, `_combine_time_step()`
     → `fix_fits_headers()`, `load_fits_for_zarr()`,
     `combine_frequency_subbands()`
3. **Add new modules**:
   - `locking.py`: Concurrent access control
   - `state.py`: Pipeline state persistence
   - `cli.py`: Typer CLI
   - `prefect_flows.py`: Optional orchestration

**Backward Compatibility**:

- Keep original function signature intact
- Internal implementation delegates to new modular functions
- No breaking changes for existing users

### 8. Testing Strategy

**Decision**: TDD with pytest, contract tests, and integration tests

**Rationale**:

- Constitution requirement: "TDD mandatory for all new features" (Section II)
- Contract tests verify API stability (FR-030: CLI entry point)
- Integration tests validate end-to-end workflows (all acceptance scenarios)

**Test Structure**:

```text
tests/ingest/
├── test_core.py              # Unit tests for conversion logic
├── test_discovery.py         # Unit tests for file discovery
├── test_validation.py        # Unit tests for validation
├── test_locking.py           # Unit tests for file locking (concurrent scenarios)
├── test_state.py             # Unit tests for state management
├── test_cli.py               # Integration tests for CLI (subprocess invocation)
└── test_prefect_flows.py     # Integration tests for Prefect (if installed)
```

**Test Data**:

- Use existing test FITS files in `notebooks/test_fits_files/`
- Download via `.ci-helpers/download_test_fits.py` in CI
- Mock S3 access with pytest fixtures
- Synthetic grid data for validation tests

### 9. Dependency Management

**Decision**: Add to existing Pixi configuration

**New Dependencies**:

- **Required**:
  - `pydantic>=2.0` (data models and validation)
  - `pydantic-settings>=2.0` (configuration management)
  - `typer>=0.12.0` (CLI framework with Pydantic support)
  - `rich>=13.0.0` (terminal formatting, progress bars)
  - `portalocker>=2.8.0` (cross-platform file locking)
- **Optional**:
  - `prefect>=2.14.0` (pipeline orchestration)

**Installation**:

```bash
# Required dependencies
pixi add --pypi pydantic pydantic-settings typer rich portalocker

# Optional Prefect feature
pixi add --feature prefect --pypi prefect
```

**pyproject.toml Update**:

```toml
[project.dependencies]
# ... existing dependencies ...
pydantic = ">=2.0"
pydantic-settings = ">=2.0"
typer = ">=0.12.0"
rich = ">=13.0.0"
portalocker = ">=2.8.0"

[project.scripts]
ovro-lwa-ingest = "ovro_lwa_portal.ingest.cli:app"

[project.optional-dependencies]
prefect = ["prefect>=2.14.0"]
```

### 10. Performance Considerations

**Decision**: Best-effort optimization with correctness priority

**Rationale**:

- Clarification: "Best effort (no specific target, optimize for correctness over
  speed)" (NFR-004)
- Existing code uses lazy loading (xarray + dask)
- Constitution: "Memory usage ≤4GB for typical datasets"

**Optimizations**:

- Chunked processing via xarray (already implemented)
- Parallel subband loading (dask)
- Atomic writes with temp files (avoid partial corruption)
- Reuse fixed FITS files (FR-013)

**Monitoring**:

- Timing metrics in verbose logging
- Memory profiling in CI (constitution requirement)
- Performance regression tests for core conversion

## Summary of Decisions

| Area                  | Decision                                      | Rationale                                                               |
| --------------------- | --------------------------------------------- | ----------------------------------------------------------------------- |
| CLI Framework         | Typer v0.12+                                  | User requirement, native Pydantic support, automatic help               |
| Data Validation       | Pydantic v2.0+ for models                     | Runtime validation, type safety, serialization, CLI integration         |
| Configuration         | pydantic-settings for config management       | Environment variables, type-safe config, validation                     |
| Orchestration         | Optional Prefect layer                        | FR-021: Core logic independent, Prefect wraps for observability         |
| File Locking          | portalocker (cross-platform)                  | FR-019: Detect concurrent writes, fail fast with clear error            |
| State Management      | Pydantic models + JSON atomic writes          | FR-027: Type-safe state, resume/rebuild prompts, crash-safe persistence |
| Logging               | Python logging + structured JSON              | FR-028/029: Configurable verbosity, detailed verbose mode               |
| Code Structure        | Modular refactor, Pydantic models, legacy API | FR-021: Framework-independent, type-safe, backward compatible           |
| Testing               | TDD with pytest, Pydantic validation tests    | Constitution Section II: TDD mandatory, runtime validation              |
| Dependencies          | Pixi + pyproject.toml updates (add Pydantic)  | Existing build system, add pydantic/pydantic-settings, optional Prefect |
| Performance           | Best-effort, correctness priority             | NFR-004: Clarified no specific time targets, Pydantic minimal overhead  |
| Interactive Prompts   | rich.Prompt for resume/rebuild                | NFR-002: User-friendly, clear options                                   |
| Progress Indicators   | rich.Progress for long operations             | NFR-001: User-friendly CLI output                                       |
| Error Messages        | Pydantic ValidationError + custom exceptions  | NFR-005: Actionable messages with field names, suggested fixes          |
| Spatial Grid Tracking | Pydantic model with checksum validation       | FR-009: Type-safe grid tracking, detect mismatches                      |

## Open Questions Resolved

All technical unknowns from the specification have been addressed:

- ✅ CLI framework choice (Typer v0.12+ with Pydantic support)
- ✅ Data validation approach (Pydantic v2.0+ for runtime validation)
- ✅ Configuration management (pydantic-settings for type-safe config)
- ✅ Prefect integration scope (optional orchestration layer)
- ✅ Concurrent access control (file locking with portalocker)
- ✅ Resume/rebuild mechanism (Pydantic state models + JSON + interactive
  prompts)
- ✅ Logging verbosity (configurable via CLI flags)
- ✅ Performance expectations (best-effort, correctness priority, Pydantic
  minimal overhead)
- ✅ Code structure (modular refactor with Pydantic models)
- ✅ Testing approach (TDD with pytest, Pydantic validation tests, contract +
  integration tests)

## Next Steps

Proceed to Phase 1: Design & Contracts

- Define data model for key entities (FITS metadata, Zarr store, pipeline state)
- Generate API contracts for core conversion functions
- Create contract tests (must fail initially per TDD)
- Extract user story validation scenarios
- Generate quickstart.md for end-to-end validation
