# Implementation Status: FITS to Zarr Ingest Package

**Feature Branch**: `001-build-an-ingest` **Specification Date**: October 17,
2025 **Last Updated**: November 6, 2025 **Status**: Partially Implemented (Core
Functionality Complete)

## Executive Summary

The FITS to Zarr ingest package has been implemented with a **pragmatic,
simplified architecture** that delivers core functionality while deferring
advanced features for future iterations. The implementation prioritizes
**working software** over comprehensive feature coverage, following agile
principles.

### Implementation Approach

Rather than implementing the full Pydantic-based architecture outlined in the
original specification, the team chose a **wrapper-based approach** that:

- Builds on the proven `fits_to_zarr_xradio.py` logic
- Adds a thin orchestration layer for progress tracking and file locking
- Provides excellent user experience through CLI and Prefect integration
- Maintains framework independence for future extensibility

## Overall Status: 🟡 Partially Complete

| Category                | Planned                   | Implemented              | Status             |
| ----------------------- | ------------------------- | ------------------------ | ------------------ |
| **Core Conversion**     | Full modular pipeline     | Wrapper-based conversion | ✅ Complete        |
| **CLI Interface**       | Full-featured CLI         | Two-command CLI          | ✅ Complete        |
| **Data Models**         | 7 Pydantic models         | Simple dataclasses       | ❌ Not Implemented |
| **State Management**    | Resume/rebuild capability | Not implemented          | ❌ Deferred        |
| **Validation**          | Grid validation           | Basic validation         | ⚠️ Partial         |
| **Prefect Integration** | Optional orchestration    | Optional orchestration   | ✅ Complete        |
| **Documentation**       | Comprehensive             | Module README            | ✅ Complete        |

## Detailed Implementation Status

### ✅ Fully Implemented Features

#### 1. CLI Interface (FR-001, FR-030, FR-031, FR-032)

**Status**: ✅ Complete (Enhanced)

**Implemented**:

- `ovro-ingest convert` command with all essential options
- `ovro-ingest fix-headers` command (not in original spec - valuable addition)
- `ovro-ingest version` command
- Rich progress bars with time elapsed
- Configurable logging levels (debug, info, warning, error)
- Beautiful terminal output with colored status indicators
- Help documentation via `--help`

**Deviations from Spec**:

- ✅ **Better**: Two-step workflow support (fix-headers → convert)
- ❌ **Missing**: Interactive resume/rebuild prompts (FR-027, NFR-002)
- ❌ **Missing**: Detailed verbose logging with per-file metrics (FR-029)

#### 2. Core Conversion (FR-002 through FR-009, FR-014 through FR-018)

**Status**: ✅ Complete

**Implemented**:

- FITS file discovery with pattern matching
- Grouping by time step and frequency
- Automatic header correction (BSCALE/BZERO, WCS keywords)
- Zarr store creation and append modes
- Deterministic ordering (sorted by frequency and time)
- WCS coordinate preservation (RA/Dec)
- Configurable spatial chunking

**Architecture**:

- `FITSToZarrConverter` class wraps `fits_to_zarr_xradio.py`
- `ConversionConfig` dataclass for configuration
- Framework-independent core logic

#### 3. File Locking (FR-019)

**Status**: ✅ Complete

**Implemented**:

- Cross-platform file locking using `portalocker`
- Lock acquisition with non-blocking mode
- Automatic lock release via context manager
- Clear error messages on lock conflicts

#### 4. Prefect Integration (FR-022)

**Status**: ✅ Complete (As Specified)

**Implemented**:

- `fits_to_zarr_flow` Prefect flow
- Task-based decomposition (validate, prepare, execute)
- Graceful degradation when Prefect not installed
- Retry logic with configurable delays
- `run_conversion_flow` convenience wrapper

### ⚠️ Partially Implemented Features

#### 1. Progress Reporting (FR-023, FR-024, NFR-001)

**Status**: ⚠️ Partial

**Implemented**:

- Generic `ProgressCallback` protocol
- Rich progress bars in CLI
- Prefect logger integration
- Basic stage reporting (start, conversion, complete)

**Missing**:

- Detailed per-time-step progress
- File-level progress reporting
- Comprehensive timing metrics
- Data validation step reporting

#### 2. Error Handling (FR-025, FR-026, NFR-005)

**Status**: ⚠️ Partial

**Implemented**:

- Clear error messages for common scenarios
- Actionable suggestions (file not found, lock conflicts)
- Colored error output in CLI
- Exception propagation with context

**Missing**:

- Comprehensive error taxonomy
- Grid mismatch detection with detailed diagnostics (FR-009)
- Recovery suggestions for all edge cases

#### 3. Logging (FR-028, FR-029)

**Status**: ⚠️ Partial

**Implemented**:

- Configurable log levels via CLI flag
- Basic logging infrastructure
- Prefect logger integration

**Missing**:

- Detailed verbose mode with per-file processing
- Timing metrics per stage
- Data validation logging
- Structured logging (JSON format)

### ❌ Not Implemented Features

#### 1. Pydantic Data Models (Planned in data-model.md)

**Status**: ❌ Not Implemented

**Planned**:

- `FITSImageFile` - Frozen Pydantic model for FITS metadata
- `FixedFITSFile` - Tracking corrected FITS files
- `TimeStepGroup` - Collections of files by observation time
- `SpatialGrid` - Grid validation with checksums
- `ZarrStore` - Output store metadata
- `PipelineState` - State management for resume/rebuild
- `ConversionMetadata` - Aggregated conversion statistics
- `WCSHeader` - WCS coordinate preservation
- `ConversionOptions` - Pydantic Settings for configuration
- `AppConfig` - Environment-based configuration

**Why Not Implemented**:

- Simplified architecture prioritizes rapid delivery
- Simple dataclasses (`ConversionConfig`) sufficient for current needs
- Pydantic adds dependency weight without immediate benefit
- Runtime validation not critical for current use cases

**Future Consideration**: Add Pydantic models if:

- Complex validation requirements emerge
- API becomes public-facing
- Configuration management becomes more complex
- Runtime type checking proves valuable

#### 2. Modular Discovery and Validation (FR-005 through FR-009)

**Status**: ❌ Not Implemented as Standalone Modules

**Planned**:

- `discovery.py` - `discover_fits_files()`, `group_by_time_step()`
- `validation.py` - `validate_spatial_grid()`
- Standalone, testable functions

**Actual Implementation**:

- Discovery logic embedded in `fits_to_zarr_xradio.py`
- Grid validation performed during conversion
- No separate, reusable functions

**Impact**: Reduced modularity but faster implementation

#### 3. State Management (FR-027, NFR-002)

**Status**: ❌ Not Implemented

**Planned**:

- `PipelineState` with JSON persistence
- State tracking: `NOT_STARTED`, `IN_PROGRESS`, `COMPLETED`, `FAILED`
- Resume capability after interruption
- Interactive prompts to resume or rebuild
- Atomic state file writes

**Why Not Implemented**:

- Complex feature requiring careful design
- Edge cases around crash recovery
- Incremental development prioritized core conversion
- Can be added in future iteration

**Workaround**: Users can manually manage incremental conversions using append
mode

#### 4. Comprehensive Grid Validation (FR-009)

**Status**: ❌ Not Explicitly Implemented

**Planned**:

- Pre-validation of spatial grids before conversion
- MD5 checksums for efficient grid comparison
- Detailed error messages with grid dimensions
- Suggested actions for mismatches

**Actual Implementation**:

- Basic grid checks during `xradio` operations
- Runtime errors if grids mismatch
- Less detailed diagnostics

**Impact**: Users may get less actionable error messages for grid issues

## Functional Requirements Coverage

### Core Conversion Capabilities (FR-001 through FR-009)

- ✅ **FR-001**: CLI interface provided
- ✅ **FR-002**: Input directory argument
- ✅ **FR-003**: Output directory argument
- ✅ **FR-004**: Zarr store name argument (with default)
- ✅ **FR-005**: FITS file discovery by pattern
- ✅ **FR-006**: Files sorted by frequency
- ✅ **FR-007**: Multiple subbands combined
- ✅ **FR-008**: Spatial grid consistency maintained
- ⚠️ **FR-009**: Grid validation (basic, not comprehensive)

### FITS Header Correction (FR-010 through FR-013)

- ✅ **FR-010**: FITS file correction detection
- ✅ **FR-011**: Fixed FITS files generated
- ✅ **FR-012**: Configurable fixed FITS directory
- ✅ **FR-013**: Reuse of existing fixed files

### Zarr Output Management (FR-014 through FR-019c)

- ✅ **FR-014**: Zarr format compatible with xarray/xradio
- ✅ **FR-015**: Append support
- ✅ **FR-016**: Rebuild/overwrite option
- ✅ **FR-017**: Atomic writes (via xradio)
- ✅ **FR-018**: Configurable chunking
- ✅ **FR-019**: File locking for concurrent writes
- ✅ **FR-019a-f**: WCS coordinate preservation (implemented in
  `fits_to_zarr_xradio.py`)

### Pipeline Management (FR-020 through FR-029)

- ⚠️ **FR-020**: Pipeline tasks (not as discrete as planned)
- ✅ **FR-021**: Framework-independent core
- ✅ **FR-022**: Optional Prefect integration
- ⚠️ **FR-023**: Progress logging (basic)
- ⚠️ **FR-024**: Summary statistics (basic)
- ✅ **FR-025**: Error handling
- ✅ **FR-026**: Input validation
- ❌ **FR-027**: Resume/rebuild prompts (not implemented)
- ⚠️ **FR-028**: Configurable logging levels (basic)
- ❌ **FR-029**: Verbose logging with metrics (not implemented)

### Installation & Discoverability (FR-030 through FR-032)

- ✅ **FR-030**: CLI entry point
- ✅ **FR-031**: Help documentation
- ✅ **FR-032**: Version information

## Non-Functional Requirements Coverage

- ✅ **NFR-001**: User-friendly CLI with progress indicators
- ❌ **NFR-002**: Resumable pipeline (not implemented - major gap)
- ✅ **NFR-003**: Deterministic conversions
- ✅ **NFR-004**: Handles typical dataset sizes (best effort)
- ⚠️ **NFR-005**: Clear error messages (good but could be better)

## Architecture Comparison

### Planned Architecture (from plan.md)

```
Layer 0: Setup
Layer 1: Pydantic Models (7 models)
Layer 2: Pydantic Settings
Layer 3: Core Utilities (locking, state, logging)
Layer 4: Conversion Logic (modular functions)
Layer 5: Discovery & Orchestration
Layer 6: CLI Interface
Layer 7: Optional Prefect Layer
Layer 8: Integration Tests
```

### Actual Architecture

```
Layer 0: Existing fits_to_zarr_xradio.py (proven logic)
Layer 1: Core wrapper (ConversionConfig, FITSToZarrConverter, FileLock)
Layer 2: CLI (Typer + Rich)
Layer 3: Optional Prefect integration
Layer 4: Documentation and tests
```

**Simplification Rationale**:

- Leverage existing, tested conversion logic
- Minimize development time
- Reduce complexity and maintenance burden
- Focus on user-facing features (CLI, progress, errors)
- Defer advanced features until proven necessary

## Task Completion Status

From `tasks.md` (79 total tasks planned):

| Layer                | Tasks Planned | Tasks Completed | Completion % |
| -------------------- | ------------- | --------------- | ------------ |
| 0: Setup             | 2             | 2               | 100%         |
| 1: Pydantic Models   | 21            | 0               | 0%           |
| 2: Pydantic Settings | 6             | 0               | 0%           |
| 3: Core Utilities    | 6             | 1 (locking)     | 17%          |
| 4: Conversion Logic  | 11            | 2 (wrapper)     | 18%          |
| 5: CLI Interface     | 8             | 8               | 100%         |
| 6: Prefect           | 4             | 4               | 100%         |
| 7: Integration Tests | 12            | ~3              | 25%          |
| 8: Documentation     | 9             | 3               | 33%          |
| **Total**            | **79**        | **~23**         | **~29%**     |

**Note**: Task completion percentage is misleading because the implementation
took a different architectural path. Core functionality is complete despite low
task count.

## Known Limitations and Future Work

### High Priority (User-Impacting)

1. **No Resume Capability** (NFR-002 violation)
   - **Impact**: Long-running conversions cannot be resumed after interruption
   - **Workaround**: Use append mode and track completed time steps manually
   - **Future Work**: Implement `PipelineState` with JSON persistence

2. **Limited Grid Validation** (FR-009 partially met)
   - **Impact**: Grid mismatches produce less actionable error messages
   - **Workaround**: Pre-validate FITS files manually
   - **Future Work**: Add `SpatialGrid` model with checksum validation

3. **Basic Progress Reporting**
   - **Impact**: No per-time-step or per-file progress details
   - **Workaround**: Monitor log files
   - **Future Work**: Enhanced progress callbacks with granular reporting

### Medium Priority (Developer Experience)

4. **No Modular Discovery/Validation**
   - **Impact**: Cannot reuse discovery logic independently
   - **Workaround**: Import from `fits_to_zarr_xradio.py`
   - **Future Work**: Extract to `discovery.py` and `validation.py`

5. **Limited Test Coverage**
   - **Impact**: Integration tests incomplete
   - **Workaround**: Manual testing
   - **Future Work**: Complete integration test suite from tasks.md

6. **No Pydantic Models**
   - **Impact**: No runtime validation, less type safety
   - **Workaround**: Manual validation in code
   - **Future Work**: Add Pydantic models if API becomes public

### Low Priority (Nice to Have)

7. **Basic Logging Configuration**
   - **Impact**: No structured logging or detailed metrics
   - **Workaround**: Increase log level to debug
   - **Future Work**: Implement comprehensive logging from FR-028/029

8. **Contract Documentation Out of Sync**
   - **Impact**: Specification documents don't match implementation
   - **Workaround**: Read source code
   - **Future Work**: Update or archive contract documents

## Architectural Decisions

### Decision 1: Wrapper vs Modular Rewrite

**Choice**: Wrapper-based architecture wrapping `fits_to_zarr_xradio.py`

**Rationale**:

- Existing conversion logic is proven and tested
- Reduces development time significantly
- Minimizes risk of introducing bugs
- Allows incremental improvements

**Trade-offs**:

- Less modularity than planned
- Harder to unit test individual components
- Dependency on monolithic conversion function

**Future Path**: Gradually extract modules from `fits_to_zarr_xradio.py` as
needed

### Decision 2: Dataclasses vs Pydantic

**Choice**: Simple dataclasses (`ConversionConfig`) instead of Pydantic models

**Rationale**:

- Simpler implementation with fewer dependencies
- Sufficient for current configuration needs
- Pydantic benefits (runtime validation, settings management) not critical yet
- Can add Pydantic later if needed

**Trade-offs**:

- No runtime type checking
- Manual validation required
- No automatic environment variable loading

**Future Path**: Consider Pydantic if:

- Configuration becomes more complex
- API becomes public-facing
- Runtime validation proves valuable

### Decision 3: No State Management (Yet)

**Choice**: Defer resume/rebuild capability

**Rationale**:

- Complex feature with many edge cases
- Append mode provides partial solution
- Most conversions complete without interruption
- Can be added later without breaking changes

**Trade-offs**:

- Users cannot resume interrupted conversions
- Long-running jobs vulnerable to interruption
- Violates NFR-002

**Future Path**: Implement `PipelineState` in future iteration based on user
feedback

### Decision 4: CLI Enhancement (fix-headers command)

**Choice**: Add `ovro-ingest fix-headers` command (not in original spec)

**Rationale**:

- Enables two-step workflow (separate header fixing from conversion)
- Useful for debugging and workflow flexibility
- Low implementation cost
- Enhances user experience

**Trade-offs**: None (pure addition)

## Success Metrics

| Metric                           | Target       | Actual        | Status |
| -------------------------------- | ------------ | ------------- | ------ |
| Core conversion works            | Yes          | Yes           | ✅     |
| CLI functional                   | Yes          | Yes           | ✅     |
| File locking prevents corruption | Yes          | Yes           | ✅     |
| WCS preservation                 | Yes          | Yes           | ✅     |
| Progress indicators              | Yes          | Yes           | ✅     |
| Prefect integration optional     | Yes          | Yes           | ✅     |
| Resume capability                | Yes          | No            | ❌     |
| Grid validation                  | Yes          | Partial       | ⚠️     |
| Comprehensive testing            | 85% coverage | ~50% coverage | ⚠️     |

**Overall Assessment**: Core functionality delivered successfully. Advanced
features deferred for future iterations.

## Recommendations

### For Users

1. **Use append mode** for incremental conversions instead of resume capability
2. **Pre-validate FITS files** manually if grid consistency is critical
3. **Monitor conversions** actively for long-running jobs
4. **Use fix-headers command** for better control over two-step workflow

### For Developers

1. **Update contract documents** to reflect actual implementation or archive
   them
2. **Add integration tests** for critical workflows
3. **Consider Pydantic migration** if API becomes public or configuration grows
4. **Implement state management** based on user feedback
5. **Extract modular functions** from `fits_to_zarr_xradio.py` as needed

### For Future Development

1. **Phase 2 Features** (based on user demand):
   - Resume/rebuild capability (NFR-002)
   - Comprehensive grid validation (FR-009)
   - Detailed progress reporting (FR-029)
   - Enhanced error diagnostics

2. **Refactoring Opportunities**:
   - Extract discovery module
   - Extract validation module
   - Add Pydantic models for public API
   - Improve test coverage

3. **Documentation**:
   - Update specification to reflect implementation
   - Add architectural decision records (ADRs)
   - Create migration guide from spec to implementation

## Conclusion

The FITS to Zarr ingest package delivers **core functionality successfully**
with a **pragmatic, simplified architecture**. While the implementation deviates
significantly from the original comprehensive specification, it achieves the
primary goal: **a working, user-friendly tool for converting OVRO-LWA FITS files
to Zarr format**.

The decision to prioritize delivery over comprehensive feature coverage follows
agile principles and allows for iterative improvement based on real user
feedback. Advanced features like resume capability and comprehensive validation
can be added in future iterations without breaking changes.

**Key Takeaway**: Sometimes the best architecture is the simplest one that
works.
