# Feature Specification: FITS to Zarr Ingest Package with CLI

**Feature Branch**: `001-build-an-ingest` **Created**: October 17, 2025
**Status**: Draft **Input**: User description: "Build an ingest package within
the ovro_lwa_portal library that converts fits files to a zarr file as it exists
in the fits_to_zarr_xradio.py module. This package should be exposed as a CLI
with arguments such as the path to the input fits files directory and path to
output zarr file. Specific python libraries that should be used are the current
library dependencies as well as typer CLI library and prefect for pythonic data
pipeline."

## Execution Flow (main)

```
1. Parse user description from Input
   → Feature description provided: Create ingest package for FITS→Zarr conversion with CLI
2. Extract key concepts from description
   → Actors: Radio astronomy researchers, data pipeline operators
   → Actions: Convert FITS files to Zarr format, execute via command line
   → Data: OVRO-LWA FITS image files, Zarr stores
   → Constraints: Use existing conversion logic, expose via CLI, use typer and prefect
3. For each unclear aspect:
   → [RESOLVED] Use existing fits_to_zarr_xradio.py logic
   → [RESOLVED] CLI framework: typer
   → [RESOLVED] Pipeline framework: prefect
4. Fill User Scenarios & Testing section
   → User flow: CLI invocation → FITS conversion → Zarr output
5. Generate Functional Requirements
   → All requirements testable via CLI execution and output validation
6. Identify Key Entities
   → FITS image files, Zarr store, conversion metadata
7. Run Review Checklist
   → No [NEEDS CLARIFICATION] markers
   → Implementation framework specified by user (typer, prefect)
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines

- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements

- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

---

## Clarifications

### Session 2025-10-17

- Q: When multiple CLI processes attempt to write to the same output Zarr path
  simultaneously, what should happen? → A: System MUST detect concurrent writes
  and fail fast with clear error (file locking)
- Q: When a conversion pipeline is interrupted (keyboard interrupt, system
  crash), how should resumption work? → A: Prompt user interactively on restart
  (ask whether to resume or rebuild)
- Q: For typical OVRO-LWA datasets (dozens of time steps, hundreds of FITS
  files), what are the acceptable processing time expectations? → A: Best effort
  (no specific target, optimize for correctness over speed)
- Q: What level of operational logging should the pipeline provide beyond
  progress indicators? → A: Configurable (user-selectable verbosity levels via
  CLI flag), with verbose option including detailed per-file processing, timing
  metrics, and data validation steps
- Q: How should the Prefect pipeline framework be utilized in this feature? → A:
  Optional integration (core logic independent, Prefect as optional
  orchestration layer)

---

## User Scenarios & Testing _(mandatory)_

### Primary User Story

As a **radio astronomy researcher** or **data pipeline operator**, I need to
convert multiple OVRO-LWA FITS image files (organized by time and frequency)
into a single, optimized Zarr data store for efficient access and analysis. I
want to execute this conversion from the command line by specifying the input
directory containing FITS files and the output path for the Zarr store, with the
process managed as a robust data pipeline that can handle errors, track
progress, and resume if interrupted.

### Acceptance Scenarios

1. **Given** a directory containing OVRO-LWA FITS files with standard naming
   convention (YYYYMMDD*HHMMSS*<SB>MHz*averaged*\*-I-image.fits), **When** I
   execute the CLI command with the input directory and output Zarr path,
   **Then** the system creates a single Zarr store combining all time steps and
   frequency subbands with correct dimensional ordering (time, frequency, l, m).

2. **Given** an existing Zarr store at the output path, **When** I execute the
   CLI command with new FITS files and the same output path, **Then** the system
   appends the new time steps to the existing Zarr store without corrupting
   existing data.

3. **Given** FITS files that need header corrections (BSCALE/BZERO application,
   missing WCS keywords), **When** the conversion process runs, **Then** the
   system automatically generates fixed FITS files in a designated directory
   before conversion.

4. **Given** a conversion process in progress, **When** I query the CLI or
   pipeline system, **Then** I can see progress information including which time
   step is being processed and overall completion status.

5. **Given** FITS files with mismatched spatial grids (l/m coordinates),
   **When** the conversion attempts to combine them, **Then** the system detects
   the mismatch, reports an error with details, and stops processing without
   producing corrupt output.

6. **Given** I want to rebuild a Zarr store from scratch, **When** I execute the
   CLI with a rebuild/overwrite flag, **Then** the system replaces any existing
   Zarr store at the output path.

7. **Given** the CLI is installed in my Python environment, **When** I invoke
   the CLI with no arguments or with `--help`, **Then** I see clear usage
   instructions including all available options and their descriptions.

### Edge Cases

- What happens when the input directory contains no matching FITS files? →
  System reports error and exits without creating output files.

- What happens when FITS files for a single time step span multiple frequency
  subbands with gaps? → System combines available subbands and logs warning
  about gaps in frequency coverage.

- What happens when the output directory doesn't exist? → System creates
  necessary parent directories automatically.

- What happens when disk space is insufficient during conversion? → System
  detects write errors and reports them clearly, leaving partial output in a
  detectable state.

- What happens when the conversion is interrupted (keyboard interrupt, system
  failure)? → Pipeline framework tracks state; on restart, the system
  interactively prompts the user to choose whether to resume from the last
  checkpoint or rebuild from scratch.

- What happens when running multiple CLI invocations simultaneously to the same
  output path? → System detects concurrent writes via file locking and fails
  fast with a clear error message indicating another process is using the output
  path.

---

## Requirements _(mandatory)_

### Functional Requirements

#### Core Conversion Capabilities

- **FR-001**: System MUST provide a command-line interface for invoking FITS to
  Zarr conversion
- **FR-002**: System MUST accept an input directory path containing FITS files
  as a required argument
- **FR-003**: System MUST accept an output directory path for the Zarr store as
  a required argument
- **FR-004**: System MUST accept an output Zarr store name as a configurable
  argument with a sensible default
- **FR-005**: System MUST discover and group FITS files by time step using the
  standard OVRO-LWA naming pattern (YYYYMMDD*HHMMSS*<SB>MHz)
- **FR-006**: System MUST sort FITS files by frequency (subband MHz) within each
  time step for deterministic ordering
- **FR-007**: System MUST combine multiple frequency subbands for each time step
  into a unified dataset
- **FR-008**: System MUST maintain consistent spatial grid (l, m coordinates)
  across all time steps
- **FR-009**: System MUST validate spatial grid consistency and abort with
  detailed error if grids differ

#### FITS Header Correction

- **FR-010**: System MUST detect FITS files requiring header corrections
  (BSCALE/BZERO application, missing WCS keywords)
- **FR-011**: System MUST generate corrected "\_fixed.fits" files in a
  designated directory before conversion
- **FR-012**: System MUST accept a configurable path for storing fixed FITS
  files
- **FR-013**: System MUST reuse existing fixed FITS files if they already exist,
  avoiding redundant corrections

#### Zarr Output Management

- **FR-014**: System MUST write converted data to Zarr format compatible with
  xarray and xradio libraries
- **FR-015**: System MUST support appending new time steps to an existing Zarr
  store
- **FR-016**: System MUST provide an option to rebuild/overwrite existing Zarr
  stores
- **FR-017**: System MUST apply safe write operations when appending to avoid
  corrupting existing data (e.g., atomic swaps)
- **FR-018**: System MUST support configurable chunking for spatial dimensions
  (l, m)
- **FR-019**: System MUST detect concurrent writes to the same output path using
  file locking and fail immediately with a clear error message

#### Pipeline Management

- **FR-020**: System MUST organize conversion as a data pipeline with discrete
  tasks (discovery, grouping, fixing, combining, writing)
- **FR-021**: System MUST implement core conversion logic independently of any
  specific orchestration framework
- **FR-022**: System MUST provide Prefect integration as an optional
  orchestration layer that wraps core conversion logic
- **FR-023**: System MUST log progress information including current time step
  being processed
- **FR-024**: System MUST report summary statistics after completion (total
  files processed, time steps, frequency range)
- **FR-025**: System MUST handle errors gracefully and provide actionable error
  messages
- **FR-026**: System MUST validate required inputs and report clear errors for
  missing or invalid arguments
- **FR-027**: System MUST detect incomplete pipeline state on startup and
  interactively prompt user to resume or rebuild
- **FR-028**: System MUST provide configurable logging verbosity levels via CLI
  flag (e.g., quiet, normal, verbose)
- **FR-029**: System MUST support verbose logging mode that includes detailed
  per-file processing, timing metrics, and data validation steps

#### Installation & Discoverability

- **FR-030**: System MUST expose the CLI as an entry point installed with the
  ovro_lwa_portal package
- **FR-031**: System MUST provide help documentation accessible via `--help`
  flag
- **FR-032**: System MUST display version information when requested

### Non-Functional Requirements

- **NFR-001**: CLI MUST provide user-friendly output with progress indicators
  during long-running conversions
- **NFR-002**: Pipeline MUST be resumable after interruption; on restart, system
  MUST interactively prompt user to choose between resuming from last checkpoint
  or rebuilding from scratch
- **NFR-003**: Conversion process MUST be deterministic (same inputs produce
  identical outputs)
- **NFR-004**: System MUST handle typical OVRO-LWA dataset sizes (dozens of time
  steps, hundreds of FITS files per time step); performance optimization is best
  effort with priority on correctness over speed
- **NFR-005**: Error messages MUST be clear enough for radio astronomy
  researchers without deep Python knowledge to diagnose common issues

### Key Entities _(include if feature involves data)_

- **FITS Image File**: Radio astronomy image file in FITS format containing
  single time step and single frequency subband data. Key attributes include
  observation timestamp (YYYYMMDD_HHMMSS), subband frequency (MHz), spatial grid
  (l, m coordinates), pixel intensity values, and header metadata (BSCALE/BZERO,
  WCS keywords, beam parameters).

- **Fixed FITS File**: Corrected version of FITS Image File with BSCALE/BZERO
  applied to pixel data, mandatory WCS keywords added (RESTFREQ, SPECSYS,
  TIMESYS, RADESYS), and identity PC matrix for spatial coordinates. Generated
  automatically when needed.

- **Zarr Store**: Cloud-optimized array storage format containing combined
  multi-dimensional dataset with dimensions (time, frequency, polarization, l,
  m). Organized as directory structure with chunked arrays and metadata files.
  Supports efficient appending and partial reading.

- **Time Step Group**: Collection of FITS files sharing the same observation
  timestamp (YYYYMMDD_HHMMSS) but spanning different frequency subbands.
  Represents a single temporal snapshot across the observed frequency range.

- **Conversion Metadata**: Information tracked during pipeline execution
  including discovered time steps, frequency coverage per time step, spatial
  grid reference, files processed, and errors encountered.

---

## Review & Acceptance Checklist

_GATE: Automated checks run during main() execution_

### Content Quality

- [x] No implementation details (languages, frameworks, APIs) - _Note: User
      specified typer and prefect as required tools, which is acceptable_
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

_Updated by main() during processing_

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (5 clarifications resolved)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---

## Dependencies & Assumptions

### Dependencies

- Existing `fits_to_zarr_xradio.py` module provides core conversion logic
- Current library dependencies: astropy, xarray, xradio, numpy, zarr, dask
- User-specified additions: typer (CLI framework), prefect (optional pipeline
  orchestration layer)

### Assumptions

- FITS files follow standard OVRO-LWA naming convention:
  `YYYYMMDD_HHMMSS_<SB>MHz_averaged_*-I-image.fits`
- Users have write permissions to output directories
- Input FITS files are well-formed and contain required header keywords (or can
  be auto-corrected)
- Spatial grids (l, m) are consistent across all observations in a dataset
- Users understand basic command-line interface usage
- Python environment has sufficient memory to load and process individual time
  step datasets
