# GitHub Issues: Zarr Chunk Optimization Documentation Suite

This file contains the full issue bodies ready for `gh issue create`. Each issue
is self-contained and designed to be assignable to an AI agent for documentation
generation.

All issues should use labels: `documentation`, `chunking`

---

## Issue 1

**Title:** Docs: Zarr Chunking Fundamentals for Radio Astronomy Data

**Body:**

## Context

This is part of a 7-issue documentation suite on Zarr chunk optimization for
cloud-based object stores. This issue covers the **foundational concepts** that
all other documentation builds upon. It should be written as a standalone page
that can be included in the project's documentation site (PR #95).

**Research document:** `.agents/research-zarr-chunk-optimization.md` **Existing
docs:** `docs/open_dataset.md`

## Scope

Write a documentation page explaining what Zarr chunks are, how they map to
cloud objects, and why chunk size matters specifically for OVRO-LWA radio
astronomy data.

## Required Content

### 1. What is a Zarr Chunk?

- Visual diagram showing how a multidimensional array is divided into chunks
- How chunks map to individual files (local) or objects (cloud storage)
- Relationship between chunk shape, chunk size (bytes), and number of chunks

### 2. Chunks and Cloud Object Stores

- One chunk = one HTTP GET request to S3/GCS/OSN
- Latency per request (~50-100ms) and why many small chunks are expensive
- Throughput characteristics of cloud storage (parallel GETs)
- Why the optimal chunk size for cloud is different from local storage

### 3. The OVRO-LWA Data Model

- 5 dimensions: time, frequency, polarization, l, m (direction cosines)
- Typical dimension sizes (see table in research doc)
- Required variables: `SKY` (float32), optional `BEAM` (float32)
- How data size scales: a single 4096x4096 frame at float32 = 64 MB

### 4. The Chunk Size Sweet Spot

- Industry consensus: 10-100 MB compressed per chunk
- Below 1 MB: HTTP overhead dominates
- Above 500 MB: wastes bandwidth when accessing subsets
- Worked example: chunk_lm=1024 on 4096x4096 image = 16 spatial chunks x 4 MB
  each

### 5. Key Concepts Glossary

- Chunk shape, chunk size, compression ratio, consolidated metadata, chunk
  alignment, access pattern, dask task graph

## External References to Incorporate

- [Zarr Performance User Guide](https://zarr.readthedocs.io/en/latest/user-guide/performance/)
- [ESIP Cloud Computing Optimization](https://esipfed.github.io/cloud-computing-cluster/resources-for-optimization.html)
- [Xarray Intro to Zarr Tutorial](https://tutorial.xarray.dev/intermediate/intro-to-zarr.html)

## Codebase References

- `src/ovro_lwa_portal/fits_to_zarr_xradio.py:315-316` -- where chunks are
  applied
- `src/ovro_lwa_portal/io.py` -- open_dataset() with chunks parameter
- `pyproject.toml:26` -- zarr v2 pin

## Output

- A markdown file at `docs/chunking-fundamentals.md`
- Include diagrams (ASCII or Mermaid) where helpful
- Cross-reference to other docs in this series

## Acceptance Criteria

- [ ] Explains chunks conceptually with visual aids
- [ ] Covers cloud vs. local storage differences
- [ ] Uses OVRO-LWA specific examples (not generic)
- [ ] Includes worked byte-size calculations
- [ ] References external sources with links
- [ ] Follows existing documentation style in `docs/open_dataset.md`

---

## Issue 2

**Title:** Docs: Understanding the OVRO-LWA Chunking Pipeline (Write Path)

**Body:**

## Context

This is part of a 7-issue documentation suite on Zarr chunk optimization for
cloud-based object stores. This issue covers the **write path** -- how the
OVRO-LWA ingest pipeline creates chunked Zarr stores from FITS files.

**Research document:** `.agents/research-zarr-chunk-optimization.md` **Existing
docs:** `docs/open_dataset.md`

## Scope

Document how the FITS-to-Zarr ingest pipeline writes chunks, what defaults are
used, what configuration options exist, and how to inspect the resulting chunk
configuration.

## Required Content

### 1. The Ingest Pipeline Flow

- Step-by-step: FITS files -> `fix_fits_headers()` -> `_load_for_combine()` ->
  `_combine_time_step()` -> `write_image()` -> Zarr store
- Where chunking decisions happen in this flow
- Visual diagram of the pipeline stages

### 2. The `chunk_lm` Parameter

- What it controls: chunk size for l and m spatial dimensions
- Default value: 1024 (meaning 1024x1024 spatial tiles)
- How to set it via CLI: `ovro-ingest --chunk-lm 512`
- How to set it via Python API: `ConversionConfig(chunk_lm=512)`
- Setting `chunk_lm=0` disables spatial chunking (creates one giant chunk per
  spatial plane)

### 3. What Happens to Other Dimensions

- Time, frequency, and polarization are NOT explicitly chunked at write time
- These dimensions follow whatever xradio/Zarr defaults apply
- Implications: each (time, frequency, polarization) combination has its own set
  of spatial chunks

### 4. Compression at Write Time

- Current state: no explicit compression configuration in the codebase
- The `xds[v].encoding = {}` pattern at lines 313, 416, 472 clears encodings
- What xradio's `write_image()` does with compression under the hood
- How to add explicit compression (showing the encoding dict pattern)

### 5. The Append Mode and Chunk Consistency

- How `_write_or_append_zarr()` works (read existing + concat + rewrite)
- Why chunk shapes must be consistent across appended time steps
- Risks of changing `chunk_lm` between appends

### 6. Inspecting Chunk Metadata

- How to read `.zarray` from a Zarr store to verify chunk configuration:
  ```python
  import json
  from pathlib import Path
  meta = json.loads((Path("store.zarr/SKY/.zarray")).read_text())
  print(f"Chunk shape: {meta['chunks']}")
  print(f"Compressor: {meta['compressor']}")
  print(f"Dtype: {meta['dtype']}")
  ```
- Using `zarr.open()` to inspect metadata programmatically
- Calculating actual compressed chunk sizes from file system

## Codebase References

- `src/ovro_lwa_portal/fits_to_zarr_xradio.py:241-330` -- `_load_for_combine()`
  with chunk application
- `src/ovro_lwa_portal/fits_to_zarr_xradio.py:445-495` --
  `_write_or_append_zarr()`
- `src/ovro_lwa_portal/fits_to_zarr_xradio.py:498-598` --
  `convert_fits_dir_to_zarr()`
- `src/ovro_lwa_portal/ingest/core.py:43-58` -- `ConversionConfig`
- `src/ovro_lwa_portal/ingest/cli.py` -- CLI `--chunk-lm` parameter

## Output

- A markdown file at `docs/chunking-write-path.md`
- Include code snippets showing key pipeline stages
- Include output examples from inspecting real .zarray metadata

## Acceptance Criteria

- [ ] Traces the complete write path from FITS to Zarr
- [ ] Documents all configuration options with examples
- [ ] Explains the encoding clearing pattern and its implications
- [ ] Shows how to inspect chunk metadata after writing
- [ ] Covers append mode behavior and consistency requirements
- [ ] References specific file:line locations in the codebase

---

## Issue 3

**Title:** Docs: Optimizing Read-Time Chunk Configuration

**Body:**

## Context

This is part of a 7-issue documentation suite on Zarr chunk optimization for
cloud-based object stores. This issue covers the **read path** -- how to
configure the `chunks=` parameter in `open_dataset()` for optimal performance
based on access patterns.

**Research document:** `.agents/research-zarr-chunk-optimization.md` **Existing
docs:** `docs/open_dataset.md` (lines 93-137 and 252-286)

## Scope

Guide users on choosing the right `chunks=` parameter for their specific
analysis workflow, with emphasis on cloud storage access.

## Required Content

### 1. How `open_dataset(chunks=...)` Works

- The three modes: `"auto"` (default), explicit dict, `None`
- How xarray/dask "auto" works: considers available memory and on-disk chunk
  shape
- When "auto" is good enough vs. when explicit configuration helps
- How `chunks=None` loads everything into memory (only for small datasets)

### 2. Access Pattern Recipes (Detailed)

For each recipe, provide: the chunk configuration, a worked example with
OVRO-LWA data, an explanation of _why_ these values were chosen, and the
expected number of HTTP requests for a typical operation.

**Recipe A: Full Spatial Images (Snapshot Analysis)**

```python
chunks={"time": 1, "frequency": 1, "l": 1024, "m": 1024}
```

- Use case: plotting images, spatial statistics, source detection
- Why: aligns with on-disk chunks, one (t, f) = few spatial GETs

**Recipe B: Time Series at a Point**

```python
chunks={"time": -1, "frequency": 1, "l": 256, "m": 256}
```

- Use case: light curves, transient monitoring
- Why: loads all times, small spatial tiles minimize waste for point access

**Recipe C: Spectral Analysis**

```python
chunks={"time": 1, "frequency": -1, "l": 256, "m": 256}
```

- Use case: spectral index maps, broadband averaging
- Why: loads all frequencies for spatial region

**Recipe D: Full Dataset Processing**

```python
chunks={"time": 5, "frequency": 10, "l": 512, "m": 512}
```

- Use case: batch processing, statistics across all dimensions
- Why: balanced chunks for mixed access

### 3. Chunk Alignment

- Why read-time chunks should be multiples of write-time chunks
- What happens with misaligned chunks (partial reads, unnecessary data transfer)
- Example: reading 512x512 from 1024x1024 on-disk chunks still downloads the
  full 1024x1024
- How to check on-disk chunk shape before choosing read chunks

### 4. Memory Considerations

- Dask task graph size: too many small chunks = too many tasks = scheduler
  overhead
- Rule of thumb: aim for 100-10,000 tasks per computation
- How to estimate memory per chunk: `chunk_shape * dtype_bytes * n_variables`
- Using `ds.chunks` to inspect effective chunk configuration after loading

### 5. Decision Flowchart

- Simple text/ASCII flowchart guiding users from "What do I want to do?" to a
  specific chunk config

## Codebase References

- `src/ovro_lwa_portal/io.py` -- `open_dataset()` function signature and chunks
  handling
- `docs/open_dataset.md:93-137` -- Existing chunking parameter docs
- `docs/open_dataset.md:252-286` -- Existing performance tips
- `src/ovro_lwa_portal/accessor.py` -- Methods that call `.compute()` (where
  chunks affect performance)

## Output

- A markdown file at `docs/chunking-read-optimization.md`
- Include worked examples with expected HTTP request counts
- Include the decision flowchart

## Acceptance Criteria

- [ ] Provides 4+ concrete access pattern recipes with full explanations
- [ ] Explains chunk alignment with worked examples
- [ ] Covers memory/task-count considerations
- [ ] Includes decision flowchart
- [ ] Builds on existing `docs/open_dataset.md` performance section (don't
      duplicate)
- [ ] All examples use OVRO-LWA specific data model (not generic Zarr)

---

## Issue 4

**Title:** Docs: Compression Strategies for Cloud-Stored Zarr Data

**Body:**

## Context

This is part of a 7-issue documentation suite on Zarr chunk optimization for
cloud-based object stores. This issue covers **compression** -- how it interacts
with chunking, what codecs to use, and how to configure compression for OVRO-LWA
data.

**Research document:** `.agents/research-zarr-chunk-optimization.md`

## Scope

Document compression options available in the Zarr v2 / numcodecs ecosystem, how
they interact with chunk sizing for cloud performance, and provide specific
recommendations for OVRO-LWA float32 radio astronomy data.

## Required Content

### 1. Why Compression Matters for Cloud Storage

- Reduces data transfer size (lower egress costs, faster downloads)
- Compressed chunk size is what matters for cloud performance, not uncompressed
- CPU cost of decompression: tradeoff between network savings and compute
  overhead
- Example: 4 MB uncompressed chunk with 4:1 compression = 1 MB transfer

### 2. Available Codecs (numcodecs)

- **Blosc:** Meta-compressor wrapping internal codecs + optional shuffle
  - cname options: 'zstd', 'lz4', 'lz4hc', 'zlib', 'snappy'
  - shuffle options: NOSHUFFLE, SHUFFLE, BITSHUFFLE
  - clevel: 1-9 (compression level)
- **Zstd (standalone):** High ratio, good speed
- **LZ4:** Very fast compression/decompression, moderate ratio
- **GZip/Zlib:** Maximum compatibility, slower
- Comparison table: codec vs. speed vs. ratio for float32 data

### 3. Recommended Configuration for OVRO-LWA

- Primary recommendation:
  `Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)`
  - Why: Zstd gives excellent compression ratio; BITSHUFFLE is specifically
    effective for float32 arrays where bit patterns repeat
  - Expected compression ratio for radio astronomy images: 2-5x (data-dependent)
- Alternative: `Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)` for faster
  decompression at slightly lower ratio
- When to use no compression: if data is already compressed or highly random

### 4. Configuring Compression at Write Time

```python
from numcodecs import Blosc

# Via xarray encoding dict
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
encoding = {
    "SKY": {"compressor": compressor},
    "BEAM": {"compressor": compressor},
}
ds.to_zarr("store.zarr", encoding=encoding)
```

- How this could be integrated into the ingest pipeline
- The encoding clearing pattern in `fits_to_zarr_xradio.py` and its implications

### 5. Filters vs. Compressors

- Delta coding: stores differences between values (good for slowly-varying data)
- Quantize: reduces precision for higher compression (lossy!)
- When filters help vs. when they add overhead
- Stacking filters + compressors in numcodecs

### 6. `write_empty_chunks=False`

- What it does: skips writing chunks that are all-fill-value
- When it helps: sparse OVRO-LWA data with many NaN/zero frames
- How to enable: `ds.to_zarr("store.zarr", write_empty_chunks=False)`

### 7. Measuring Compression Effectiveness

- How to check actual compressed sizes: `os.path.getsize()` on chunk files
- Compression ratio = uncompressed / compressed
- Impact on cloud read performance: recalculate optimal chunk shape using
  compressed size

## External References to Incorporate

- [Zarr Performance User Guide](https://zarr.readthedocs.io/en/latest/user-guide/performance/)
- [numcodecs documentation](https://numcodecs.readthedocs.io/)
- [HackMD Zarr Chunking Guide](https://hackmd.io/@brivadeneira/rkqm_XYHgg)
- [MarkTechPost Zarr Guide](https://www.marktechpost.com/2025/09/16/a-coding-guide-to-implement-zarr-for-large-scale-data-chunking-compression-indexing-and-visualization-techniques/)

## Codebase References

- `pyproject.toml:27` -- numcodecs pin (`>=0.15,<0.16`)
- `pyproject.toml:26` -- zarr v2 pin
- `src/ovro_lwa_portal/fits_to_zarr_xradio.py:312-316` -- encoding clearing +
  chunking

## Output

- A markdown file at `docs/chunking-compression.md`
- Include codec comparison table
- Include code examples for configuring compression

## Acceptance Criteria

- [ ] Explains compression-chunking interaction for cloud performance
- [ ] Compares available codecs with performance characteristics
- [ ] Provides specific recommendation for OVRO-LWA float32 data
- [ ] Shows how to configure compression in code
- [ ] Covers write_empty_chunks for sparse data
- [ ] Includes measurement/verification instructions

---

## Issue 5

**Title:** Docs: Benchmarking Zarr Read Performance

**Body:**

## Context

This is part of a 7-issue documentation suite on Zarr chunk optimization for
cloud-based object stores. This issue covers **benchmarking** -- practical
techniques for measuring and comparing Zarr read performance under different
configurations.

**Research document:** `.agents/research-zarr-chunk-optimization.md`

## Scope

Provide a practical benchmarking guide with methodology, scripts, and
interpretation guidance for measuring Zarr chunk read performance on both local
and cloud stores.

## Required Content

### 1. Why Benchmarking is Necessary

- No universal optimal chunk size -- depends on data, access patterns, and
  infrastructure
- Published recommendations are starting points, not guarantees
- Cloud performance varies by provider, region, time of day, and concurrency

### 2. Key Metrics to Measure

- **Wall-clock time:** Total time for a representative operation
- **Time to first byte (TTFB):** Measures initial access latency
- **Number of HTTP requests:** Proxy for cloud overhead (inspect via fsspec
  logging)
- **Effective throughput:** Data bytes received / wall time
- **Chunk utilization ratio:** Bytes actually used / total bytes transferred
- **Dask task count:** Number of tasks in the computation graph

### 3. Benchmarking Methodology

1. Define 3-5 representative access patterns (match real user workflows)
2. Create a controlled environment (consistent network, no background jobs)
3. Warm vs. cold cache: run once to warm, then measure; OR clear caches between
   runs
4. Multiple runs per configuration (minimum 5) for statistical reliability
5. Vary one parameter at a time (chunk shape OR compression OR read-time chunks)
6. Record environment: Python versions, network type, instance type if cloud

### 4. Benchmarking Script Template

Complete, runnable Python script that:

- Benchmarks spatial access (load one full image)
- Benchmarks time-series access (load all times at one pixel)
- Benchmarks spectral access (load all frequencies at one location)
- Compares multiple chunk configurations
- Outputs a results table with mean/std/min/max
- Uses `time.perf_counter()` for accurate timing

```python
"""Zarr chunk benchmarking script for OVRO-LWA data.

Usage:
    python benchmark_chunks.py <zarr_store_path> [--cloud]
"""
import time
import numpy as np
import ovro_lwa_portal

def benchmark(name, fn, n_runs=5):
    """Run a benchmark function multiple times and report stats."""
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"{name}: mean={times.mean():.3f}s std={times.std():.3f}s "
          f"min={times.min():.3f}s max={times.max():.3f}s")
    return times

# ... [full script to be written by agent]
```

### 5. Using Dask Diagnostics

- `dask.distributed.performance_report("report.html")` for detailed profiling
- Dask dashboard for real-time monitoring (localhost:8787)
- Task stream visualization: identifying I/O bottlenecks
- How to interpret the profiling output

### 6. Comparing Configurations

- Template comparison matrix:

| Configuration     | Spatial (s) | Time-series (s) | Spectral (s) | HTTP Requests |
| ----------------- | ----------- | --------------- | ------------ | ------------- |
| Default (auto)    |             |                 |              |               |
| Spatial-optimized |             |                 |              |               |
| Time-optimized    |             |                 |              |               |
| Custom 1          |             |                 |              |               |

### 7. Interpreting Results

- What "good" looks like: effective throughput > 50 MB/s local, > 10 MB/s cloud
- When to stop optimizing: diminishing returns past 2x improvement
- How to translate benchmark results into production chunk configuration

### 8. Common Pitfalls

- OS page cache masking I/O time (use
  `purge`/`sync; echo 3 > /proc/sys/vm/drop_caches`)
- fsspec caching between runs (disable or clear explicitly)
- Network variability on shared cloud instances
- Not testing with realistic data sizes

## External References to Incorporate

- [ESIP Cloud Optimization](https://esipfed.github.io/cloud-computing-cluster/resources-for-optimization.html)
  -- 63x improvement from chunk alignment
- [Zarr Performance Guide](https://zarr.readthedocs.io/en/latest/user-guide/performance/)

## Codebase References

- `src/ovro_lwa_portal/io.py` -- open_dataset() (the function being benchmarked)
- `src/ovro_lwa_portal/accessor.py` -- radport methods that trigger computation

## Output

- A markdown file at `docs/chunking-benchmarking.md`
- Include the complete benchmarking script (can also be a standalone .py file in
  `docs/` or `examples/`)
- Include example output showing what results look like

## Acceptance Criteria

- [ ] Provides complete, runnable benchmarking script
- [ ] Covers both local and cloud benchmarking
- [ ] Explains methodology with statistical rigor
- [ ] Shows how to use Dask diagnostics
- [ ] Includes comparison matrix template
- [ ] Covers common pitfalls and how to avoid them

---

## Issue 6

**Title:** Docs: Cloud Storage Configuration and Optimization for Zarr Access

**Body:**

## Context

This is part of a 7-issue documentation suite on Zarr chunk optimization for
cloud-based object stores. This issue covers **cloud-specific configuration** --
how to set up and optimize cloud storage access for Zarr data.

**Research document:** `.agents/research-zarr-chunk-optimization.md` **Existing
docs:** `docs/open_dataset.md` (lines 150-182 cover basic cloud setup)

## Scope

Document cloud-specific considerations for optimal Zarr access, including fsspec
configuration, consolidated metadata, caching strategies, and provider-specific
settings (S3, GCS, OSN).

## Required Content

### 1. How Cloud Zarr Access Works

- fsspec/s3fs/gcsfs as filesystem abstraction layer
- Each chunk = one GET request (no range requests within chunks in Zarr v2)
- Metadata requests: `.zmetadata` (consolidated) vs. per-array
  `.zarray`/`.zattrs`
- How `xr.open_zarr()` constructs the fsspec mapper

### 2. Consolidated Metadata

- What it is: single `.zmetadata` file containing all array metadata
- Why it matters: without it, opening a store requires one request per array +
  one per group
- How to check if a store has consolidated metadata
- How to create it: `zarr.convenience.consolidate_metadata("store.zarr")`
- `xr.open_zarr(consolidated=True)` (the default) -- fails gracefully if not
  present
- How the ingest pipeline handles consolidated metadata

### 3. Provider-Specific Configuration

**AWS S3:**

```python
ds = ovro_lwa_portal.open_dataset(
    "s3://bucket/store.zarr",
    storage_options={
        "key": "...",
        "secret": "...",
        "client_kwargs": {"region_name": "us-west-2"},
    }
)
```

- Authentication: env vars, IAM roles, explicit credentials
- Region selection and its impact on latency

**Google Cloud Storage:**

```python
ds = ovro_lwa_portal.open_dataset(
    "gs://bucket/store.zarr",
    storage_options={"token": "/path/to/credentials.json"}
)
```

**OSN (Open Storage Network):**

```python
ds = ovro_lwa_portal.open_dataset(
    "s3://bucket/store.zarr",
    storage_options={
        "key": "...",
        "secret": "...",
        "client_kwargs": {
            "endpoint_url": "https://caltech1.osn.mghpcc.org"
        },
    }
)
```

- OSN endpoints and how they differ from AWS S3
- HTTPS vs S3 protocol access performance
- Using `resolve_source()` to debug DOI-to-S3 resolution

### 4. Caching Strategies

- **fsspec `simplecache`:** Transparent caching of individual chunk files
  ```python
  import fsspec
  fs = fsspec.filesystem(
      "simplecache",
      target_protocol="s3",
      target_options={"key": "...", "secret": "..."},
      cache_storage="/tmp/zarr_cache",
  )
  ```
- **fsspec `filecache`:** Persistent local cache that survives restarts
- **Full local download:** When to just `aws s3 sync` or `rclone` the whole
  store
- Decision guide: when each caching strategy is appropriate

### 5. Concurrent Access Tuning

- How Dask parallelizes chunk fetches across workers
- s3fs `max_concurrency` parameter
- fsspec retry and timeout configuration
- Balancing concurrency vs. rate limiting (cloud provider throttling)

### 6. Cost Considerations

- GET request pricing (AWS: $0.0004/1000 requests)
- Data transfer/egress costs
- How chunk count affects request costs
- Strategies to minimize costs: caching, larger chunks, local processing

### 7. Troubleshooting Cloud Access

- "Connection timeout" -- check endpoint, credentials, network
- "403 Forbidden" -- check permissions, bucket policy
- "NoSuchBucket" -- verify bucket name and endpoint
- Slow reads -- check chunk sizes, region alignment, caching
- Using `resolve_source()` for debugging DOI resolution

## Codebase References

- `src/ovro_lwa_portal/io.py` -- fsspec mapper construction, storage_options
  handling, resolve_source()
- `src/ovro_lwa_portal/io.py:284-371` -- `_check_remote_access()` pre-check
- `docs/open_dataset.md:150-182` -- Existing cloud storage docs (don't
  duplicate, extend)

## Output

- A markdown file at `docs/chunking-cloud-storage.md`
- Provider-specific configuration examples
- Caching strategy decision guide

## Acceptance Criteria

- [ ] Covers S3, GCS, and OSN with provider-specific examples
- [ ] Explains consolidated metadata with verification steps
- [ ] Documents caching strategies with decision guide
- [ ] Includes concurrency and cost considerations
- [ ] Provides troubleshooting section for common cloud issues
- [ ] Builds on (not duplicates) existing cloud docs in open_dataset.md

---

## Issue 7

**Title:** Docs: Chunk Optimization Decision Guide and Quick Reference

**Body:**

## Context

This is the **final issue** in a 7-issue documentation suite on Zarr chunk
optimization for cloud-based object stores. It synthesizes all other
documentation into an actionable decision guide and quick reference card.

**Research document:** `.agents/research-zarr-chunk-optimization.md`
**Prerequisite docs:** Issues 1-6 should be completed first (this issue
cross-references them)

## Scope

Create a quick-reference decision guide that helps users immediately find the
right chunk configuration for their use case, plus a troubleshooting FAQ.

## Required Content

### 1. Decision Flowchart

ASCII or Mermaid flowchart:

```
Start: "What's your data size?"
├── < 1 GB → Use chunks=None (load into memory)
├── 1-10 GB → Use chunks="auto" (good enough)
└── > 10 GB → "Where is your data?"
    ├── Local → chunks="auto" works well
    └── Cloud → "What's your primary access pattern?"
        ├── Full images → Recipe A
        ├── Time series → Recipe B
        ├── Spectral → Recipe C
        └── Mixed → Recipe D
```

### 2. Quick Reference Table

| Scenario                | Write `chunk_lm` | Read `chunks=`                                      | Expected Perf       | Notes                    |
| ----------------------- | ---------------- | --------------------------------------------------- | ------------------- | ------------------------ |
| Small local dataset     | any              | `None`                                              | Instant             | Loads to RAM             |
| Large local dataset     | 1024             | `"auto"`                                            | Good                | Dask handles it          |
| Cloud: spatial analysis | 1024             | `{"time": 1, "frequency": 1, "l": 1024, "m": 1024}` | ~4 chunks per image | Aligns with write chunks |
| Cloud: time series      | 512              | `{"time": -1, "frequency": 1, "l": 512, "m": 512}`  | Many time chunks    | Minimize spatial waste   |
| Cloud: spectral         | 1024             | `{"time": 1, "frequency": -1, "l": 256, "m": 256}`  | All freqs at once   | Small spatial tiles      |
| Cloud: batch processing | 1024             | `{"time": 5, "frequency": 10, "l": 512, "m": 512}`  | Balanced            | Good general purpose     |

### 3. Anti-Patterns (What NOT to Do)

- Using `chunks=None` on large remote datasets (OOM or multi-GB download)
- Using very small chunks (< 1 MB) on cloud stores (HTTP overhead)
- Misaligned read chunks (e.g., 512 from 1024 on-disk -- still downloads full
  chunk)
- No consolidated metadata on remote stores (N+1 metadata requests)
- Too many Dask tasks (>100,000) from tiny chunks
- Changing `chunk_lm` between ingest append operations

### 4. Troubleshooting FAQ

**Q: My cloud reads are slow. What should I check?**

1. Check chunk sizes: `ds.chunks` -- are they too small?
2. Check chunk alignment: do read chunks match write chunks?
3. Check consolidated metadata: does `.zmetadata` exist?
4. Check network: ping latency to cloud endpoint
5. Check compression: are chunks < 1 MB after compression?
6. Try caching: use fsspec simplecache for repeated access

**Q: I'm running out of memory.**

1. Reduce chunk sizes in the largest dimensions
2. Process in smaller batches with explicit `.compute()` calls
3. Use Dask distributed scheduler for larger-than-memory processing
4. Avoid `chunks=None` on large datasets

**Q: Dask is creating too many tasks and the scheduler is slow.**

1. Increase chunk sizes (fewer, larger chunks)
2. Use `ds.chunk({"dim": larger_value})` to consolidate
3. Avoid operations that expand the task graph (e.g., rolling windows with small
   chunks)

**Q: How do I rechunk an existing Zarr store?**

- Using `rechunker` library for efficient out-of-core rechunking
- Using `ds.chunk({...}).to_zarr("new_store.zarr")` for simple cases
- When to rechunk vs. when to just tune read-time chunks

### 5. Cross-Reference Links

Link to every other document in the series:

- Chunking Fundamentals (Issue 1)
- Write Path Pipeline (Issue 2)
- Read-Time Optimization (Issue 3)
- Compression Strategies (Issue 4)
- Benchmarking Guide (Issue 5)
- Cloud Storage Configuration (Issue 6)

### 6. One-Page Cheat Sheet

Condensed reference that fits in a single printed page:

- Key numbers: 10-100 MB compressed sweet spot, 50-100ms S3 latency
- Default config: `chunk_lm=1024`, `chunks="auto"`
- Cloud essentials: consolidated metadata, aligned chunks, caching
- Debug commands: check `.zarray`, inspect `ds.chunks`, count Dask tasks

## Output

- A markdown file at `docs/chunking-decision-guide.md`
- Include flowchart (Mermaid or ASCII)
- Include the one-page cheat sheet

## Acceptance Criteria

- [ ] Decision flowchart covers all common scenarios
- [ ] Quick reference table is complete and accurate
- [ ] Anti-patterns section warns about common mistakes
- [ ] Troubleshooting FAQ covers the top 5 user issues
- [ ] Cross-references all other docs in the series
- [ ] Cheat sheet is genuinely concise (one printable page)
- [ ] Standalone: a user can use this doc without reading the others
