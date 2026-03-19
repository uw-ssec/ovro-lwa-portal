# Compression Strategies for Cloud-Stored Zarr Data

Compression reduces the number of bytes transferred from cloud storage, directly
improving read performance and reducing bandwidth costs. For OVRO-LWA datasets
stored on S3-compatible object stores like OSN, the compressed chunk size
determines network latency, not the uncompressed chunk size.

This guide explains compression options available in the Zarr v2 ecosystem, how
to configure them for OVRO-LWA data, and how to measure their effectiveness.

## Why Compression Matters for Cloud Storage

Every Zarr chunk maps to a single HTTP GET request on cloud storage. The number
of bytes transferred directly determines transfer time, while the number of
requests determines latency overhead. Compression attacks the transfer time
component by reducing payload size.

Consider the tradeoff: a 4 MB uncompressed chunk takes 40 milliseconds to
transfer on a 100 MB/s connection, while a 1 MB compressed chunk takes only 10
milliseconds. (This 4:1 ratio is illustrative — actual compression depends on
the compressor and data characteristics.) The HTTP request overhead remains
constant at 50–100 milliseconds per request, but compression saves 30
milliseconds per chunk in transfer time. Over thousands of chunks, this saving
compounds significantly.

Compression adds CPU cost for decompression on the client side. Modern codecs
like Zstd and LZ4 decompress at 500–2000 MB/s on typical CPUs, making
decompression time negligible compared to network transfer time for
cloud-resident data. The network savings almost always justify the compute
overhead for remote access patterns.

**What matters for cloud performance:** The compressed chunk size determines how
much data must be transferred per HTTP GET request. A 4 MB uncompressed chunk
might compress to 1 MB with typical settings, falling within the 1–10 MB target
range for cloud-optimized chunk sizes. Without compression, that same logical
chunk would require transferring the full 4 MB, potentially pushing into the
"too large" category where bandwidth is wasted on unused data.

This interaction between compression and chunking means that uncompressed chunk
size and compressed chunk size must both be considered when optimizing for cloud
access. Start with the uncompressed chunk size in the 10–100 MB range from
[Chunking Fundamentals](chunking-fundamentals.md), then apply compression to
reduce the actual transfer size by a factor of 2–6x depending on the codec and
data characteristics.

## Available Codecs (numcodecs)

The OVRO-LWA project pins `numcodecs>=0.15,<0.16` (pyproject.toml:27), which
provides the compression codec registry for Zarr v2. All codecs in this section
are available through the `numcodecs` package.

### Blosc: Meta-Compressor with Shuffle

Blosc is a meta-compressor that wraps other compression codecs and adds optional
byte shuffle filters for improved compression of numerical arrays. It is the
recommended choice for scientific data due to its flexibility and performance.

**Configuration parameters:**

- `cname`: Internal compression codec. Options: `'zstd'`, `'lz4'`, `'lz4hc'`,
  `'zlib'`, `'snappy'`
- `clevel`: Compression level (1–9). Higher values increase compression ratio at
  the cost of slower compression time.
- `shuffle`: Byte shuffle mode. Options:
  - `Blosc.NOSHUFFLE` (0): No shuffle, compress bytes as-is
  - `Blosc.SHUFFLE` (1): Byte shuffle, groups bytes of the same significance
    across array elements
  - `Blosc.BITSHUFFLE` (2): Bit shuffle, groups bits of the same significance
    across array elements

**Shuffle modes explained:**

- **NOSHUFFLE**: Compress the raw byte stream without rearrangement. Use when
  data has no spatial structure or is already delta-encoded.
- **SHUFFLE**: Rearrange bytes so that the least significant bytes from all
  elements are grouped together, followed by the next significant bytes, etc.
  This grouping improves compression when adjacent array elements have similar
  magnitudes.
- **BITSHUFFLE**: Like SHUFFLE but operates at the bit level, grouping the least
  significant bits across all elements together. Particularly effective for
  float32 arrays where sign, exponent, and mantissa bits have different entropy
  characteristics.

For float32 sky brightness data, BITSHUFFLE typically outperforms SHUFFLE
because floating-point representations benefit from bit-level alignment of
exponent and mantissa fields.

### Zstd (Standalone)

Zstandard (zstd) provides high compression ratios with fast decompression.
Modern CPUs decompress zstd at 400–600 MB/s single-threaded, making it suitable
for latency-sensitive cloud access.

**Configuration:**

```python
from numcodecs import Zstd

compressor = Zstd(level=3)  # level 1-22; 3 is a balanced default
```

Use standalone Zstd when you do not need shuffle filters. For scientific arrays,
Blosc(cname='zstd', shuffle=Blosc.BITSHUFFLE) usually outperforms standalone
Zstd due to the shuffle preprocessing.

### LZ4

LZ4 prioritizes decompression speed over compression ratio. It decompresses at
2000+ MB/s, faster than network transfer rates in most scenarios. Use LZ4 when
read latency is critical and you can tolerate larger chunk sizes.

**Configuration:**

```python
from numcodecs import LZ4

compressor = LZ4(acceleration=1)  # acceleration 1-65537; higher = faster, lower ratio
```

For scientific workflows, Blosc(cname='lz4', shuffle=Blosc.SHUFFLE) provides
better compression than standalone LZ4 on numerical arrays while retaining fast
decompression.

### GZip / Zlib

GZip (RFC 1952) and zlib (RFC 1950) provide maximum compatibility across tools
and languages. Compression ratio is comparable to zstd at similar levels, but
decompression is 2–4x slower (100–200 MB/s).

**Configuration:**

```python
from numcodecs import GZip

compressor = GZip(level=6)  # level 1-9; 6 is default
```

Use GZip only when compatibility with non-Python tools is required. For pure
Python/xarray workflows, Blosc(cname='zstd') provides better performance.

### Codec Comparison Table

| Codec                            | Compression Ratio  | Decompression Speed | Recommended Use Case                   |
| -------------------------------- | ------------------ | ------------------- | -------------------------------------- |
| Blosc(cname='zstd', BITSHUFFLE)  | High (4–6x)        | Fast (400–600 MB/s) | **Default for float32 astronomy data** |
| Blosc(cname='lz4', SHUFFLE)      | Medium (2–4x)      | Very fast (2+ GB/s) | Low-latency read-heavy workloads       |
| Blosc(cname='lz4hc', BITSHUFFLE) | Medium-high (3–5x) | Fast (2+ GB/s)      | Balanced compression and speed         |
| Zstd (standalone)                | High (4–6x)        | Fast (400–600 MB/s) | Non-array data, text, metadata         |
| LZ4 (standalone)                 | Low-medium (2–3x)  | Very fast (2+ GB/s) | Minimal CPU overhead required          |
| GZip / Zlib                      | High (4–6x)        | Slow (100–200 MB/s) | Cross-tool compatibility required      |

**Note on compression ratios:** The ratios shown are illustrative for typical
radio astronomy float32 image data. Actual ratios depend on data
characteristics, including sky structure, noise levels, and dynamic range.
Always measure compression on representative data samples for production
planning.

## Recommended Configuration for OVRO-LWA

For OVRO-LWA float32 sky brightness arrays, the recommended compressor is:

```python
from numcodecs import Blosc

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
```

**Why this configuration:**

- **zstd** provides excellent compression ratio (typically 3–5x for astronomical
  images) with fast decompression (400–600 MB/s), balancing storage savings and
  read performance.
- **clevel=3** offers a good tradeoff between compression time and ratio. Higher
  levels (5–7) improve ratio by 5–10% but slow compression significantly. Lower
  levels (1–2) sacrifice 10–20% ratio for faster writes.
- **BITSHUFFLE** is specifically effective for float32 data where adjacent
  values share similar bit patterns. Radio astronomy images have spatial
  coherence and dynamic range structure that benefit from bit-level shuffling.

**Alternative: Faster decompression at the cost of larger chunks**

If read latency is critical and storage/bandwidth constraints are relaxed, use:

```python
compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
```

This configuration decompresses 3–5x faster than zstd but achieves 20–40% lower
compression ratios. The resulting chunks will be larger, requiring more
bandwidth per request.

**When to use no compression:**

Set `compressor=None` if data is already compressed (e.g., from upstream
processing) or consists of random noise with no compressible structure. For
OVRO-LWA sky images, some compression is almost always beneficial due to the
presence of empty sky regions, point sources, and instrumental structure.

**Expected compression ratios for radio astronomy images:**

Typical compression ratios for float32 sky brightness data fall in the range of
2–5x, depending on:

- **Sky structure:** Images dominated by empty sky (near-zero values) compress
  better than crowded fields with many sources.
- **Dynamic range:** High dynamic range (bright point sources on faint
  background) compresses well because many pixels cluster near background
  levels.
- **Noise characteristics:** Thermal noise is less compressible than structured
  signal, so signal-dominated images achieve higher ratios.

These ratios are typical for the domain based on published literature for
comparable radio telescope data, not verified measurements on OVRO-LWA data.
Measure compression ratios on your specific datasets to confirm expected
behavior.

**Full encoding configuration example:**

```python
from numcodecs import Blosc
import xarray as xr

# Configure compressor
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Apply to all data variables
ds = xr.open_dataset("input.nc")
encoding = {
    var: {"compressor": compressor}
    for var in ds.data_vars
}

# Write to Zarr with compression
ds.to_zarr("output.zarr", encoding=encoding)
```

For the OVRO-LWA ingest pipeline, this encoding should be applied in
`_load_for_combine()` or `_write_or_append_zarr()` after the encoding clearing
step but before the `write_image()` call.

## Configuring Compression at Write Time

The OVRO-LWA ingest pipeline does not currently configure explicit compression
settings. Compression behavior is determined by xradio's `write_image()`
defaults, which delegate to xarray's `.to_zarr()`, which in turn applies Zarr's
built-in Blosc compressor with default settings.

### The Encoding Clearing Pattern

The pipeline explicitly clears variable encodings at three points in
src/ovro_lwa_portal/fits_to_zarr_xradio.py:

1. Line 313: After loading and applying WCS coordinates in `_load_for_combine()`
2. Line 416: After combining time steps in `_combine_time_step()`
3. Line 472: Before concatenating existing data in `_write_or_append_zarr()`

The pattern:

```python
for v in xds.data_vars:
    xds[v].encoding = {}
```

This clearing ensures that no stale encoding metadata (chunk shapes, compression
settings, fill values) is inherited from intermediate processing steps. Any
compression settings xradio applied during `read_image()` are discarded before
writing. The result: xradio's `write_image()` function applies its own defaults
without interference.

**Implication:** Without explicit encoding configuration in the pipeline,
compression settings are applied opaquely by xradio and Zarr defaults. Users
cannot control compression codecs or levels without modifying the source code.

!!! note "Verifying Default Compression"

    The actual compression codec and settings that xradio's `write_image()`
    applies by default are not explicitly documented in xradio's API reference.
    To verify the configuration used for your dataset, inspect the `.zarray`
    metadata file as described in
    [Write Path Pipeline section 6](chunking-write-path.md#inspecting-chunk-metadata).

### Modifying the Ingest Pipeline for Custom Compression

To configure compression explicitly in the ingest pipeline, insert encoding
configuration after the clearing step but before the `write_image()` call.

**Location in code:** src/ovro_lwa_portal/fits_to_zarr_xradio.py:312–316

**Current code:**

```python
# Line 312
for v in xds.data_vars:
    xds[v].encoding = {}

if chunk_lm and {"l", "m"} <= set(xds.dims):
    xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})
```

**Modified code with compression configuration:**

```python
from numcodecs import Blosc

# Clear encoding as before
for v in xds.data_vars:
    xds[v].encoding = {}

# Apply custom compressor
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
for v in xds.data_vars:
    xds[v].encoding = {"compressor": compressor}

# Apply chunking as before
if chunk_lm and {"l", "m"} <= set(xds.dims):
    xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})
```

This pattern sets the compressor for all data variables (SKY, BEAM) before
chunking. The encoding dictionary persists through the `write_image()` call and
is applied during Zarr array creation.

For production deployments, expose the compressor configuration as a parameter
in `ConversionConfig` to allow users to specify compression settings via the CLI
or Python API without modifying source code.

## Filters vs. Compressors

Filters transform data before compression, potentially improving compression
ratios or reducing storage precision. Filters are stacked before the compressor
in the numcodecs pipeline.

### Delta Coding

Delta coding stores the difference between adjacent values rather than the raw
values. This transformation is effective for slowly-varying time series or
monotonically increasing sequences.

**Configuration:**

```python
from numcodecs import Delta, Blosc

filters = [Delta(dtype='<f4')]  # delta filter for float32
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

encoding = {
    "time_series_variable": {
        "filters": filters,
        "compressor": compressor,
    }
}
```

For OVRO-LWA spatial images, delta coding is **not recommended** because sky
brightness varies discontinuously (point sources, edges, noise). Delta coding is
more appropriate for smooth time-domain signals like timestamps or slowly
varying telescope pointing coordinates.

### Quantize (Lossy Compression)

The Quantize filter reduces floating-point precision by rounding values to a
fixed number of significant digits. This is a lossy transformation that
increases compression ratio at the cost of introducing rounding errors.

**Configuration:**

```python
from numcodecs import Quantize, Blosc

# Round to 3 decimal digits
filters = [Quantize(digits=3, dtype='<f4')]
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

encoding = {
    "SKY": {
        "filters": filters,
        "compressor": compressor,
    }
}
```

!!! warning "Lossy Compression May Affect Scientific Measurements"

    Quantization introduces rounding errors that propagate through downstream
    analysis. Photometry, spectral fitting, and other quantitative measurements
    may be affected. Verify that quantization errors are acceptable for your
    science goals before using this filter on observational data. Consider
    applying Quantize only to derived products (e.g., visualization images) or
    intermediate processing steps where precision loss is acceptable.

For OVRO-LWA data, Quantize is **not currently used** in the ingest pipeline.
The default configuration preserves full float32 precision. Filters are not
necessary for achieving good compression ratios on radio astronomy images; Blosc
with BITSHUFFLE alone provides 3–5x compression without data loss.

## write_empty_chunks

The `write_empty_chunks` parameter controls whether Zarr writes chunks that
contain only fill values (NaN, zero, or a user-specified fill value). By
default, Zarr stores all chunks explicitly, even if they contain no valid data.

For OVRO-LWA datasets, sky images may include frames with missing data, flagged
observations, or regions outside the primary beam. If these frames consist
entirely of NaN or zero values, storing them as explicit chunks wastes storage
and network bandwidth.

**Setting write_empty_chunks=False:**

```python
import xarray as xr

ds = xr.open_dataset("input.nc")
ds.to_zarr("output.zarr", write_empty_chunks=False)
```

With `write_empty_chunks=False`, Zarr skips writing chunks that contain only
fill values. When reading, xarray and Zarr automatically reconstruct these
chunks on-the-fly by filling with the appropriate fill value. This optimization
reduces storage footprint and eliminates HTTP GET requests for empty chunks
during read operations.

**When to use write_empty_chunks=False:**

- Sparse datasets with many NaN-filled frames (e.g., flagged data, partial sky
  coverage)
- Time-lapse observations where not all time-frequency points have valid data
- Datasets with large regions outside the field of view or below sensitivity
  thresholds

**When NOT to use write_empty_chunks=False:**

- Dense datasets where most chunks contain valid data; checking for empty chunks
  before writing adds overhead without benefit
- Datasets where distinguishing "no data written" from "data is all NaN" is
  scientifically meaningful; sparse storage treats both cases identically

!!! warning "Use Only for Sparse Data"

    Enable `write_empty_chunks=False` only if you know your data is sparse. For
    dense data, this option adds per-chunk overhead to check fill values before
    writing, with no storage benefit. Inspect your dataset's NaN fraction before
    enabling this optimization.

**Current state in the OVRO-LWA ingest pipeline:**

The ingest pipeline does not currently configure `write_empty_chunks`. The
default behavior stores all chunks explicitly. For typical OVRO-LWA observations
with full time-frequency coverage, this default is appropriate. If your workflow
includes heavy flagging or partial observations, consider adding
`write_empty_chunks=False` to the `write_image()` call in
`_write_or_append_zarr()`.

## Measuring Compression Effectiveness

After converting data with a specific compression configuration, measure the
actual compressed sizes to verify performance.

### Compressed Chunk Size on Disk

Each Zarr chunk is stored as a binary file (local storage) or object (cloud
storage). The file size is the compressed size.

**Measuring a single chunk:**

```python
import os
from pathlib import Path

# Path to a specific chunk (time=0, freq=0, pol=0, l-tile=0, m-tile=0)
chunk_path = Path("store.zarr/SKY/0.0.0.0.0")

if chunk_path.exists():
    compressed_size = chunk_path.stat().st_size

    # Calculate uncompressed size for a 1024×1024 float32 chunk
    uncompressed_size = 1024 * 1024 * 4  # bytes

    ratio = uncompressed_size / compressed_size
    print(f"Compressed size: {compressed_size / 1024:.1f} KB")
    print(f"Uncompressed size: {uncompressed_size / 1024:.1f} KB")
    print(f"Compression ratio: {ratio:.2f}x")
else:
    print("Chunk file not found")
```

**Example output:**

```
Compressed size: 872.3 KB
Uncompressed size: 4096.0 KB
Compression ratio: 4.70x
```

This measurement confirms that a 4 MB uncompressed chunk compressed to ~870 KB,
well within the 1–10 MB target range for cloud-optimized chunk sizes.

### Total Store Compression Ratio

To measure compression across the entire Zarr store:

```python
import zarr
from pathlib import Path

# Open the store
store_path = Path("store.zarr")
z = zarr.open(store_path, mode='r')

# Get uncompressed size from metadata
sky_array = z['SKY']
uncompressed_bytes = sky_array.nbytes

# Get compressed size from disk
compressed_bytes = sum(
    f.stat().st_size
    for f in (store_path / 'SKY').rglob('*')
    if f.is_file() and not f.name.startswith('.')
)

ratio = uncompressed_bytes / compressed_bytes
print(f"Uncompressed: {uncompressed_bytes / 1e9:.2f} GB")
print(f"Compressed:   {compressed_bytes / 1e9:.2f} GB")
print(f"Compression ratio: {ratio:.2f}x")
```

This calculation sums the size of all chunk files in the SKY variable directory,
excluding metadata files (`.zarray`, `.zattrs`). The ratio indicates overall
compression effectiveness for the dataset.

### Using Compression Ratios to Adjust Chunk Size

Compression ratio affects the optimal uncompressed chunk size. If compressed
chunks are smaller than 1 MB, consider increasing `chunk_lm` to reduce the
number of HTTP requests for cloud access.

**Decision logic:**

```python
# Measure compressed chunk size
compressed_size_mb = 0.8  # example: 800 KB

if compressed_size_mb < 1.0:
    print("Compressed chunks are below 1 MB.")
    print("Consider increasing chunk_lm to reduce HTTP request overhead.")
    print(f"Current chunk_lm=1024 → suggest chunk_lm=1536 or 2048")
elif compressed_size_mb > 10.0:
    print("Compressed chunks exceed 10 MB.")
    print("Consider decreasing chunk_lm to reduce bandwidth waste.")
    print(f"Current chunk_lm=2048 → suggest chunk_lm=1024 or 1536")
else:
    print(f"Compressed chunk size ({compressed_size_mb:.1f} MB) is in the optimal range.")
```

Cross-reference [Chunking Fundamentals](chunking-fundamentals.md) for the
rationale behind the 1–10 MB target range (with 10–100 MB for very large
datasets).

!!! tip "Inspect .zarray Before Measuring Individual Chunks"

    Before measuring chunk file sizes, inspect the `.zarray` metadata to
    confirm the chunk shape and compressor configuration. Mismatched chunk
    shapes or incorrect compressor settings will produce misleading measurements.
    See [Write Path Pipeline section 6](chunking-write-path.md#inspecting-chunk-metadata)
    for `.zarray` inspection procedures.

**Bash alternative for cloud storage:**

For datasets on S3 or OSN, use the cloud provider's CLI to inspect object sizes:

```bash
# AWS S3: list first 20 chunks with sizes
aws s3 ls s3://bucket/store.zarr/SKY/ --recursive --human-readable | head -20

# OSN via s3cmd: list chunks
s3cmd ls s3://bucket/store.zarr/SKY/ --recursive --human-readable | head -20
```

These commands show compressed object sizes directly, avoiding the need to
download chunks for local inspection.

## External References

- [Zarr Performance User Guide](https://zarr.readthedocs.io/en/stable/user-guide/performance.html) -
  Official Zarr compression and performance recommendations
- [numcodecs Documentation](https://numcodecs.readthedocs.io/en/stable/) -
  Detailed codec and filter API reference
- [Blosc Documentation](https://www.blosc.org/pages/blosc-in-depth/) - In-depth
  explanation of shuffle modes and internal codecs

## See Also

- [Chunking Fundamentals](chunking-fundamentals.md) - Chunk size sweet spot and
  cloud storage performance characteristics
- [Write Path Pipeline](chunking-write-path.md) - How the ingest pipeline
  applies chunking and encoding at write time
- [FITS to Zarr Conversion](../fits-to-zarr.md) - User guide for the conversion
  CLI and Python API
