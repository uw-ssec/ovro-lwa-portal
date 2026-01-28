# Loading OVRO-LWA Data with `open_dataset()`

The `open_dataset()` function provides a unified interface for loading OVRO-LWA
data from multiple sources, including local file paths, remote URLs, and DOI
identifiers.

## Installation

### Basic Installation

The core `open_dataset()` function works with local files using the base
installation:

```bash
pip install ovro_lwa_portal
```

### Remote Data Access

For remote data access (S3, Google Cloud Storage, DOI resolution), install with
the `remote` extras:

```bash
pip install 'ovro_lwa_portal[remote]'
```

This installs additional dependencies:

- `requests` - For DOI resolution
- `s3fs` - For AWS S3 access
- `gcsfs` - For Google Cloud Storage access
- `fsspec` - Unified filesystem interface
- `caltechdata_api` - Caltech Data API integration

## Quick Start

### Load from Local Path

```python
import ovro_lwa_portal

# Load a local zarr store
ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

# Access data
print(ds)
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Frequency range: {ds.frequency.values.min():.2e} to {ds.frequency.values.max():.2e} Hz")
```

### Load from Remote URL

```python
# Load from HTTPS
ds = ovro_lwa_portal.open_dataset("https://data.ovro.caltech.edu/obs_12345.zarr")

# Load from S3 (requires s3fs)
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

# Load from Google Cloud Storage (requires gcsfs)
ds = ovro_lwa_portal.open_dataset("gs://ovro-lwa-data/obs_12345.zarr")
```

### Load via DOI

```python
# With 'doi:' prefix
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")

# Without prefix
ds = ovro_lwa_portal.open_dataset("10.5281/zenodo.1234567")
```

## Function Signature

```python
def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    engine: str = "zarr",
    validate: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
```

### Parameters

- **source** (str or Path): Data source, can be:
  - Local file path (e.g., `"/path/to/data.zarr"`)
  - Remote URL (e.g., `"s3://bucket/data.zarr"`, `"https://..."`)
  - DOI string (e.g., `"doi:10.xxxx/xxxxx"` or `"10.xxxx/xxxxx"`)

- **chunks** (dict, str, or None, default `"auto"`): Chunking strategy for lazy
  loading:
  - `dict`: Explicit chunk sizes per dimension, e.g.,
    `{"time": 100, "frequency": 50}`
  - `"auto"`: Let xarray/dask determine optimal chunks (recommended)
  - `None`: Load entire dataset into memory (not recommended for large data)

- **engine** (str, default `"zarr"`): Backend engine for loading data. Currently
  supports `"zarr"`.

- **validate** (bool, default `True`): If True, validate that loaded data
  conforms to OVRO-LWA data model.

- **kwargs**: Additional arguments passed to the underlying loader (e.g.,
  `xr.open_zarr`)

### Returns

- **xr.Dataset**: OVRO-LWA dataset with standardized structure

## Usage Examples

### Custom Chunking

For large datasets, customize chunking to optimize memory usage and performance:

```python
# Explicit chunk sizes
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50, "l": 512, "m": 512}
)

# Auto chunking (recommended for most cases)
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks="auto"
)

# Load entire dataset into memory (small datasets only)
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks=None
)
```

### Disable Validation

Skip validation for faster loading when you're confident about the data
structure:

```python
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    validate=False
)
```

### Working with Cloud Storage

#### AWS S3

Set up AWS credentials via environment variables or `~/.aws/credentials`:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

Then load data:

```python
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/observation.zarr")
```

#### Google Cloud Storage

Set up GCS credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

Then load data:

```python
ds = ovro_lwa_portal.open_dataset("gs://ovro-lwa-data/observation.zarr")
```

### DOI Resolution

The function automatically resolves DOIs to data URLs using the Caltech Data
API:

```python
# DOI resolution happens automatically
ds = ovro_lwa_portal.open_dataset("doi:10.22002/abc123")

# The function:
# 1. Detects the DOI format
# 2. Resolves it to a data URL via Caltech Data API
# 3. Loads the data from the resolved URL
```

## Data Validation

By default, `open_dataset()` validates that the loaded data conforms to the
OVRO-LWA data model:

- **Expected dimensions**: `time`, `frequency`, `l`, `m`
- **Expected variables**: `SKY`, `BEAM`

If validation fails, warnings are logged but the dataset is still returned. This
allows working with non-standard or experimental data formats.

To disable validation:

```python
ds = ovro_lwa_portal.open_dataset("path/to/data.zarr", validate=False)
```

## Error Handling

The function provides clear error messages for common issues:

```python
from ovro_lwa_portal.io import DataSourceError

try:
    ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except DataSourceError as e:
    print(f"Failed to load data: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

### Common Errors

1. **FileNotFoundError**: Local path doesn't exist

   ```
   FileNotFoundError: Local path does not exist: /path/to/data.zarr
   ```

2. **ImportError**: Missing dependency for remote access

   ```
   ImportError: s3fs is required for S3 access. Install with: pip install s3fs
   ```

3. **DataSourceError**: Failed to load or resolve data
   ```
   DataSourceError: Failed to resolve DOI 10.xxxx/xxxxx: ...
   ```

## Performance Tips

### Lazy Loading

By default, `open_dataset()` uses lazy loading with dask, which means:

- Data is not loaded into memory immediately
- Only the chunks you access are loaded
- You can work with datasets larger than available RAM

```python
# Data is not loaded yet
ds = ovro_lwa_portal.open_dataset("large_dataset.zarr")

# Only this subset is loaded into memory
subset = ds.sel(time=slice(0, 10), frequency=slice(0, 5))
result = subset.SKY.mean().compute()
```

### Optimal Chunking

Choose chunk sizes based on your access patterns:

```python
# For time-series analysis (accessing many time steps)
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 1000, "frequency": 10, "l": 256, "m": 256}
)

# For spatial analysis (accessing full images)
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 1, "frequency": 1, "l": 1024, "m": 1024}
)
```

### Caching Remote Data

For frequently accessed remote data, consider caching locally:

```python
import fsspec

# Use fsspec caching
with fsspec.open_local(
    "simplecache::s3://bucket/data.zarr",
    s3={"anon": False},
    simplecache={"cache_storage": "/tmp/cache"}
) as local_path:
    ds = ovro_lwa_portal.open_dataset(local_path)
```

## Integration with Analysis Workflows

### Basic Analysis

```python
import ovro_lwa_portal
import matplotlib.pyplot as plt

# Load data
ds = ovro_lwa_portal.open_dataset("observation.zarr")

# Compute mean intensity over time
mean_intensity = ds.SKY.mean(dim="time")

# Plot
plt.figure(figsize=(10, 8))
mean_intensity.isel(frequency=0, polarization=0).plot()
plt.title("Mean Sky Intensity")
plt.show()
```

### WCS-Aware Plotting

```python
from astropy.wcs import WCS
import matplotlib.pyplot as plt

# Load data
ds = ovro_lwa_portal.open_dataset("observation.zarr")

# Reconstruct WCS from stored header
wcs_header_str = ds.attrs.get('fits_wcs_header') or ds.wcs_header_str.item().decode('utf-8')
wcs = WCS(wcs_header_str)

# Create WCS-aware plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(ds.SKY.isel(time=0, frequency=0, polarization=0).values)
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.grid(color='white', ls='dotted')
plt.show()
```

### Parallel Processing

```python
import ovro_lwa_portal
from dask.distributed import Client

# Start dask client for parallel processing
client = Client()

# Load data (lazy)
ds = ovro_lwa_portal.open_dataset("large_dataset.zarr")

# Parallel computation
result = ds.SKY.mean(dim=["l", "m"]).compute()

print(result)
client.close()
```

## Using the `radport` Accessor

After loading a dataset, you can access OVRO-LWA-specific visualization and
analysis features through the `radport` xarray accessor.

### Basic Usage

```python
import ovro_lwa_portal

# Load data
ds = ovro_lwa_portal.open_dataset("observation.zarr")

# Create a default visualization
fig = ds.radport.plot()
```

The accessor is automatically registered when you import `ovro_lwa_portal`, so
it's available on any xarray Dataset that has the required OVRO-LWA structure.

### Plotting Options

The `plot()` method supports various customization options:

```python
# Plot a specific time, frequency, and polarization
fig = ds.radport.plot(
    time_idx=5,      # Time index
    freq_idx=10,     # Frequency index
    pol=0,           # Polarization index
)

# Customize the colormap and scale
fig = ds.radport.plot(
    cmap="viridis",  # Matplotlib colormap
    vmin=-1.0,       # Minimum value for color scale
    vmax=16.0,       # Maximum value for color scale
)

# Use robust scaling for data with outliers
fig = ds.radport.plot(robust=True)

# Customize figure size
fig = ds.radport.plot(figsize=(12, 10))

# Plot without colorbar
fig = ds.radport.plot(add_colorbar=False)
```

### Plot Method Parameters

| Parameter      | Type  | Default     | Description                                    |
| -------------- | ----- | ----------- | ---------------------------------------------- |
| `var`          | str   | `"SKY"`     | Variable to plot (`"SKY"` or `"BEAM"`)         |
| `time_idx`     | int   | `0`         | Time index for the snapshot                    |
| `freq_idx`     | int   | `0`         | Frequency index for the snapshot               |
| `pol`          | int   | `0`         | Polarization index                             |
| `freq_mhz`     | float | `None`      | Select frequency by MHz (overrides `freq_idx`) |
| `time_mjd`     | float | `None`      | Select time by MJD (overrides `time_idx`)      |
| `cmap`         | str   | `"inferno"` | Matplotlib colormap name                       |
| `vmin`         | float | `None`      | Minimum value for color scale                  |
| `vmax`         | float | `None`      | Maximum value for color scale                  |
| `robust`       | bool  | `False`     | Use 2nd/98th percentile for scaling            |
| `mask_radius`  | int   | `None`      | Circular mask radius in pixels                 |
| `figsize`      | tuple | `(8, 6)`    | Figure size in inches                          |
| `add_colorbar` | bool  | `True`      | Whether to add a colorbar                      |

### Selecting by Value (MHz, MJD)

Instead of using indices, you can select by physical values:

```python
# Select frequency by MHz (more intuitive than index)
fig = ds.radport.plot(freq_mhz=50.0)

# Select time by MJD value
fig = ds.radport.plot(time_mjd=60000.5)

# Combine both
fig = ds.radport.plot(freq_mhz=50.0, time_mjd=60000.5)
```

### Selection Helper Methods

The accessor provides helper methods for finding indices:

```python
# Find index for a specific frequency in MHz
freq_idx = ds.radport.nearest_freq_idx(50.0)  # Returns index nearest to 50 MHz

# Find index for a specific time in MJD
time_idx = ds.radport.nearest_time_idx(60000.5)  # Returns index nearest to MJD

# Find indices for specific (l, m) coordinates
l_idx, m_idx = ds.radport.nearest_lm_idx(0.0, 0.0)  # Returns indices for center
```

### Circular Masking

For all-sky images, edge pixels may be invalid. Use `mask_radius` to apply a
circular mask:

```python
# Mask pixels outside radius of 1800 pixels from center
fig = ds.radport.plot(mask_radius=1800)
```

### Plotting BEAM Data

If your dataset contains BEAM data, you can plot it by specifying the variable:

```python
# Check if BEAM data is available
if ds.radport.has_beam:
    fig = ds.radport.plot(var="BEAM")
```

### Dataset Validation

The accessor automatically validates that the dataset has the required structure
for OVRO-LWA data when you access it. If validation fails, you'll get an
informative error message:

```python
import xarray as xr

# This will raise a ValueError with details about what's missing
invalid_ds = xr.Dataset({"other_var": (["x", "y"], [[1, 2], [3, 4]])})
try:
    invalid_ds.radport.plot()
except ValueError as e:
    print(f"Validation error: {e}")
```

Required dataset structure:

- **Dimensions**: `time`, `frequency`, `polarization`, `l`, `m`
- **Variables**: `SKY` (required), `BEAM` (optional)

### Complete Example

```python
import ovro_lwa_portal

# Load data from DOI
ds = ovro_lwa_portal.open_dataset("doi:10.22002/example")

# Print dataset info
print(f"Time steps: {len(ds.time)}")
print(f"Frequencies: {len(ds.frequency)}")
print(f"Has BEAM data: {ds.radport.has_beam}")

# Create visualization at specific frequency (by value in MHz)
fig = ds.radport.plot(
    freq_mhz=50.0,      # Select 50 MHz directly
    time_idx=0,
    cmap="inferno",
    robust=True,
    mask_radius=1800,   # Apply circular mask
    figsize=(10, 8),
)

# Save the figure
fig.savefig("observation_snapshot.png", dpi=150, bbox_inches="tight")
```

## Advanced Visualization Methods

The `radport` accessor provides additional methods for common analysis tasks.

### Spatial Cutouts

Extract and visualize a rectangular region of interest:

```python
# Extract a cutout (returns DataArray)
cutout = ds.radport.cutout(
    l_center=0.0,   # Center l coordinate
    m_center=0.0,   # Center m coordinate
    dl=0.1,         # Half-width in l direction
    dm=0.1,         # Half-width in m direction
    freq_mhz=50.0,  # Frequency selection
)

# Or extract and plot in one step
fig = ds.radport.plot_cutout(
    l_center=0.0, m_center=0.0,
    dl=0.1, dm=0.1,
    freq_mhz=50.0,
    robust=True,
)
```

### Dynamic Spectrum

View intensity variations across time and frequency at a single pixel:

```python
# Extract dynamic spectrum (returns DataArray)
dynspec = ds.radport.dynamic_spectrum(l=0.0, m=0.0)

# Or extract and plot in one step
fig = ds.radport.plot_dynamic_spectrum(
    l=0.0, m=0.0,    # Pixel location
    cmap="inferno",
    robust=True,
)
```

### Difference Maps

Identify transient sources or spectral features by differencing adjacent frames:

```python
# Time difference (current frame minus previous)
fig = ds.radport.plot_diff(
    mode="time",
    time_idx=5,      # Computes frame[5] - frame[4]
    freq_mhz=50.0,
)

# Frequency difference (current channel minus previous)
fig = ds.radport.plot_diff(
    mode="frequency",
    freq_idx=10,     # Computes freq[10] - freq[9]
    time_idx=0,
)

# Or get the data directly
diff_data = ds.radport.diff(mode="time", time_idx=5)
```

## Data Quality Methods

The accessor provides methods for assessing data quality and finding valid
frames.

### Finding Valid Frames

Automatically find the first frame with sufficient valid data:

```python
# Find first frame with at least 10% valid pixels
time_idx, freq_idx = ds.radport.find_valid_frame()

# Use higher threshold (50% valid pixels)
time_idx, freq_idx = ds.radport.find_valid_frame(min_finite_fraction=0.5)

# Then plot the valid frame
fig = ds.radport.plot(time_idx=time_idx, freq_idx=freq_idx)
```

### Data Availability Map

Compute the fraction of finite pixels for each time/frequency combination:

```python
# Get 2D array of data availability
frac = ds.radport.finite_fraction()

# Visualize where data is available
frac.plot(x="time", y="frequency")
```

## Grid Plots

Create multi-panel visualizations for comparing across time and frequency.

### Basic Grid Plot

```python
# Plot all time/frequency combinations (can be many panels!)
fig = ds.radport.plot_grid()

# Limit to specific times and frequencies
fig = ds.radport.plot_grid(
    time_indices=[0, 1, 2],
    freq_indices=[0, 1, 2],
)

# Select frequencies by MHz
fig = ds.radport.plot_grid(
    time_indices=[0, 1],
    freq_mhz_list=[46.0, 50.0, 54.0],
)
```

### Frequency Grid

Compare all frequencies at a single time:

```python
# All frequencies at time index 0
fig = ds.radport.plot_frequency_grid(time_idx=0)

# Specific frequencies only
fig = ds.radport.plot_frequency_grid(
    time_idx=0,
    freq_mhz_list=[46.0, 50.0, 54.0],
)
```

### Time Grid

Compare all times at a single frequency:

```python
# All times at the nearest frequency to 50 MHz
fig = ds.radport.plot_time_grid(freq_mhz=50.0)

# Specific times only
fig = ds.radport.plot_time_grid(
    freq_mhz=50.0,
    time_indices=[0, 1, 2, 3],
)
```

### Grid Plot Options

| Parameter        | Type  | Default      | Description                                   |
| ---------------- | ----- | ------------ | --------------------------------------------- |
| `time_indices`   | list  | `None`       | Time indices to plot (all if None)            |
| `freq_indices`   | list  | `None`       | Frequency indices to plot (all if None)       |
| `freq_mhz_list`  | list  | `None`       | Frequencies in MHz (overrides `freq_indices`) |
| `var`            | str   | `"SKY"`      | Variable to plot                              |
| `pol`            | int   | `0`          | Polarization index                            |
| `ncols`          | int   | `4`          | Number of columns in grid                     |
| `panel_size`     | tuple | `(3.0, 2.6)` | Size of each panel in inches                  |
| `cmap`           | str   | `"inferno"`  | Colormap                                      |
| `robust`         | bool  | `True`       | Use percentile-based global scaling           |
| `mask_radius`    | int   | `None`       | Circular mask radius                          |
| `share_colorbar` | bool  | `True`       | Use shared colorbar for all panels            |

## 1D Analysis Methods

Extract and analyze time series and frequency spectra at specific locations.

### Light Curves (Time Series)

Extract intensity as a function of time at a specific (l, m) location:

```python
# Extract light curve at center of image at 50 MHz
lc = ds.radport.light_curve(l=0.0, m=0.0, freq_mhz=50.0)

# Plot directly
lc.plot()

# Or use the convenience plotting method
fig = ds.radport.plot_light_curve(l=0.0, m=0.0, freq_mhz=50.0)
```

### Frequency Spectra

Extract intensity as a function of frequency at a specific location and time:

```python
# Extract spectrum at center of image at first time step
spec = ds.radport.spectrum(l=0.0, m=0.0, time_idx=0)

# Plot with frequency in MHz
fig = ds.radport.plot_spectrum(l=0.0, m=0.0, time_idx=0, freq_unit="MHz")
```

### Time-Averaged Images

Compute the mean image across all (or selected) time steps:

```python
# Average all time steps
avg = ds.radport.time_average()

# Average only specific times
avg = ds.radport.time_average(time_indices=[0, 1, 2])

# Plot the time-averaged image at 50 MHz
fig = ds.radport.plot_time_average(freq_mhz=50.0)
```

### Frequency-Averaged Images

Compute the mean image across all (or a band of) frequencies:

```python
# Average all frequencies
avg = ds.radport.frequency_average()

# Average only 45-55 MHz band
avg = ds.radport.frequency_average(freq_min_mhz=45.0, freq_max_mhz=55.0)

# Plot the frequency-averaged image at first time
fig = ds.radport.plot_frequency_average(time_idx=0)

# Plot with frequency band selection
fig = ds.radport.plot_frequency_average(
    time_idx=0,
    freq_min_mhz=45.0,
    freq_max_mhz=55.0,
)
```

## WCS & Coordinate Methods

If your dataset includes WCS (World Coordinate System) information, you can use
coordinate transformation and WCS-projected plotting. WCS headers are typically
stored in variable attributes or dataset attributes.

### Checking WCS Availability

```python
# Check if WCS information is available
if ds.radport.has_wcs:
    print("WCS information found!")
    fig = ds.radport.plot_wcs()
else:
    print("No WCS information available")
```

### Coordinate Transforms

Convert between pixel indices and celestial coordinates (RA/Dec):

```python
# Convert pixel indices to RA/Dec
ra, dec = ds.radport.pixel_to_coords(l_idx=100, m_idx=100)
print(f"Pixel (100, 100) -> RA={ra:.2f}°, Dec={dec:.2f}°")

# Convert RA/Dec to pixel indices
l_idx, m_idx = ds.radport.coords_to_pixel(ra=180.0, dec=45.0)
print(f"RA=180°, Dec=45° -> Pixel ({l_idx}, {m_idx})")
```

### WCS Plotting

Plot with celestial coordinate grid overlay:

```python
# Basic WCS plot
fig = ds.radport.plot_wcs()

# Plot with customizations
fig = ds.radport.plot_wcs(
    freq_mhz=50.0,        # Frequency selection
    time_idx=0,           # Time index
    cmap="inferno",       # Colormap
    mask_radius=1800,     # Circular mask
    grid_color="white",   # Grid line color
    grid_alpha=0.6,       # Grid transparency
    grid_linestyle=":",   # Grid line style
    label_color="white",  # Axis label color
    facecolor="black",    # Background color
)
```

### WCS Method Parameters

| Parameter        | Type  | Default     | Description                             |
| ---------------- | ----- | ----------- | --------------------------------------- |
| `var`            | str   | `"SKY"`     | Variable to plot                        |
| `time_idx`       | int   | `0`         | Time index                              |
| `freq_idx`       | int   | `0`         | Frequency index                         |
| `freq_mhz`       | float | `None`      | Frequency in MHz (overrides `freq_idx`) |
| `pol`            | int   | `0`         | Polarization index                      |
| `cmap`           | str   | `"inferno"` | Colormap                                |
| `vmin`           | float | `None`      | Minimum value for color scale           |
| `vmax`           | float | `None`      | Maximum value for color scale           |
| `robust`         | bool  | `True`      | Use percentile-based scaling            |
| `mask_radius`    | int   | `None`      | Circular mask radius in pixels          |
| `figsize`        | tuple | `(10, 10)`  | Figure size in inches                   |
| `add_colorbar`   | bool  | `True`      | Whether to add a colorbar               |
| `grid_color`     | str   | `"white"`   | Coordinate grid color                   |
| `grid_alpha`     | float | `0.6`       | Grid transparency                       |
| `grid_linestyle` | str   | `":"`       | Grid line style                         |
| `label_color`    | str   | `"white"`   | Axis label color                        |
| `facecolor`      | str   | `"black"`   | Background color                        |

### WCS Data Sources

The accessor looks for WCS information in the following locations (in order):

1. Variable attributes: `ds["SKY"].attrs["fits_wcs_header"]`
2. Dataset attributes: `ds.attrs["fits_wcs_header"]`
3. Dedicated variable: `ds["wcs_header_str"]`

WCS headers should be in FITS format and typically include:

```text
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CRPIX1  =                 2048.0
CRPIX2  =                 2048.0
CRVAL1  =                  180.0
CRVAL2  =                   45.0
CDELT1  =                -0.0125
CDELT2  =                 0.0125
```

## Animation & Export Methods

Create animations and export frames as individual image files.

### Time Evolution Animation

Animate changes over time at a fixed frequency:

```python
# Create animation and save to file
anim = ds.radport.animate_time(freq_mhz=50.0, output_file="time_evolution.mp4")

# Create animation for notebook display
anim = ds.radport.animate_time(freq_mhz=50.0)

# Display in Jupyter notebook
from IPython.display import HTML
HTML(anim.to_jshtml())

# Save as GIF
anim = ds.radport.animate_time(
    freq_mhz=50.0,
    output_file="animation.gif",
    fps=10,
    mask_radius=1800,
)
```

### Frequency Sweep Animation

Animate changes across frequencies at a fixed time:

```python
# Create frequency sweep animation
anim = ds.radport.animate_frequency(time_idx=0, output_file="freq_sweep.mp4")

# With customizations
anim = ds.radport.animate_frequency(
    time_idx=0,
    cmap="viridis",
    fps=3,
    robust=True,
)
```

### Export Frames

Export all (or selected) time/frequency combinations as individual image files:

```python
# Export all frames
files = ds.radport.export_frames("./frames")
print(f"Exported {len(files)} frames")

# Export specific time/frequency combinations
files = ds.radport.export_frames(
    "./frames",
    time_indices=[0, 1, 2],
    freq_indices=[0, 5, 10],
)

# Custom format and template
files = ds.radport.export_frames(
    "./frames",
    format="jpg",
    filename_template="{var}_t{time_idx:04d}_f{freq_mhz:.1f}MHz.{format}",
    dpi=200,
)
```

### Animation Parameters

| Parameter               | Type  | Default     | Description                           |
| ----------------------- | ----- | ----------- | ------------------------------------- |
| `freq_idx` / `time_idx` | int   | `0`         | Index for fixed dimension             |
| `freq_mhz` / `time_mjd` | float | `None`      | Select by value (overrides index)     |
| `var`                   | str   | `"SKY"`     | Variable to animate                   |
| `pol`                   | int   | `0`         | Polarization index                    |
| `output_file`           | str   | `None`      | Path to save animation (.mp4 or .gif) |
| `fps`                   | int   | `5`         | Frames per second                     |
| `cmap`                  | str   | `"inferno"` | Colormap                              |
| `vmin` / `vmax`         | float | `None`      | Color scale limits                    |
| `robust`                | bool  | `True`      | Use percentile-based scaling          |
| `mask_radius`           | int   | `None`      | Circular mask radius in pixels        |
| `figsize`               | tuple | `(8, 6)`    | Figure size in inches                 |
| `dpi`                   | int   | `100`       | Resolution for saved animation        |

### Export Parameters

| Parameter           | Type | Default                          | Description                               |
| ------------------- | ---- | -------------------------------- | ----------------------------------------- |
| `output_dir`        | str  | (required)                       | Directory to save images                  |
| `time_indices`      | list | `None`                           | Time indices to export (all if None)      |
| `freq_indices`      | list | `None`                           | Frequency indices to export (all if None) |
| `format`            | str  | `"png"`                          | Image format (png, jpg, pdf)              |
| `filename_template` | str  | `"{var}_t{...}_f{...}.{format}"` | Filename pattern                          |
| `dpi`               | int  | `150`                            | Resolution for saved images               |

### Dependencies

- **MP4 animations**: Require `ffmpeg` to be installed on the system
- **GIF animations**: Use `pillow` (included with matplotlib)

## Sliding Window Time-Frequency Analysis

Analyze data using sliding windows across time and frequency dimensions to
detect variable and transient radio sources.

### Sliding Window Stacks

Create averaged image stacks using a sliding kernel that moves across both time
and frequency dimensions:

```python
# Create sliding window stacks
stacks = ds.radport.sliding_window_stacks(
    l_center=0.0,      # Center l coordinate of cutout
    m_center=0.0,      # Center m coordinate of cutout
    cutout_size=0.1,   # Half-width of cutout region
    time_window=5,     # Number of time steps per window
    freq_window=3,     # Number of frequency channels per window
)

# Access the averaged images
stacks.stack.isel(kernel_time=0, kernel_freq=0).plot()

# Access RMS per kernel
stacks.rms.isel(kernel_time=0, kernel_freq=0).plot()

# Use stepping to reduce computation
stacks = ds.radport.sliding_window_stacks(
    l_center=0.0, m_center=0.0, cutout_size=0.1,
    time_window=5, freq_window=3,
    time_step=2, freq_step=2,  # Slide by 2 steps
)
```

The output dataset contains:

- `stack`: (kernel_time, kernel_freq, l, m) - averaged images
- `rms`: (kernel_time, kernel_freq, l, m) - RMS per kernel
- `peak_flux`: (kernel_time, kernel_freq) - peak flux per kernel
- `peak_l`, `peak_m`: (kernel_time, kernel_freq) - peak positions
- `n_valid`: (kernel_time, kernel_freq) - count of valid pixels

### Variability Index

Compute variability metrics for each pixel across time and frequency:

```python
# Modulation index (default): std(flux) / mean(flux)
var_idx = ds.radport.variability_index(
    l_center=0.0, m_center=0.0, cutout_size=0.2
)
var_idx.plot()

# Chi-squared metric for detecting transients
chi2 = ds.radport.variability_index(
    l_center=0.0, m_center=0.0, cutout_size=0.2,
    metric='chi_squared'
)

# Peak-to-mean ratio
peak_ratio = ds.radport.variability_index(
    l_center=0.0, m_center=0.0, cutout_size=0.2,
    metric='peak_to_mean'
)
```

Available metrics:

- `modulation_index`: Fractional variability (std/mean). Values ~0 indicate
  steady sources, >0.3 indicates significant variability.
- `chi_squared`: Deviation from constant flux (normalized).
- `peak_to_mean`: Ratio of maximum to average flux. Values ~1 indicate steady
  sources, >2 indicates transients.

### Finding Variable Sources

Search for variable sources across the full field of view:

```python
# Find variable sources with default thresholds
candidates = ds.radport.find_variable_sources(
    time_window=5,
    freq_window=3,
    snr_threshold=5.0,          # Minimum SNR
    variability_threshold=0.3,   # Minimum modulation index
)

print(f"Found {candidates.sizes['candidate']} variable sources")

# Access candidate properties
if candidates.sizes['candidate'] > 0:
    print(f"Most variable source: l={candidates.l.values[0]:.3f}, "
          f"m={candidates.m.values[0]:.3f}")

    # Plot light curve of most variable source
    candidates.light_curve.isel(candidate=0).plot()
```

The output dataset contains for each candidate:

- `l`, `m`: Source position in direction cosines
- `snr`: Peak signal-to-noise ratio
- `variability`: Modulation index
- `peak_time_idx`, `peak_freq_idx`: Indices of peak flux
- `peak_flux`, `mean_flux`: Flux statistics
- `light_curve`: Flux vs time at peak frequency

### Animate Sliding Window Stacks

Create animations showing how the averaged image changes as the kernel slides:

```python
# Create stacks first
stacks = ds.radport.sliding_window_stacks(
    l_center=0.0, m_center=0.0, cutout_size=0.1,
    time_window=5, freq_window=3
)

# Animate through time dimension
anim = ds.radport.animate_sliding_window(stacks, dimension='time')

# Display in Jupyter notebook
from IPython.display import HTML
HTML(anim.to_jshtml())

# Save to file
anim = ds.radport.animate_sliding_window(
    stacks,
    dimension='frequency',
    output_file='sliding_window.mp4',
    fps=10
)
```

### Sliding Window Parameters

| Parameter               | Type  | Default | Description                               |
| ----------------------- | ----- | ------- | ----------------------------------------- |
| `l_center`              | float | req.    | Center l coordinate of cutout             |
| `m_center`              | float | req.    | Center m coordinate of cutout             |
| `cutout_size`           | float | req.    | Half-width of cutout in l/m               |
| `time_window`           | int   | req.    | Number of time steps per window           |
| `freq_window`           | int   | req.    | Number of frequency channels per window   |
| `time_step`             | int   | `1`     | Step size for sliding time window         |
| `freq_step`             | int   | `1`     | Step size for sliding frequency window    |
| `var`                   | str   | `"SKY"` | Variable to analyze                       |
| `pol`                   | int   | `0`     | Polarization index                        |
| `min_valid_fraction`    | float | `0.5`   | Minimum valid pixel fraction per kernel   |
| `snr_threshold`         | float | `5.0`   | Minimum SNR for variable source detection |
| `variability_threshold` | float | `0.3`   | Minimum modulation index for detection    |
| `exclude_horizon`       | bool  | `True`  | Exclude pixels near horizon (l²+m² > 0.9) |
| `max_candidates`        | int   | `100`   | Maximum candidates to return              |

## Source Detection Methods

Analyze images for noise characteristics and detect significant sources.

### Local RMS Noise Map

Compute local RMS noise estimate using a sliding box:

```python
# Compute RMS noise map with 50-pixel box
rms = ds.radport.rms_map(freq_mhz=50.0, box_size=50)
rms.plot()

# Use larger box for smoother estimate
rms = ds.radport.rms_map(box_size=100)
```

### Signal-to-Noise Ratio Map

Compute SNR map (signal divided by local RMS):

```python
# Get SNR map
snr = ds.radport.snr_map(freq_mhz=50.0, box_size=50)

# Find significant pixels (SNR > 5σ)
significant = snr.where(snr > 5)

# Plot SNR map with diverging colormap
fig = ds.radport.plot_snr_map(freq_mhz=50.0, mask_radius=1800)
```

### Peak Detection

Find local maxima above an SNR threshold:

```python
# Find peaks with SNR > 5
peaks = ds.radport.find_peaks(freq_mhz=50.0, threshold_sigma=5.0)
print(f"Found {len(peaks)} peaks")

# Peaks are sorted by SNR (brightest first)
for p in peaks[:5]:
    print(f"  l={p['l']:.3f}, m={p['m']:.3f}, "
          f"flux={p['flux']:.2f} Jy, SNR={p['snr']:.1f}")

# Adjust minimum separation between peaks
peaks = ds.radport.find_peaks(
    threshold_sigma=3.0,
    min_separation=10,  # pixels
)
```

### Peak Flux Map

Find the maximum flux at each pixel across all time steps:

```python
# Get peak flux map (maximum across time)
peak_map = ds.radport.peak_flux_map(freq_mhz=50.0)
peak_map.plot()
```

### Source Detection Parameters

| Parameter         | Type  | Default | Description             |
| ----------------- | ----- | ------- | ----------------------- |
| `time_idx`        | int   | `0`     | Time index              |
| `freq_idx`        | int   | `None`  | Frequency index         |
| `freq_mhz`        | float | `None`  | Frequency in MHz        |
| `var`             | str   | `"SKY"` | Variable to analyze     |
| `pol`             | int   | `0`     | Polarization index      |
| `box_size`        | int   | `50`    | Box size for local RMS  |
| `threshold_sigma` | float | `5.0`   | SNR threshold for peaks |
| `min_separation`  | int   | `5`     | Minimum peak separation |

### Dependencies

- Uses `scipy.ndimage` for efficient local statistics (uniform_filter,
  maximum_filter)

## Spectral Analysis Methods

Analyze spectral properties across the frequency dimension.

### Spectral Index at a Point

Compute the spectral index (power-law slope) at a specific location:

```python
# Spectral index at image center using first and last frequencies
alpha = ds.radport.spectral_index(l=0.0, m=0.0)
print(f"Spectral index: {alpha:.2f}")

# Use specific frequency range in MHz
alpha = ds.radport.spectral_index(l=0.1, m=-0.2, freq1_mhz=46.0, freq2_mhz=54.0)

# Use frequency indices
alpha = ds.radport.spectral_index(l=0.0, m=0.0, freq1_idx=0, freq2_idx=-1)
```

The spectral index α is defined by S ∝ ν^α, computed as: α = log(S₂/S₁) /
log(ν₂/ν₁)

Typical values:

- α ≈ -0.7: Synchrotron emission (most radio sources)
- α ≈ +2.0: Thermal emission (optically thick)
- α ≈ -0.1: Free-free emission (optically thin)

### Spectral Index Map

Compute spectral index for every pixel in the image:

```python
# Get spectral index map
alpha_map = ds.radport.spectral_index_map(time_idx=0)
alpha_map.plot(vmin=-2, vmax=1, cmap='RdBu_r')

# With specific frequency range
alpha_map = ds.radport.spectral_index_map(freq1_mhz=46.0, freq2_mhz=54.0)

# Plot with diverging colormap and horizon mask
fig = ds.radport.plot_spectral_index_map(
    freq1_mhz=46.0,
    freq2_mhz=54.0,
    mask_radius=1800,
    vmin=-2, vmax=1
)
```

### Integrated Flux Density

Integrate flux density over a frequency range at a specific location:

```python
# Integrate over all frequencies at image center
flux_int = ds.radport.integrated_flux(l=0.0, m=0.0)
print(f"Integrated flux: {flux_int:.2e} Jy·Hz")

# Integrate over specific frequency range
flux_int = ds.radport.integrated_flux(
    l=0.1, m=-0.2,
    freq1_mhz=46.0, freq2_mhz=54.0
)

# Use frequency indices
flux_int = ds.radport.integrated_flux(l=0.0, m=0.0, freq1_idx=0, freq2_idx=5)
```

Uses trapezoidal integration (numpy.trapezoid) for accurate results.

### Spectral Analysis Parameters

| Parameter   | Type  | Default  | Description             |
| ----------- | ----- | -------- | ----------------------- |
| `l`         | float | required | l direction cosine      |
| `m`         | float | required | m direction cosine      |
| `time_idx`  | int   | `0`      | Time index              |
| `freq1_idx` | int   | `None`   | First frequency index   |
| `freq2_idx` | int   | `None`   | Second frequency index  |
| `freq1_mhz` | float | `None`   | First frequency in MHz  |
| `freq2_mhz` | float | `None`   | Second frequency in MHz |
| `var`       | str   | `"SKY"`  | Variable to analyze     |
| `pol`       | int   | `0`      | Polarization index      |

### Notes

- Non-positive flux values result in NaN spectral indices
- At least two frequency channels are required for spectral index calculation
- Frequency range defaults to first and last channels if not specified

## API Reference

For complete API documentation, see:

- [ovro_lwa_portal.io module](../src/ovro_lwa_portal/io.py)
- [xarray documentation](https://docs.xarray.dev/)
- [dask documentation](https://docs.dask.org/)

## Troubleshooting

### Issue: "s3fs is required for S3 access"

**Solution**: Install the remote extras:

```bash
pip install 'ovro_lwa_portal[remote]'
```

### Issue: "Failed to resolve DOI"

**Possible causes**:

1. DOI doesn't exist or is malformed
2. Network connectivity issues
3. Caltech Data API is unavailable

**Solution**: Check the DOI is correct and try accessing it directly in a
browser.

### Issue: "Dataset may not be OVRO-LWA format"

**Cause**: The loaded dataset doesn't have expected dimensions or variables.

**Solution**: This is just a warning. If you're working with non-standard data,
disable validation:

```python
ds = ovro_lwa_portal.open_dataset("data.zarr", validate=False)
```

### Issue: Memory errors with large datasets

**Solution**: Use smaller chunks or ensure lazy loading is enabled:

```python
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 10, "frequency": 10, "l": 256, "m": 256}
)
```

## See Also

- [FITS to Zarr Conversion](../src/ovro_lwa_portal/ingest/README.md)
- [xarray I/O documentation](https://docs.xarray.dev/en/stable/user-guide/io.html)
- [fsspec documentation](https://filesystem-spec.readthedocs.io/)
- [Caltech Data](https://data.caltech.edu/)
