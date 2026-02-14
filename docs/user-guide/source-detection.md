# Source Detection

The `radport` accessor provides methods for detecting and analyzing radio sources in OVRO-LWA data.

## Signal-to-Noise Maps

### RMS Map

Calculate the root-mean-square (noise) map:

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

# Calculate RMS over time
rms = ds.radport.rms_map(freq_idx=10)

# The result is a 2D map (l, m)
print(f"RMS map shape: {rms.shape}")
print(f"Median RMS: {rms.median().values:.3e}")
```

### SNR Map

Calculate signal-to-noise ratio map:

```python
# SNR map at a specific time and frequency
snr = ds.radport.snr_map(time_idx=0, freq_idx=10)

# Plot it
ds.radport.plot_snr_map(
    snr,
    vmin=0,
    vmax=10,
    cmap='hot'
)
```

**Interpretation:**

- SNR < 3: Noise-dominated
- SNR 3-5: Marginal detections
- SNR > 5: Significant detections
- SNR > 10: Strong sources

## Peak Finding

### Basic Peak Detection

Find local maxima in an image:

```python
# Find peaks with SNR > 5
peaks = ds.radport.find_peaks(
    time_idx=0,
    freq_idx=10,
    threshold=5.0,
    min_separation=10  # Minimum separation in pixels
)

print(f"Found {len(peaks)} peaks")
print(f"Peak locations (l, m): {peaks}")
```

### Peak Flux Map

Create a map showing only peak values:

```python
# Peak flux map
peak_map = ds.radport.peak_flux_map(
    time_idx=0,
    freq_idx=10,
    threshold=5.0
)

# Plot with base image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original image
ds.radport.plot(time_idx=0, freq_idx=10, ax=axes[0])
axes[0].set_title('Original')

# Peak map
axes[1].imshow(peak_map, origin='lower', cmap='hot')
axes[1].set_title('Detected Peaks')
plt.tight_layout()
plt.show()
```

### Visualizing Detections

Overlay detected sources on the image:

```python
import matplotlib.pyplot as plt

# Detect sources
peaks = ds.radport.find_peaks(
    time_idx=0,
    freq_idx=10,
    threshold=5.0
)

# Plot image
ds.radport.plot(time_idx=0, freq_idx=10)

# Overlay detections
ax = plt.gca()
if len(peaks) > 0:
    l_coords, m_coords = zip(*peaks)
    ax.scatter(
        l_coords, m_coords,
        marker='o',
        facecolors='none',
        edgecolors='red',
        s=200,
        linewidths=2,
        label=f'{len(peaks)} detections'
    )
    ax.legend()

plt.show()
```

## Multi-Frequency Detection

Detect sources across multiple frequencies:

```python
# Detection at multiple frequencies
freq_indices = [0, 5, 10, 15]
all_peaks = {}

for freq_idx in freq_indices:
    peaks = ds.radport.find_peaks(
        time_idx=0,
        freq_idx=freq_idx,
        threshold=5.0
    )
    all_peaks[freq_idx] = peaks
    print(f"Frequency {freq_idx}: {len(peaks)} sources")
```

## Transient Detection

Identify sources that appear or vary significantly:

### Time-Domain Variability

```python
import numpy as np

# Calculate variability for each pixel
time_indices = range(0, 100)
l_idx, m_idx = 512, 512

# Get light curve
lc = ds.radport.light_curve(l_idx=l_idx, m_idx=m_idx)

# Calculate variability metrics
mean_flux = lc.mean().values
std_flux = lc.std().values
variability = std_flux / mean_flux if mean_flux > 0 else 0

print(f"Variability index: {variability:.2f}")

# High variability indicates transient or variable source
if variability > 0.5:
    print("Potentially variable source detected!")
```

### Difference Imaging

Detect transients using difference imaging:

```python
# Reference image (time-averaged)
reference = ds.radport.time_average(time_min=0, time_max=50)

# Current image
current = ds['SKY'].isel(time=100, frequency=10)

# Difference
difference = current - reference.isel(frequency=10)

# Find significant changes
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(difference.values, origin='lower', cmap='RdBu_r')
plt.colorbar(label='Flux Difference')
plt.title('Transient Detection: Difference Image')
plt.show()
```

## Source Cataloging

Build a source catalog from detections:

```python
import pandas as pd

# Detect sources
peaks = ds.radport.find_peaks(
    time_idx=0,
    freq_idx=10,
    threshold=5.0
)

# Extract properties for each source
catalog = []
for l_idx, m_idx in peaks:
    # Get position
    l_val = ds.coords['l'].values[l_idx]
    m_val = ds.coords['m'].values[m_idx]

    # Get peak flux
    flux = ds['SKY'].isel(
        time=0,
        frequency=10,
        l=l_idx,
        m=m_idx
    ).values

    # Get spectral index if multiple frequencies available
    if len(ds.coords['frequency']) > 1:
        alpha = ds.radport.spectral_index(
            freq_idx1=0,
            freq_idx2=10,
            l_idx=l_idx,
            m_idx=m_idx
        )
    else:
        alpha = np.nan

    catalog.append({
        'l': l_val,
        'm': m_val,
        'l_idx': l_idx,
        'm_idx': m_idx,
        'flux': flux,
        'spectral_index': alpha
    })

# Convert to DataFrame
df = pd.DataFrame(catalog)
df = df.sort_values('flux', ascending=False)

print(df.head())
```

## Advanced Detection

### Matched Filter

For detecting sources with known shapes:

```python
from scipy.ndimage import gaussian_filter

# Get image
image = ds['SKY'].isel(time=0, frequency=10).values

# Apply matched filter (Gaussian)
filtered = gaussian_filter(image, sigma=2.0)

# Find peaks in filtered image
from scipy.signal import find_peaks_cwt
# ... peak detection on filtered image
```

### Blob Detection

For extended sources:

```python
from skimage.feature import blob_log

# Get image
image = ds['SKY'].isel(time=0, frequency=10).values

# Detect blobs (extended sources)
blobs = blob_log(
    image,
    min_sigma=1,
    max_sigma=10,
    num_sigma=10,
    threshold=0.1
)

print(f"Detected {len(blobs)} extended sources")

# Visualize
import matplotlib.pyplot as plt
ds.radport.plot(time_idx=0, freq_idx=10)
ax = plt.gca()

for blob in blobs:
    y, x, r = blob
    circle = plt.Circle(
        (x, y), r,
        color='red',
        fill=False,
        linewidth=2
    )
    ax.add_patch(circle)

plt.show()
```

## Quality Filtering

Filter detections based on quality criteria:

```python
# Get SNR map
snr = ds.radport.snr_map(time_idx=0, freq_idx=10)

# Find peaks
peaks = ds.radport.find_peaks(
    time_idx=0,
    freq_idx=10,
    threshold=5.0
)

# Filter by additional criteria
filtered_peaks = []
for l_idx, m_idx in peaks:
    snr_val = snr[m_idx, l_idx]  # Note: y, x order

    # Check if SNR is high enough
    if snr_val > 7.0:
        # Check if not at edge
        margin = 10
        if (margin < l_idx < snr.shape[1] - margin and
            margin < m_idx < snr.shape[0] - margin):
            filtered_peaks.append((l_idx, m_idx))

print(f"Filtered: {len(peaks)} → {len(filtered_peaks)} sources")
```

## Best Practices

1. **Background Estimation**: Ensure accurate background/noise estimation

2. **Edge Effects**: Exclude detections near image edges

3. **False Positives**: Use conservative thresholds (SNR > 5-7)

4. **Validation**: Visually inspect detections

5. **Multi-Frequency Confirmation**: Confirm detections across multiple frequencies

```python
# Multi-frequency validation
def validate_source(ds, l_idx, m_idx, freq_indices, snr_threshold=5.0):
    """Check if source is detected at multiple frequencies."""
    detections = 0
    for freq_idx in freq_indices:
        snr = ds.radport.snr_map(time_idx=0, freq_idx=freq_idx)
        if snr[m_idx, l_idx] > snr_threshold:
            detections += 1
    return detections >= len(freq_indices) * 0.5  # 50% criterion
```

## Next Steps

- Learn about [spectral analysis](spectral-analysis.md) of detected sources
- Try the [transient analysis tutorial](../tutorials/transient-analysis.md)
- Explore the [API reference](../api/radport-accessor.md)
