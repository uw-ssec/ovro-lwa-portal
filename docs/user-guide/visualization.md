# Visualization

The `radport` accessor provides comprehensive visualization methods for OVRO-LWA
data.

For an introduction to plotting, refer to the
[Basic Plotting](../getting-started/basic-plotting.md) guide.

## Cutout Visualization

Extract and visualize regions of interest:

```python
# Create cutout around a region
cutout = ds.radport.cutout(
    l_min=-0.3, l_max=0.3,
    m_min=-0.3, m_max=0.3,
    time_min=0, time_max=10,
    freq_min=5, freq_max=15
)

# Plot the cutout
cutout.radport.plot_cutout(
    time_idx=0,
    freq_idx=0,
    show_box=True  # Show original region boundaries
)
```

## Time-Frequency Visualizations

### Dynamic Spectrum

Visualize temporal evolution at a specific spatial location:

```python
# Extract dynamic spectrum
dyn_spec = ds.radport.dynamic_spectrum(l_idx=512, m_idx=512)

# Plot it
ds.radport.plot_dynamic_spectrum(
    dyn_spec,
    cmap='plasma',
    aspect='auto',
    interpolation='nearest'
)
```

For spatial regions:

```python
# Average over a region
dyn_spec_region = ds.radport.dynamic_spectrum(
    l_min=500, l_max=524,
    m_min=500, m_max=524
)
```

### Difference Plots

Visualize changes between time steps or frequencies:

```python
# Time difference
diff_time = ds.radport.diff(dim='time', n=1)
ds.radport.plot_diff(diff_time, freq_idx=10)

# Frequency difference
diff_freq = ds.radport.diff(dim='frequency', n=1)
ds.radport.plot_diff(diff_freq, time_idx=0)
```

## Light Curves and Spectra

### Light Curves

Track intensity over time:

```python
# Single pixel
lc = ds.radport.light_curve(l_idx=512, m_idx=512)
ds.radport.plot_light_curve(lc, label='Source A')

# Multiple sources
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))

for l, m, label in [(512, 512, 'Source A'), (600, 600, 'Source B')]:
    lc = ds.radport.light_curve(l_idx=l, m_idx=m)
    ds.radport.plot_light_curve(lc, label=label, ax=ax)

ax.legend()
plt.show()
```

### Frequency Spectra

```python
# Spectrum at a location
spec = ds.radport.spectrum(l_idx=512, m_idx=512)
ds.radport.plot_spectrum(spec)

# Spatially averaged spectrum
spec_avg = ds.radport.spectrum()
ds.radport.plot_spectrum(spec_avg, label='Average')
```

## Grid Plots

### Time Grid

Plot multiple time steps side-by-side:

```python
# Grid of 4 time steps
ds.radport.plot_time_grid(
    freq_idx=10,
    time_indices=[0, 5, 10, 15],
    ncols=2,
    figsize=(12, 10),
    cmap='inferno'
)
```

### Frequency Grid

Plot multiple frequencies:

```python
# Grid of 4 frequencies
ds.radport.plot_frequency_grid(
    time_idx=0,
    freq_indices=[0, 5, 10, 15],
    ncols=2,
    figsize=(12, 10)
)
```

### General Grid Plot

Maximum flexibility:

```python
# Custom grid configuration
ds.radport.plot_grid(
    time_indices=[0, 5, 10],
    freq_indices=[0, 10],
    nrows=2,
    ncols=3,
    figsize=(15, 10),
    share_colorbar=True
)
```

## Averaged Visualizations

### Time Average

```python
# Average over all time
time_avg = ds.radport.time_average()
ds.radport.plot_time_average(time_avg, freq_idx=10)

# Average over time range
time_avg = ds.radport.time_average(time_min=10, time_max=50)
```

### Frequency Average

```python
# Average over all frequencies
freq_avg = ds.radport.frequency_average()
ds.radport.plot_frequency_average(freq_avg, time_idx=0)

# Average over frequency range
freq_avg = ds.radport.frequency_average(freq_min=5, freq_max=15)
```

## Advanced Customization

### Custom Colormaps

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Log scale with custom colormap
ds.radport.plot(
    time_idx=0,
    freq_idx=10,
    cmap='magma',
    norm='log',
    vmin=1e-3,
    vmax=1e2
)
```

### Multiple Panels

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot different frequencies
for ax, freq_idx in zip(axes, [0, 5, 10]):
    ds.radport.plot(time_idx=0, freq_idx=freq_idx, ax=ax)
    ax.set_title(f'Frequency index: {freq_idx}')

plt.tight_layout()
plt.show()
```

### Overlaying Contours

```python
import matplotlib.pyplot as plt

# Base image
ds.radport.plot(time_idx=0, freq_idx=10)

# Overlay contours
ax = plt.gca()
data = ds['SKY'].isel(time=0, frequency=10).values
ax.contour(data, colors='white', alpha=0.5, levels=5)
plt.show()
```

## Validation Tools

### Finding Valid Frames

Find frames without NaN values:

```python
# Find first valid frame
valid_idx = ds.radport.find_valid_frame(freq_idx=10)
ds.radport.plot(time_idx=valid_idx, freq_idx=10)
```

### Finite Fraction

Check data quality:

```python
# Get fraction of finite values
fraction = ds.radport.finite_fraction(time_idx=0, freq_idx=10)
print(f"Valid data: {fraction*100:.1f}%")
```

## Next Steps

- Learn about [animations](animations.md)
- Explore [WCS coordinate plotting](wcs-coordinates.md)
- Try [spectral analysis](spectral-analysis.md)
- See the [API reference](../api/radport-accessor.md)
