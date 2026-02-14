# Basic Plotting

The `radport` accessor provides rich visualization capabilities for OVRO-LWA data.

## Quick Plot

The simplest way to visualize your data:

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

# Plot a single frame
ds.radport.plot()
```

This creates a 2D sky map with:

- Automatic colorbar scaling
- Coordinate labels (l, m or RA, Dec)
- Title with observation metadata

## Selecting Time and Frequency

```python
# Plot specific time and frequency
ds.radport.plot(time_idx=0, freq_idx=10)

# Use nearest value instead of index
freq_mhz = 40.0
time_idx = ds.radport.nearest_time_idx(mjd=59000.5)
freq_idx = ds.radport.nearest_freq_idx(freq_mhz)

ds.radport.plot(time_idx=time_idx, freq_idx=freq_idx)
```

## Customizing Plots

```python
# Set colormap and normalization
ds.radport.plot(
    time_idx=0,
    freq_idx=0,
    cmap='viridis',
    norm='log'
)

# Add custom title
ds.radport.plot(
    time_idx=0,
    freq_idx=0,
    title='OVRO-LWA Sky Map at 40 MHz'
)
```

## Cutout Regions

Extract and plot a region of interest:

```python
# Define cutout box (l, m coordinates)
cutout = ds.radport.cutout(
    l_min=-0.2, l_max=0.2,
    m_min=-0.2, m_max=0.2
)

# Plot the cutout
cutout.radport.plot_cutout()
```

## Dynamic Spectra

Visualize time-frequency evolution:

```python
# Extract dynamic spectrum at a pixel
dyn_spec = ds.radport.dynamic_spectrum(l_idx=512, m_idx=512)

# Plot it
ds.radport.plot_dynamic_spectrum(dyn_spec)
```

## Light Curves and Spectra

```python
# Light curve at a specific location
lc = ds.radport.light_curve(l_idx=512, m_idx=512)
ds.radport.plot_light_curve(lc)

# Spectrum averaged over space and time
spec = ds.radport.spectrum()
ds.radport.plot_spectrum(spec)
```

## Multiple Subplots

Create grids of plots:

```python
# Plot multiple time steps
ds.radport.plot_time_grid(
    freq_idx=10,
    time_indices=[0, 10, 20, 30],
    figsize=(12, 10)
)

# Plot multiple frequencies
ds.radport.plot_frequency_grid(
    time_idx=0,
    freq_indices=[0, 5, 10, 15],
    figsize=(12, 10)
)
```

## Saving Figures

```python
import matplotlib.pyplot as plt

ds.radport.plot()
plt.savefig('ovro_skymap.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Next Steps

- Learn about [coordinate systems](coordinate-systems.md)
- Explore advanced [visualization methods](../user-guide/visualization.md)
- Create [animations](../user-guide/animations.md)
