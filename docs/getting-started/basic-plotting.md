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

## Saving Figures

```python
import matplotlib.pyplot as plt

ds.radport.plot()
plt.savefig('ovro_skymap.png', dpi=300, bbox_inches='tight')
plt.close()
```

## What Else Can You Plot?

The `radport` accessor supports many more visualization methods beyond basic
sky maps. See the [Visualization](../user-guide/visualization.md) guide for:

- **Cutout regions** — extract and plot sub-regions of interest
- **Dynamic spectra** — time-frequency waterfalls at a pixel or spatial region
- **Light curves and spectra** — track intensity over time or frequency
- **Grid plots** — multi-panel layouts across time steps or frequencies
- **Difference plots** — visualize changes between frames
- **Averaged visualizations** — time or frequency averaged images
- **Contour overlays** and custom colormaps

## Next Steps

- Learn about [coordinate systems](coordinate-systems.md)
- Explore the full [Visualization](../user-guide/visualization.md) guide
- Create [animations](../user-guide/animations.md)
