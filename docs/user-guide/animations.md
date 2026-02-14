# Animations

The `radport` accessor provides methods for creating animations of time-series and frequency data.

## Time Animation

Animate data evolution over time:

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

# Create animation over time at a fixed frequency
anim = ds.radport.animate_time(
    freq_idx=10,
    interval=100,  # milliseconds between frames
    cmap='viridis',
    norm='log'
)

# Display in Jupyter
from IPython.display import HTML
HTML(anim.to_jshtml())

# Or save to file
anim.save('time_evolution.mp4', writer='ffmpeg', fps=10)
```

### Customizing Time Animations

```python
# Specify time range
anim = ds.radport.animate_time(
    freq_idx=10,
    time_min=0,
    time_max=100,
    interval=50,
    cmap='plasma',
    figsize=(10, 8)
)
```

### Adding Markers

Overlay markers for tracked sources:

```python
import matplotlib.pyplot as plt

# Detect sources in first frame
peaks = ds.radport.find_peaks(time_idx=0, freq_idx=10, threshold=5.0)

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))

def animate_with_markers(time_idx):
    ax.clear()

    # Plot frame
    frame = ds['SKY'].isel(time=time_idx, frequency=10)
    im = ax.imshow(frame.values, origin='lower', cmap='viridis')

    # Overlay markers
    if len(peaks) > 0:
        l_coords, m_coords = zip(*peaks)
        ax.scatter(l_coords, m_coords, marker='o',
                  facecolors='none', edgecolors='red', s=200, linewidths=2)

    ax.set_title(f'Time {time_idx}')
    return im,

from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, animate_with_markers, frames=range(100), interval=100)
```

## Frequency Animation

Animate across frequency channels:

```python
# Create animation over frequency at a fixed time
anim = ds.radport.animate_frequency(
    time_idx=0,
    interval=100,
    cmap='inferno'
)

# Save
anim.save('frequency_sweep.mp4', writer='ffmpeg', fps=10)
```

### Customizing Frequency Animations

```python
# Specify frequency range
anim = ds.radport.animate_frequency(
    time_idx=0,
    freq_min=0,
    freq_max=20,
    interval=150,
    figsize=(10, 8),
    norm='log'
)
```

## Exporting Frame Sequences

Export individual frames for external processing:

```python
# Export frames as PNG images
ds.radport.export_frames(
    output_dir='frames/',
    freq_idx=10,
    time_min=0,
    time_max=100,
    format='png',
    dpi=150,
    cmap='viridis'
)

# Creates: frames/frame_0000.png, frame_0001.png, ...
```

### Custom Frame Export

```python
# Export with custom naming
ds.radport.export_frames(
    output_dir='output/',
    freq_idx=10,
    format='png',
    prefix='ovro_',
    dpi=300,
    cmap='plasma',
    norm='log'
)
```

### Creating Video from Frames

Use ffmpeg externally to create video:

```bash
ffmpeg -framerate 10 -pattern_type glob -i 'frames/*.png' \
       -c:v libx264 -pix_fmt yuv420p output.mp4
```

## Multi-Panel Animations

Animate multiple views simultaneously:

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

def update(frame):
    # Clear axes
    for ax in axes.flat:
        ax.clear()

    # Different frequencies
    for ax, freq_idx in zip(axes.flat, [0, 5, 10, 15]):
        data = ds['SKY'].isel(time=frame, frequency=freq_idx)
        ax.imshow(data.values, origin='lower', cmap='viridis')
        freq_mhz = ds.coords['frequency'].values[freq_idx] / 1e6
        ax.set_title(f'{freq_mhz:.1f} MHz')

    fig.suptitle(f'Time frame {frame}')
    plt.tight_layout()

anim = FuncAnimation(fig, update, frames=range(100), interval=100)
anim.save('multi_freq.mp4', writer='ffmpeg', fps=10)
```

## Dynamic Spectrum Animation

Animate a moving window through a dynamic spectrum:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Get full dynamic spectrum
l_idx, m_idx = 512, 512
dyn_spec = ds.radport.dynamic_spectrum(l_idx=l_idx, m_idx=m_idx)

# Create sliding window animation
fig, ax = plt.subplots(figsize=(10, 6))
window_size = 20

def update(frame):
    ax.clear()
    start = frame
    end = min(frame + window_size, len(dyn_spec.coords['time']))

    ax.imshow(
        dyn_spec[start:end].T,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Dynamic Spectrum: frames {start}-{end}')

frames = range(0, len(dyn_spec.coords['time']) - window_size, 5)
anim = FuncAnimation(fig, update, frames=frames, interval=100)
```

## Side-by-Side Comparison

Animate comparisons (e.g., before/after processing):

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create two versions (e.g., raw vs. processed)
ds_raw = ds
ds_processed = ds.radport.time_average()  # Or any processing

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def update(frame):
    for ax in axes:
        ax.clear()

    # Raw
    axes[0].imshow(
        ds_raw['SKY'].isel(time=frame, frequency=10).values,
        origin='lower',
        cmap='viridis'
    )
    axes[0].set_title('Raw')

    # Processed (if time-independent, show same frame)
    axes[1].imshow(
        ds_processed.isel(frequency=10).values,
        origin='lower',
        cmap='viridis'
    )
    axes[1].set_title('Time Average')

    fig.suptitle(f'Frame {frame}')

anim = FuncAnimation(fig, update, frames=range(100), interval=100)
```

## Animation Performance Tips

### 1. Use Appropriate Time Steps

```python
# Animate every 5th frame for faster rendering
time_indices = range(0, 100, 5)
anim = ds.radport.animate_time(
    freq_idx=10,
    time_min=0,
    time_max=100,
    interval=100
)
```

### 2. Reduce Spatial Resolution

```python
# Downsample before animating
ds_small = ds.coarsen(l=2, m=2, boundary='trim').mean()
anim = ds_small.radport.animate_time(freq_idx=10)
```

### 3. Precompute Frames

```python
# Precompute frames for complex operations
frames = []
for t in range(100):
    frame = ds['SKY'].isel(time=t, frequency=10).values
    # Apply any processing
    frames.append(frame)

# Now animate precomputed frames
# (custom animation code)
```

## Saving Options

### MP4 (Recommended)

```python
anim.save('output.mp4', writer='ffmpeg', fps=10, bitrate=1800)
```

### GIF

```python
anim.save('output.gif', writer='pillow', fps=5)
```

### High Quality

```python
anim.save(
    'output.mp4',
    writer='ffmpeg',
    fps=30,
    dpi=300,
    bitrate=5000,
    extra_args=['-vcodec', 'libx264', '-crf', '15']
)
```

## Requirements

Animations require matplotlib and ffmpeg:

```bash
# Install ffmpeg
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Or use pillow for GIF (already included)
pip install pillow
```

## Example: Full Pipeline

```python
import ovro_lwa_portal as ovro
import matplotlib.pyplot as plt

# Load data
ds = ovro.open_dataset("path/to/data.zarr")

# Detect sources
peaks = ds.radport.find_peaks(time_idx=0, freq_idx=10, threshold=5.0)

# Create animation with overlays
anim = ds.radport.animate_time(freq_idx=10, interval=100)

# Save
anim.save('ovro_animation.mp4', writer='ffmpeg', fps=10)

print("Animation saved!")
```

## Next Steps

- Learn about [visualization methods](visualization.md)
- Try the [transient analysis tutorial](../tutorials/transient-analysis.md)
- Explore the [API reference](../api/radport-accessor.md)
