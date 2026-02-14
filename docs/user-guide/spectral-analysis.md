# Spectral Analysis

The `radport` accessor provides methods for analyzing spectral properties of radio sources.

## Frequency Spectra

### Basic Spectrum

Extract frequency spectrum at a spatial location:

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

# Spectrum at a pixel
spec = ds.radport.spectrum(l_idx=512, m_idx=512)

# Plot it
ds.radport.plot_spectrum(spec)
```

### Spatially Averaged Spectrum

```python
# Average over entire field
spec_avg = ds.radport.spectrum()

# Average over a region
spec_region = ds.radport.spectrum(
    l_min=500, l_max=524,
    m_min=500, m_max=524
)

# Average over time as well
spec_full = ds.radport.spectrum(
    l_min=500, l_max=524,
    m_min=500, m_max=524,
    time_min=0, time_max=100
)
```

## Spectral Index

The spectral index (α) describes how flux density varies with frequency: S ∝ ν^α

### Point Spectral Index

Calculate spectral index at a location:

```python
# Spectral index between two frequencies
alpha = ds.radport.spectral_index(
    freq_idx1=0,
    freq_idx2=10,
    l_idx=512,
    m_idx=512
)

print(f"Spectral index: {alpha:.2f}")
```

### Spectral Index Map

Create a map of spectral indices across the field:

```python
# Calculate spectral index map
alpha_map = ds.radport.spectral_index_map(
    freq_idx1=0,
    freq_idx2=10,
    time_idx=0
)

# Plot it
ds.radport.plot_spectral_index_map(
    alpha_map,
    vmin=-2,
    vmax=2,
    cmap='RdBu_r'
)
```

**Interpretation:**

- α ≈ -0.7: Typical synchrotron emission (many AGN)
- α ≈ 0: Flat spectrum sources (some quasars)
- α > 0: Inverted spectrum (self-absorbed sources)
- α < -1: Steep spectrum sources

### Multi-Frequency Spectral Index

For more than two frequencies:

```python
# Use multiple frequency pairs
freq_pairs = [(0, 5), (5, 10), (10, 15)]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (f1, f2) in zip(axes, freq_pairs):
    alpha_map = ds.radport.spectral_index_map(
        freq_idx1=f1,
        freq_idx2=f2,
        time_idx=0
    )
    ds.radport.plot_spectral_index_map(alpha_map, ax=ax)
    ax.set_title(f'α between freq {f1} and {f2}')

plt.tight_layout()
plt.show()
```

## Integrated Flux

Calculate total flux density in a region:

```python
# Integrated flux at one frequency/time
flux = ds.radport.integrated_flux(
    time_idx=0,
    freq_idx=10,
    l_min=500, l_max=524,
    m_min=500, m_max=524
)

print(f"Integrated flux: {flux:.3e} Jy")
```

### Flux Evolution

Track how flux changes with time:

```python
import matplotlib.pyplot as plt

# Calculate flux for each time step
fluxes = []
times = ds.coords['time'].values

for t_idx in range(len(times)):
    flux = ds.radport.integrated_flux(
        time_idx=t_idx,
        freq_idx=10,
        l_min=500, l_max=524,
        m_min=500, m_max=524
    )
    fluxes.append(flux)

# Plot flux evolution
plt.figure(figsize=(10, 6))
plt.plot(times, fluxes, 'o-')
plt.xlabel('Time (MJD)')
plt.ylabel('Integrated Flux (Jy)')
plt.title('Flux Evolution')
plt.grid(True, alpha=0.3)
plt.show()
```

### Spectral Energy Distribution (SED)

```python
# Calculate flux at each frequency
freqs = ds.coords['frequency'].values / 1e6  # Convert to MHz
fluxes = []

for f_idx in range(len(freqs)):
    flux = ds.radport.integrated_flux(
        time_idx=0,
        freq_idx=f_idx,
        l_min=500, l_max=524,
        m_min=500, m_max=524
    )
    fluxes.append(flux)

# Plot SED
plt.figure(figsize=(10, 6))
plt.loglog(freqs, fluxes, 'o-')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux Density (Jy)')
plt.title('Spectral Energy Distribution')
plt.grid(True, alpha=0.3, which='both')
plt.show()
```

## Spectral Fitting

Fit power law to spectral data:

```python
import numpy as np
from scipy.optimize import curve_fit

# Power law model: S = A * (ν/ν0)^α
def power_law(freq, amplitude, alpha, freq0=40.0):
    return amplitude * (freq / freq0)**alpha

# Get flux densities
freqs = ds.coords['frequency'].values / 1e6
fluxes = [
    ds.radport.integrated_flux(
        time_idx=0, freq_idx=i,
        l_min=500, l_max=524,
        m_min=500, m_max=524
    )
    for i in range(len(freqs))
]

# Fit
popt, pcov = curve_fit(power_law, freqs, fluxes)
amplitude, alpha = popt

print(f"Amplitude: {amplitude:.3e} Jy")
print(f"Spectral index: {alpha:.2f}")

# Plot fit
plt.figure(figsize=(10, 6))
plt.loglog(freqs, fluxes, 'o', label='Data')
freq_fit = np.linspace(freqs.min(), freqs.max(), 100)
plt.loglog(freq_fit, power_law(freq_fit, *popt), '-', label=f'Fit (α={alpha:.2f})')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux Density (Jy)')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()
```

## Spectral Variability

Analyze how spectra change over time:

```python
# Spectrum at different times
times_to_plot = [0, 50, 100]
freqs = ds.coords['frequency'].values / 1e6

plt.figure(figsize=(10, 6))

for t_idx in times_to_plot:
    spec = ds.radport.spectrum(
        l_idx=512, m_idx=512,
        time_min=t_idx, time_max=t_idx+1
    )
    plt.plot(freqs, spec.values, 'o-', label=f'Time {t_idx}')

plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity')
plt.title('Spectral Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Frequency Averaging

Average over frequency bands:

```python
# Average over low frequencies
low_freq = ds.radport.frequency_average(freq_min=0, freq_max=5)

# Average over high frequencies
high_freq = ds.radport.frequency_average(freq_min=10, freq_max=15)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ds.radport.plot_frequency_average(low_freq, time_idx=0, ax=axes[0])
axes[0].set_title('Low Frequency')
ds.radport.plot_frequency_average(high_freq, time_idx=0, ax=axes[1])
axes[1].set_title('High Frequency')
plt.tight_layout()
plt.show()
```

## Best Practices

1. **Quality Check**: Always check for valid data before spectral analysis

```python
# Check data quality
fraction = ds.radport.finite_fraction(time_idx=0, freq_idx=10)
if fraction < 0.9:
    print(f"Warning: Only {fraction*100:.1f}% valid data")
```

2. **Background Subtraction**: Consider subtracting background for accurate flux measurements

3. **Beam Correction**: Account for primary beam attenuation if analyzing off-axis sources

4. **RFI Flagging**: Identify and flag radio frequency interference before spectral analysis

## Next Steps

- Learn about [source detection](source-detection.md)
- Try the [spectral mapping tutorial](../tutorials/spectral-mapping.md)
- Explore the [API reference](../api/radport-accessor.md)
