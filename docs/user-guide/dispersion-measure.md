# Dispersion Measure Correction

Dispersion measure (DM) correction is essential for pulsar and fast radio burst
(FRB) observations.

## What is Dispersion?

Radio waves traveling through ionized plasma experience a frequency-dependent
delay:

$$
\Delta t = 4.15 \times 10^{3} \, \text{ms} \times \text{DM} \times \left( f_{\text{low}}^{-2} - f_{\text{high}}^{-2} \right)
$$

where:

- DM is the dispersion measure in pc cm⁻³
- f is frequency in MHz

This causes pulses to arrive at different times across the frequency band,
"smearing" the signal.

## Dispersion Correction Methods

### Method 1: Coherent Dedispersion

!!! note "Coming Soon" Coherent dedispersion methods are planned for future
releases. Track progress in
[GitHub Issue #85](https://github.com/uw-ssec/ovro-lwa-portal/issues/85).

### Method 2: Incoherent Dedispersion

Apply time shifts to align frequencies:

```python
import numpy as np
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

def dedisperse_incoherent(ds, dm, ref_freq_mhz=None):
    """
    Apply incoherent dedispersion.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    dm : float
        Dispersion measure in pc cm^-3
    ref_freq_mhz : float, optional
        Reference frequency in MHz (default: highest frequency)

    Returns
    -------
    xarray.Dataset
        Dedispersed dataset
    """
    freqs_hz = ds.coords['frequency'].values
    freqs_mhz = freqs_hz / 1e6

    if ref_freq_mhz is None:
        ref_freq_mhz = freqs_mhz.max()

    # Calculate delays
    delays_ms = 4.15e3 * dm * (freqs_mhz**-2 - ref_freq_mhz**-2)

    # Convert to time samples
    time_vals = ds.coords['time'].values
    dt = np.median(np.diff(time_vals))  # Time resolution in days
    dt_ms = dt * 86400 * 1000  # Convert to milliseconds

    delays_samples = np.round(delays_ms / dt_ms).astype(int)

    # Apply shifts
    dedispersed = ds.copy()
    for freq_idx, shift in enumerate(delays_samples):
        if shift != 0:
            dedispersed['SKY'][:, freq_idx, :, :] = np.roll(
                ds['SKY'][:, freq_idx, :, :].values,
                shift=-shift,
                axis=0
            )

    return dedispersed


# Example: Dedisperse for Crab pulsar (DM ≈ 56.7)
ds_dedispersed = dedisperse_incoherent(ds, dm=56.7)
```

## Crab Pulsar Example

The Crab pulsar is an excellent test case with DM ≈ 56.7 pc cm⁻³.

### Before Dedispersion

```python
import matplotlib.pyplot as plt

# Extract dynamic spectrum at pulsar location
l_idx, m_idx = ds.radport.nearest_lm_idx(l=0.0, m=0.0)  # Adjust coordinates
dyn_spec = ds.radport.dynamic_spectrum(l_idx=l_idx, m_idx=m_idx)

# Plot
ds.radport.plot_dynamic_spectrum(dyn_spec)
plt.title('Before Dedispersion')
plt.show()
```

You should see diagonal features due to dispersion.

### After Dedispersion

```python
# Dedisperse
ds_dedispersed = dedisperse_incoherent(ds, dm=56.7)

# Extract dynamic spectrum
dyn_spec_dd = ds_dedispersed.radport.dynamic_spectrum(l_idx=l_idx, m_idx=m_idx)

# Plot
ds_dedispersed.radport.plot_dynamic_spectrum(dyn_spec_dd)
plt.title('After Dedispersion (DM=56.7)')
plt.show()
```

Pulses should now align vertically across all frequencies.

## DM Optimization

Find the optimal DM by maximizing SNR:

```python
def optimize_dm(ds, l_idx, m_idx, dm_range):
    """
    Find optimal DM by maximizing peak SNR.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    l_idx, m_idx : int
        Spatial location
    dm_range : array-like
        DM values to test (pc cm^-3)

    Returns
    -------
    float
        Optimal DM
    """
    snrs = []

    for dm in dm_range:
        # Dedisperse
        ds_dd = dedisperse_incoherent(ds, dm=dm)

        # Get light curve (summed over frequency)
        lc = ds_dd.radport.light_curve(l_idx=l_idx, m_idx=m_idx)

        # Calculate SNR
        mean_val = lc.mean().values
        std_val = lc.std().values
        max_val = lc.max().values

        snr = (max_val - mean_val) / std_val if std_val > 0 else 0
        snrs.append(snr)

    # Find maximum
    best_idx = np.argmax(snrs)
    best_dm = dm_range[best_idx]

    return best_dm, np.array(snrs)


# Test DM range around known value
dm_range = np.linspace(50, 65, 30)
best_dm, snrs = optimize_dm(ds, l_idx, m_idx, dm_range)

# Plot DM curve
plt.figure(figsize=(10, 6))
plt.plot(dm_range, snrs, 'o-')
plt.axvline(best_dm, color='red', linestyle='--', label=f'Best DM={best_dm:.1f}')
plt.xlabel('Dispersion Measure (pc cm⁻³)')
plt.ylabel('SNR')
plt.title('DM Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Fast Radio Bursts (FRBs)

For FRB detection and analysis:

```python
# High DM range for FRBs (typically 100-3000 pc cm^-3)
dm_range_frb = np.linspace(100, 2000, 50)

# Search for FRBs
def search_frbs(ds, dm_range, snr_threshold=8.0):
    """Search for FRB candidates."""
    candidates = []

    for l_idx in range(0, ds.dims['l'], 50):  # Spatial search
        for m_idx in range(0, ds.dims['m'], 50):
            best_dm, snrs = optimize_dm(ds, l_idx, m_idx, dm_range)

            if snrs.max() > snr_threshold:
                candidates.append({
                    'l_idx': l_idx,
                    'm_idx': m_idx,
                    'dm': best_dm,
                    'snr': snrs.max()
                })

    return candidates

# Run search (may take time!)
# candidates = search_frbs(ds, dm_range_frb)
```

## Visualization Tools

### Waterfall Plot

Classic pulsar visualization:

```python
def plot_waterfall(ds, l_idx, m_idx, dm=None):
    """Create waterfall plot."""
    if dm is not None:
        ds = dedisperse_incoherent(ds, dm=dm)

    dyn_spec = ds.radport.dynamic_spectrum(l_idx=l_idx, m_idx=m_idx)

    plt.figure(figsize=(12, 8))
    plt.imshow(
        dyn_spec.T,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='nearest'
    )
    plt.xlabel('Time')
    plt.ylabel('Frequency Channel')
    plt.colorbar(label='Intensity')
    plt.title(f'Waterfall Plot (DM={dm if dm else 0})')
    plt.show()

# Plot
plot_waterfall(ds, l_idx=512, m_idx=512, dm=56.7)
```

### Integrated Profile

Fold the data at pulsar period:

```python
# For known pulsar period
def fold_pulsar(ds, l_idx, m_idx, period_s, dm=None):
    """Fold pulsar light curve at known period."""
    if dm is not None:
        ds = dedisperse_incoherent(ds, dm=dm)

    # Get light curve summed over frequency
    lc = ds.radport.light_curve(l_idx=l_idx, m_idx=m_idx)

    # Get time in seconds
    times = ds.coords['time'].values * 86400  # MJD to seconds

    # Calculate phase
    phase = (times % period_s) / period_s

    # Bin by phase
    n_bins = 50
    phase_bins = np.linspace(0, 1, n_bins + 1)
    profile = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        profile[i] = lc.values[mask].mean()

    return np.linspace(0, 1, n_bins), profile

# Example for Crab (period ≈ 33 ms)
phase, profile = fold_pulsar(ds, l_idx, m_idx, period_s=0.033, dm=56.7)

plt.figure(figsize=(10, 6))
plt.plot(phase, profile, 'o-')
plt.xlabel('Pulse Phase')
plt.ylabel('Intensity')
plt.title('Folded Pulse Profile')
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices

1. **Know Your Source**: Start with published DM values when available

2. **Frequency Range**: Wider bands increase dispersion sensitivity

3. **Time Resolution**: Ensure adequate sampling for your science case

4. **SNR Optimization**: Use DM optimization for unknown sources

5. **Validation**: Visually inspect dedispersed dynamic spectra

## Next Steps

- Try the [pulsar dedispersion tutorial](../tutorials/pulsar-dedispersion.md)
- Learn about [transient analysis](../tutorials/transient-analysis.md)
- Explore the [API reference](../api/radport-accessor.md)
