# Pulsar Dedispersion Tutorial

This tutorial demonstrates how to detect and analyze pulsars in OVRO-LWA data using the
Crab pulsar (B0531+21, DM ≈ 56.7 pc cm⁻³) as the example target. You will learn how
to apply incoherent dedispersion, optimize the dispersion measure, and fold the data
at the pulsar period.

## Background

Radio pulses traveling through the ionized interstellar medium (ISM) experience a
frequency-dependent delay described by:

$$
\Delta t = 4.15 \times 10^{3} \, \text{ms} \times \text{DM} \times \left( f_{\text{low}}^{-2} - f_{\text{high}}^{-2} \right)
$$

where DM (dispersion measure) is the integrated electron column density in pc cm⁻³
and *f* is frequency in MHz. Lower frequencies arrive later, causing a characteristic
diagonal sweep in the dynamic spectrum.

## Prerequisites

```python
import ovro_lwa_portal as ovro
import numpy as np
import matplotlib.pyplot as plt
```

## Step 1: Load Data and Locate the Pulsar

```python
ds = ovro.open_dataset("path/to/crab_data.zarr")

# Check available frequencies
freqs_mhz = ds.coords["frequency"].values / 1e6
print(f"Frequency range: {freqs_mhz.min():.1f} – {freqs_mhz.max():.1f} MHz")
print(f"Number of time steps: {ds.sizes['time']}")
```

Locate the Crab pulsar. If WCS coordinates are available, use RA/Dec:

```python
if ds.radport.has_wcs:
    # Crab Nebula: RA = 83.633°, Dec = 22.014°
    l_idx, m_idx = ds.radport.coords_to_pixel(ra=83.633, dec=22.014)
    print(f"Crab pulsar at pixel: l={l_idx}, m={m_idx}")
else:
    # Fall back to the image center or a known pixel position
    l_idx = ds.sizes["l"] // 2
    m_idx = ds.sizes["m"] // 2
    print(f"Using image center: l={l_idx}, m={m_idx}")
```

## Step 2: View the Dispersed Dynamic Spectrum

Extract and plot the dynamic spectrum at the pulsar position:

```python
l_val = float(ds.coords["l"].values[l_idx])
m_val = float(ds.coords["m"].values[m_idx])

dyn = ds.radport.dynamic_spectrum(l=l_val, m=m_val)
ds.radport.plot_dynamic_spectrum(dyn)
plt.title("Before Dedispersion")
plt.show()
```

You should see the characteristic diagonal sweep: low-frequency emission arriving
later than high-frequency emission.

## Step 3: Incoherent Dedispersion

Incoherent dedispersion corrects the dispersion delay by shifting each frequency
channel in time. Define the dedispersion function:

```python
def dedisperse_incoherent(ds, dm, ref_freq_mhz=None):
    """Apply incoherent dedispersion to a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    dm : float
        Dispersion measure in pc cm⁻³.
    ref_freq_mhz : float, optional
        Reference frequency in MHz. Defaults to the highest frequency.

    Returns
    -------
    xarray.Dataset
        Dedispersed dataset.
    """
    freqs_hz = ds.coords["frequency"].values
    freqs_mhz = freqs_hz / 1e6

    if ref_freq_mhz is None:
        ref_freq_mhz = freqs_mhz.max()

    # Time delay at each frequency relative to the reference
    delays_ms = 4.15e3 * dm * (freqs_mhz**-2 - ref_freq_mhz**-2)

    # Convert delays to integer sample shifts
    time_vals = ds.coords["time"].values
    dt_days = np.median(np.diff(time_vals))
    dt_ms = dt_days * 86400 * 1000
    delays_samples = np.round(delays_ms / dt_ms).astype(int)

    # Apply circular shifts per frequency channel
    dedispersed = ds.copy()
    for freq_idx, shift in enumerate(delays_samples):
        if shift != 0:
            dedispersed["SKY"][:, freq_idx, :, :] = np.roll(
                ds["SKY"][:, freq_idx, :, :].values,
                shift=-shift,
                axis=0,
            )

    return dedispersed
```

## Step 4: Apply Known DM

Apply the Crab pulsar's known DM:

```python
ds_dd = dedisperse_incoherent(ds, dm=56.7)
```

Compare before and after:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before
dyn_before = ds.radport.dynamic_spectrum(l=l_val, m=m_val)
ds.radport.plot_dynamic_spectrum(dyn_before, ax=axes[0])
axes[0].set_title("Before Dedispersion")

# After
dyn_after = ds_dd.radport.dynamic_spectrum(l=l_val, m=m_val)
ds_dd.radport.plot_dynamic_spectrum(dyn_after, ax=axes[1])
axes[1].set_title("After Dedispersion (DM = 56.7)")

plt.tight_layout()
plt.show()
```

After dedispersion, pulses should align vertically across all frequencies.

## Step 5: DM Optimization

When the DM is not known precisely, search for the value that maximizes the
signal-to-noise ratio of the dedispersed pulse:

```python
def optimize_dm(ds, l_idx, m_idx, dm_range):
    """Find the optimal DM by maximizing peak SNR.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    l_idx, m_idx : int
        Spatial pixel indices of the source.
    dm_range : array-like
        DM values to test in pc cm⁻³.

    Returns
    -------
    tuple
        (best_dm, snr_array)
    """
    l_val = float(ds.coords["l"].values[l_idx])
    m_val = float(ds.coords["m"].values[m_idx])
    snrs = []

    for dm in dm_range:
        ds_dd = dedisperse_incoherent(ds, dm=dm)
        lc = ds_dd.radport.light_curve(l=l_val, m=m_val)

        mean_val = float(lc.mean().values)
        std_val = float(lc.std().values)
        max_val = float(lc.max().values)

        snr = (max_val - mean_val) / std_val if std_val > 0 else 0
        snrs.append(snr)

    best_idx = np.argmax(snrs)
    return dm_range[best_idx], np.array(snrs)
```

Run the optimization around the expected DM:

```python
dm_range = np.linspace(50, 65, 30)
best_dm, snrs = optimize_dm(ds, l_idx, m_idx, dm_range)

print(f"Optimal DM: {best_dm:.1f} pc cm⁻³")

plt.figure(figsize=(10, 5))
plt.plot(dm_range, snrs, "o-")
plt.axvline(best_dm, color="red", linestyle="--", label=f"Best DM = {best_dm:.1f}")
plt.xlabel("Dispersion Measure (pc cm⁻³)")
plt.ylabel("Peak SNR")
plt.title("DM Optimization")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 6: Pulse Profile Folding

Fold the dedispersed light curve at the known pulsar period to build an averaged
pulse profile. The Crab pulsar has a period of approximately 33 ms.

```python
def fold_pulsar(ds, l_idx, m_idx, period_s, dm=None, n_bins=50):
    """Fold a light curve at a known pulsar period.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    l_idx, m_idx : int
        Spatial pixel indices.
    period_s : float
        Pulsar period in seconds.
    dm : float, optional
        If provided, dedisperse before folding.
    n_bins : int
        Number of phase bins.

    Returns
    -------
    tuple
        (phase_centers, profile)
    """
    if dm is not None:
        ds = dedisperse_incoherent(ds, dm=dm)

    l_val = float(ds.coords["l"].values[l_idx])
    m_val = float(ds.coords["m"].values[m_idx])
    lc = ds.radport.light_curve(l=l_val, m=m_val)

    # Convert MJD times to seconds
    times_s = ds.coords["time"].values * 86400

    # Compute pulse phase
    phase = (times_s % period_s) / period_s

    # Bin by phase
    phase_bins = np.linspace(0, 1, n_bins + 1)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if mask.any():
            profile[i] = float(lc.values[mask].mean())

    phase_centers = 0.5 * (phase_bins[:-1] + phase_bins[1:])
    return phase_centers, profile
```

Apply it to the Crab pulsar:

```python
phase, profile = fold_pulsar(ds, l_idx, m_idx, period_s=0.033, dm=56.7)

plt.figure(figsize=(10, 5))
plt.plot(phase, profile, "o-")
plt.xlabel("Pulse Phase")
plt.ylabel("Intensity")
plt.title("Folded Crab Pulse Profile (P ≈ 33 ms, DM ≈ 56.7)")
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 7: Publication-Quality Summary Figure

Combine all results into a single multi-panel figure:

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Dispersed dynamic spectrum
dyn_before = ds.radport.dynamic_spectrum(l=l_val, m=m_val)
ds.radport.plot_dynamic_spectrum(dyn_before, ax=axes[0, 0])
axes[0, 0].set_title("(a) Dispersed")

# Panel 2: Dedispersed dynamic spectrum
ds_dd = dedisperse_incoherent(ds, dm=best_dm)
dyn_after = ds_dd.radport.dynamic_spectrum(l=l_val, m=m_val)
ds_dd.radport.plot_dynamic_spectrum(dyn_after, ax=axes[0, 1])
axes[0, 1].set_title(f"(b) Dedispersed (DM = {best_dm:.1f})")

# Panel 3: DM optimization curve
axes[1, 0].plot(dm_range, snrs, "o-")
axes[1, 0].axvline(best_dm, color="red", linestyle="--")
axes[1, 0].set_xlabel("DM (pc cm⁻³)")
axes[1, 0].set_ylabel("Peak SNR")
axes[1, 0].set_title("(c) DM Optimization")
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Folded pulse profile
phase, profile = fold_pulsar(ds, l_idx, m_idx, period_s=0.033, dm=best_dm)
axes[1, 1].plot(phase, profile, "o-")
axes[1, 1].set_xlabel("Pulse Phase")
axes[1, 1].set_ylabel("Intensity")
axes[1, 1].set_title("(d) Folded Pulse Profile")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Crab Pulsar Analysis", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

## Summary

This tutorial covered:

1. Locating a pulsar in the image (pixel or WCS coordinates)
2. Viewing the dispersed dynamic spectrum
3. Implementing incoherent dedispersion
4. Applying a known DM and comparing before/after
5. Optimizing DM by maximizing peak SNR
6. Folding the dedispersed light curve at the pulsar period

## Next Steps

- [Dispersion Measure Correction Guide](../user-guide/dispersion-measure.md) -- FRB search and advanced DM techniques
- [Transient Analysis Tutorial](transient-analysis.md) -- Detecting transient sources
- [Spectral Mapping Tutorial](spectral-mapping.md) -- Spectral properties of sources
- [API Reference](../api/radport-accessor.md) -- Full method documentation
