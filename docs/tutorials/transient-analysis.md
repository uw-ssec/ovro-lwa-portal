# Transient Analysis Tutorial

This tutorial walks through an end-to-end workflow for detecting and
characterizing transient radio sources in OVRO-LWA data, covering data loading,
reference image creation, difference imaging, source detection, and light curve
analysis.

## Prerequisites

```python
import ovro_lwa_portal as ovro
import numpy as np
import matplotlib.pyplot as plt
```

## Step 1: Load and Inspect the Data

```python
ds = ovro.open_dataset("path/to/data.zarr")

# Check dimensions
print(ds)
print(f"Time steps:  {ds.sizes['time']}")
print(f"Frequencies: {ds.sizes['frequency']}")
print(f"Image size:  {ds.sizes['l']} x {ds.sizes['m']}")

# Verify data quality at the first frame
fraction = ds.radport.finite_fraction(time_idx=0, freq_idx=0)
print(f"Valid data fraction: {fraction * 100:.1f}%")
```

Pick a frequency to work with throughout the tutorial:

```python
freqs_mhz = ds.coords["frequency"].values / 1e6
print(f"Available frequencies: {freqs_mhz} MHz")

# Choose a frequency
freq_idx = ds.radport.nearest_freq_idx(50.0)
print(f"Using freq_idx={freq_idx} ({freqs_mhz[freq_idx]:.1f} MHz)")
```

## Step 2: Create a Reference Image

A time-averaged image serves as the quiescent baseline. Sources that appear only
in individual frames (but not in the average) are transient candidates.

```python
# Average over all time steps
ref = ds.radport.time_average()

# Visualize the reference at our chosen frequency
ds.radport.plot_time_average(ref, freq_idx=freq_idx)
plt.title("Time-Averaged Reference Image")
plt.show()
```

## Step 3: Browse Individual Frames

Scan through a few frames to see if anything stands out:

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, t_idx in zip(axes, [0, 50, 100]):
    ds.radport.plot(time_idx=t_idx, freq_idx=freq_idx, ax=ax)
    ax.set_title(f"t={t_idx}")

plt.tight_layout()
plt.show()
```

## Step 4: Difference Imaging

Subtract the reference to isolate time-variable emission:

```python
# Use the built-in diff method
fig = ds.radport.plot_diff(mode="time", time_idx=50, freq_idx=freq_idx)
plt.title("Difference Image (frame 50 − average)")
plt.show()
```

You can also compute the difference manually for more control:

```python
current = ds["SKY"].isel(time=50, frequency=freq_idx, polarization=0).values
reference = ref.isel(frequency=freq_idx, polarization=0).values

diff = current - reference

plt.figure(figsize=(8, 8))
plt.imshow(diff, origin="lower", cmap="RdBu_r")
plt.colorbar(label="Flux Difference (Jy)")
plt.title("Manual Difference Image")
plt.show()
```

Red/blue spots indicate brightening/fading relative to the mean.

## Step 5: Source Detection with SNR Maps

Quantify significance using a signal-to-noise map:

```python
# SNR map at a specific frame
snr = ds.radport.snr_map(time_idx=50, freq_idx=freq_idx)
ds.radport.plot_snr_map(snr, vmin=0, vmax=10, cmap="hot")
plt.title("SNR Map")
plt.show()
```

Find peaks above a detection threshold:

```python
peaks = ds.radport.find_peaks(
    time_idx=50,
    freq_idx=freq_idx,
    threshold=5.0,
)
print(f"Detected {len(peaks)} sources above SNR=5")
```

Overlay detections on the sky image:

```python
ds.radport.plot(time_idx=50, freq_idx=freq_idx)
ax = plt.gca()

if len(peaks) > 0:
    l_coords, m_coords = zip(*peaks)
    ax.scatter(
        l_coords, m_coords,
        marker="o", facecolors="none", edgecolors="red",
        s=200, linewidths=2, label=f"{len(peaks)} detections",
    )
    ax.legend()

plt.show()
```

## Step 6: Extract Light Curves

For each candidate, extract a light curve to check for time variability:

```python
fig, axes = plt.subplots(min(len(peaks), 3), 1, figsize=(10, 8), sharex=True)
if not hasattr(axes, "__iter__"):
    axes = [axes]

for ax, (l_idx, m_idx) in zip(axes, peaks[:3]):
    l_val = float(ds.coords["l"].values[l_idx])
    m_val = float(ds.coords["m"].values[m_idx])

    lc = ds.radport.light_curve(l=l_val, m=m_val)
    ds.radport.plot_light_curve(lc, ax=ax)
    ax.set_title(f"Source at l={l_val:.3f}, m={m_val:.3f}")

plt.tight_layout()
plt.show()
```

## Step 7: Dynamic Spectrum Analysis

Inspect the frequency-time structure of a candidate:

```python
l_val = float(ds.coords["l"].values[peaks[0][0]])
m_val = float(ds.coords["m"].values[peaks[0][1]])

dyn = ds.radport.dynamic_spectrum(l=l_val, m=m_val)
ds.radport.plot_dynamic_spectrum(dyn)
plt.title("Dynamic Spectrum of Candidate 0")
plt.show()
```

A broadband brightening across all frequencies suggests an astrophysical
transient, while narrow-band features may indicate radio frequency interference
(RFI).

## Step 8: Variability Classification

Compute a variability index (standard deviation / mean) to rank candidates:

```python
variability = []

for l_idx, m_idx in peaks:
    l_val = float(ds.coords["l"].values[l_idx])
    m_val = float(ds.coords["m"].values[m_idx])

    lc = ds.radport.light_curve(l=l_val, m=m_val)
    mean_val = float(lc.mean().values)
    std_val = float(lc.std().values)

    v_idx = std_val / mean_val if mean_val > 0 else 0.0
    variability.append({
        "l_idx": l_idx, "m_idx": m_idx,
        "mean_flux": mean_val, "variability": v_idx,
    })

# Sort by variability
variability.sort(key=lambda x: x["variability"], reverse=True)

print("Top variable sources:")
for v in variability[:5]:
    print(f"  l={v['l_idx']}, m={v['m_idx']}: "
          f"V={v['variability']:.2f}, mean={v['mean_flux']:.3e}")
```

!!! tip A variability index above ~0.5 is a strong indicator of genuine
time-variable emission, though the exact threshold depends on your noise level.

## Step 9: Multi-Frequency Confirmation

Confirm that candidates appear at multiple frequencies to rule out instrumental
artifacts:

```python
freq_indices = list(range(ds.sizes["frequency"]))

confirmed = []
for src in variability[:5]:
    detections = 0
    for fi in freq_indices:
        snr_map = ds.radport.snr_map(time_idx=50, freq_idx=fi)
        if snr_map[src["m_idx"], src["l_idx"]] > 5.0:
            detections += 1

    frac = detections / len(freq_indices)
    print(f"Source l={src['l_idx']}, m={src['m_idx']}: "
          f"detected in {detections}/{len(freq_indices)} bands ({frac:.0%})")

    if frac >= 0.5:
        confirmed.append(src)

print(f"\n{len(confirmed)} sources confirmed across ≥50% of frequency bands")
```

## Summary

This tutorial covered:

1. Loading data and checking quality
2. Creating a time-averaged reference image
3. Difference imaging to isolate transients
4. SNR-based source detection
5. Light curve extraction and dynamic spectrum analysis
6. Variability classification and multi-frequency confirmation

## Next Steps

- [Pulsar Dedispersion Tutorial](pulsar-dedispersion.md) -- Analyze dispersed
  pulsars
- [Spectral Mapping Tutorial](spectral-mapping.md) -- Map spectral properties
- [Source Detection Guide](../user-guide/source-detection.md) -- Detailed
  detection methods
- [API Reference](../api/radport-accessor.md) -- Full method documentation
