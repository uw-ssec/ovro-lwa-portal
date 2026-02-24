# Spectral Mapping Tutorial

This tutorial demonstrates how to create spectral index maps, build spectral energy
distributions (SEDs), fit power laws, and classify radio sources by their spectral
properties using the `radport` accessor.

## Background

The spectral index *α* describes how a source's flux density *S* varies with
frequency *ν*:

$$
S \propto \nu^{\alpha}
$$

Typical spectral index values at low radio frequencies:

| α | Interpretation | Examples |
|---|---|---|
| ≈ −0.7 | Optically-thin synchrotron | Most AGN, supernova remnants |
| ≈ 0 | Flat spectrum | Compact quasar cores |
| > 0 | Inverted / self-absorbed | Gigahertz-peaked sources |
| < −1 | Ultra-steep spectrum | Aged electron populations, high-*z* radio galaxies |

## Prerequisites

```python
import ovro_lwa_portal as ovro
import numpy as np
import matplotlib.pyplot as plt
```

## Step 1: Load Data and Inspect Frequencies

```python
ds = ovro.open_dataset("path/to/data.zarr")

freqs_mhz = ds.coords["frequency"].values / 1e6
print(f"Frequency range: {freqs_mhz.min():.1f} – {freqs_mhz.max():.1f} MHz")
print(f"Number of channels: {len(freqs_mhz)}")
```

Visualize the field at the lowest and highest frequencies:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ds.radport.plot(time_idx=0, freq_idx=0, ax=axes[0])
axes[0].set_title(f"{freqs_mhz[0]:.1f} MHz")

ds.radport.plot(time_idx=0, freq_idx=-1, ax=axes[1])
axes[1].set_title(f"{freqs_mhz[-1]:.1f} MHz")

plt.tight_layout()
plt.show()
```

## Step 2: Create a Spectral Index Map

Compute *α* between the lowest and highest frequency channels:

```python
alpha_map = ds.radport.spectral_index_map(
    freq_idx1=0,
    freq_idx2=len(freqs_mhz) - 1,
    time_idx=0,
)

ds.radport.plot_spectral_index_map(
    alpha_map,
    vmin=-2,
    vmax=2,
    cmap="RdBu_r",
)
plt.title(f"Spectral Index ({freqs_mhz[0]:.0f}–{freqs_mhz[-1]:.0f} MHz)")
plt.show()
```

Blue regions indicate steep-spectrum sources (α < 0) while red indicates
flat or inverted spectra (α ≥ 0).

## Step 3: Detect Sources and Measure Spectral Indices

Combine source detection with spectral index measurement:

```python
# Detect sources at the lowest frequency
peaks = ds.radport.find_peaks(time_idx=0, freq_idx=0, threshold=5.0)
print(f"Detected {len(peaks)} sources")

# Measure spectral index at each peak
catalog = []
for l_idx, m_idx in peaks:
    alpha = ds.radport.spectral_index(
        freq_idx1=0,
        freq_idx2=len(freqs_mhz) - 1,
        l_idx=l_idx,
        m_idx=m_idx,
    )
    catalog.append({
        "l_idx": l_idx,
        "m_idx": m_idx,
        "spectral_index": alpha,
    })

# Print top results
catalog.sort(key=lambda x: x["spectral_index"])
print("\nSteepest-spectrum sources:")
for src in catalog[:5]:
    print(f"  l={src['l_idx']}, m={src['m_idx']}: α = {src['spectral_index']:.2f}")
```

## Step 4: Build a Spectral Energy Distribution

Extract flux density at every frequency channel for a single source:

```python
# Pick the brightest source
src = catalog[0]
l_idx, m_idx = src["l_idx"], src["m_idx"]

flux_values = []
for fi in range(len(freqs_mhz)):
    flux = ds.radport.integrated_flux(
        time_idx=0,
        freq_idx=fi,
        l_min=l_idx - 2,
        l_max=l_idx + 2,
        m_min=m_idx - 2,
        m_max=m_idx + 2,
    )
    flux_values.append(flux)

flux_values = np.array(flux_values)

# Plot the SED
plt.figure(figsize=(10, 6))
plt.loglog(freqs_mhz, flux_values, "o-", color="steelblue")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Flux Density (Jy)")
plt.title(f"SED for source at l={l_idx}, m={m_idx}")
plt.grid(True, alpha=0.3, which="both")
plt.show()
```

## Step 5: Power-Law Fitting

Fit a power law *S = A (ν / ν₀)^α* to the SED:

```python
from scipy.optimize import curve_fit

def power_law(freq, amplitude, alpha, freq0=40.0):
    """Power-law spectral model."""
    return amplitude * (freq / freq0) ** alpha

# Filter out non-positive flux values for log fitting
valid = flux_values > 0
freqs_valid = freqs_mhz[valid]
flux_valid = flux_values[valid]

popt, pcov = curve_fit(power_law, freqs_valid, flux_valid, p0=[1.0, -0.7])
amplitude, alpha_fit = popt
alpha_err = np.sqrt(np.diag(pcov))[1]

print(f"Fitted spectral index: α = {alpha_fit:.2f} ± {alpha_err:.2f}")
print(f"Amplitude at 40 MHz:   A = {amplitude:.3e} Jy")

# Overlay fit on SED
plt.figure(figsize=(10, 6))
plt.loglog(freqs_valid, flux_valid, "o", label="Data")

freq_fit = np.linspace(freqs_valid.min(), freqs_valid.max(), 100)
plt.loglog(freq_fit, power_law(freq_fit, *popt), "-",
           label=f"Fit: α = {alpha_fit:.2f}")

plt.xlabel("Frequency (MHz)")
plt.ylabel("Flux Density (Jy)")
plt.title("Power-Law Fit")
plt.legend()
plt.grid(True, alpha=0.3, which="both")
plt.show()
```

## Step 6: Multi-Frequency Spectral Index Maps

Track how the spectral index varies across different frequency pairs:

```python
n_freq = len(freqs_mhz)
pairs = [(0, n_freq // 3), (n_freq // 3, 2 * n_freq // 3), (2 * n_freq // 3, n_freq - 1)]

fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5))

for ax, (f1, f2) in zip(axes, pairs):
    alpha_map = ds.radport.spectral_index_map(
        freq_idx1=f1,
        freq_idx2=f2,
        time_idx=0,
    )
    ds.radport.plot_spectral_index_map(alpha_map, ax=ax, vmin=-2, vmax=2, cmap="RdBu_r")
    ax.set_title(f"{freqs_mhz[f1]:.0f}–{freqs_mhz[f2]:.0f} MHz")

plt.tight_layout()
plt.show()
```

Differences between panels indicate spectral curvature — the spectral index
is not constant across the band.

## Step 7: Source Classification

Classify detected sources by spectral index:

```python
steep = [s for s in catalog if s["spectral_index"] < -1.0]
normal = [s for s in catalog if -1.0 <= s["spectral_index"] < -0.3]
flat = [s for s in catalog if -0.3 <= s["spectral_index"] <= 0.3]
inverted = [s for s in catalog if s["spectral_index"] > 0.3]

print(f"Ultra-steep (α < -1.0):  {len(steep)}")
print(f"Normal (−1.0 ≤ α < −0.3): {len(normal)}")
print(f"Flat (−0.3 ≤ α ≤ 0.3):   {len(flat)}")
print(f"Inverted (α > 0.3):      {len(inverted)}")
```

Visualize the classification on the sky:

```python
ds.radport.plot(time_idx=0, freq_idx=0)
ax = plt.gca()

colors = {"steep": "blue", "normal": "green", "flat": "orange", "inverted": "red"}
for label, sources, color in [
    ("steep", steep, "blue"),
    ("normal", normal, "green"),
    ("flat", flat, "orange"),
    ("inverted", inverted, "red"),
]:
    if sources:
        ls, ms = zip(*[(s["l_idx"], s["m_idx"]) for s in sources])
        ax.scatter(ls, ms, c=color, s=100, marker="o", edgecolors="white",
                   linewidths=0.5, label=f"{label} ({len(sources)})")

ax.legend(loc="upper right")
plt.title("Source Classification by Spectral Index")
plt.show()
```

## Summary

This tutorial covered:

1. Creating spectral index maps between frequency pairs
2. Detecting sources and measuring their spectral indices
3. Building spectral energy distributions (SEDs)
4. Fitting power-law models to SED data
5. Examining spectral curvature across the band
6. Classifying sources by spectral type

## Next Steps

- [Spectral Analysis Guide](../user-guide/spectral-analysis.md) -- Frequency averaging, spectral variability
- [Source Detection Guide](../user-guide/source-detection.md) -- Advanced detection methods
- [Transient Analysis Tutorial](transient-analysis.md) -- Time-domain analysis
- [API Reference](../api/radport-accessor.md) -- Full method documentation
