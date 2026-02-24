# WCS Coordinates

WCS (World Coordinate System) information is preserved during FITS-to-Zarr conversion,
enabling celestial coordinate (RA/Dec) overlays on sky images. The `radport` accessor
provides methods to check WCS availability, convert between pixel and sky coordinates,
and create publication-quality WCS-projected plots.

## Checking WCS Availability

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

if ds.radport.has_wcs:
    print("WCS coordinates available")
```

The WCS header is searched in three locations (in order):

1. Variable attributes (`fits_wcs_header` on the `SKY` variable)
2. Dataset attributes (`fits_wcs_header` on the dataset)
3. A `wcs_header_str` data variable in the dataset

!!! note
    WCS functionality requires [astropy](https://www.astropy.org/) to be installed.

## Pixel-to-Sky Conversion

Convert pixel indices along the `l` and `m` dimensions to celestial coordinates:

```python
ra, dec = ds.radport.pixel_to_coords(l_idx=100, m_idx=100)
print(f"RA: {ra:.4f} deg, Dec: {dec:.4f} deg")
```

- RA is returned in the range [0, 360) degrees
- Dec is returned in degrees
- Raises `ValueError` if indices are out of bounds

## Sky-to-Pixel Conversion

Convert RA/Dec coordinates to pixel indices:

```python
l_idx, m_idx = ds.radport.coords_to_pixel(ra=180.0, dec=45.0)
print(f"Pixel: l={l_idx}, m={m_idx}")
```

- Returns rounded integer indices
- Raises `ValueError` if the coordinates fall outside the image bounds
- Uses the FK5 celestial frame

## Plotting with WCS Projection

The `plot_wcs` method creates plots with RA/Dec coordinate axes and a grid overlay:

```python
# Basic WCS plot
fig = ds.radport.plot_wcs(freq_mhz=50.0)
```

### Customizing the Plot

```python
# Publication-quality dark-background plot
fig = ds.radport.plot_wcs(
    freq_mhz=50.0,
    mask_radius=1800,
    cmap="inferno",
    grid_color="white",
    grid_alpha=0.6,
    grid_linestyle=":",
    label_color="white",
    facecolor="black",
    figsize=(10, 10),
)
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `var` | `"SKY"` | Data variable to plot (`"SKY"` or `"BEAM"`) |
| `time_idx` | `0` | Time index |
| `freq_idx` | `0` | Frequency index (ignored if `freq_mhz` is set) |
| `freq_mhz` | `None` | Select frequency by value in MHz |
| `cmap` | `"inferno"` | Matplotlib colormap |
| `robust` | `True` | Use 2nd/98th percentile for color scaling |
| `mask_radius` | `None` | Circular mask radius in pixels |
| `grid_color` | `"white"` | Color of the RA/Dec grid lines |
| `grid_alpha` | `0.6` | Transparency of grid lines |
| `facecolor` | `"black"` | Plot background color |
| `add_colorbar` | `True` | Whether to include a colorbar |

The plot automatically handles RA axis inversion (RA increases to the left on the sky)
and uses robust percentile-based scaling by default.

## Example Workflows

### Locate a Source by RA/Dec and Extract a Light Curve

```python
# Find the Crab Nebula (RA=83.633, Dec=22.014)
l_idx, m_idx = ds.radport.coords_to_pixel(ra=83.633, dec=22.014)

# Get the l, m coordinate values at those indices
l_val = float(ds.coords["l"].values[l_idx])
m_val = float(ds.coords["m"].values[m_idx])

# Extract and plot the light curve
lc = ds.radport.light_curve(l=l_val, m=m_val)
fig = ds.radport.plot_light_curve(lc)
```

### Create a Cutout Around a Known Position

```python
# Convert RA/Dec to pixel coordinates
l_idx, m_idx = ds.radport.coords_to_pixel(ra=83.633, dec=22.014)
l_val = float(ds.coords["l"].values[l_idx])
m_val = float(ds.coords["m"].values[m_idx])

# Extract a cutout centered on the source
cutout = ds.radport.cutout(l_center=l_val, m_center=m_val, dl=0.1, dm=0.1)
fig = ds.radport.plot_cutout(cutout)
```

### WCS Plot with Detected Sources

```python
import matplotlib.pyplot as plt

# Create WCS plot
fig = ds.radport.plot_wcs(freq_mhz=50.0, mask_radius=1800)
ax = fig.axes[0]

# Overlay detected sources
peaks = ds.radport.find_peaks(time_idx=0, freq_idx=0, threshold=5.0)
for l_idx, m_idx in peaks:
    ra, dec = ds.radport.pixel_to_coords(int(l_idx), int(m_idx))
    ax.plot(ra, dec, "r+", markersize=10, transform=ax.get_transform("fk5"))

plt.show()
```

## How WCS Is Preserved

The WCS pipeline works as follows:

1. **Original FITS files** contain standard WCS headers (CRVAL, CRPIX, CDELT, etc.)
2. **Header fixing** (`fix_fits_headers`) corrects any non-standard headers
3. **Zarr conversion** stores the complete WCS header as a string attribute (`fits_wcs_header`) on the data variable or dataset
4. **At load time**, the `radport` accessor parses the stored header string back into an astropy `WCS` object on demand

This means WCS information survives the FITS-to-Zarr conversion pipeline and is available
whenever the original FITS files contained valid WCS headers.

## See Also

- [Coordinate Systems](../getting-started/coordinate-systems.md) -- Introduction to pixel (l, m) and celestial coordinate systems
- [Visualization](visualization.md) -- General plotting methods
- [API Reference](../api/radport-accessor.md) -- Full method documentation for `pixel_to_coords`, `coords_to_pixel`, and `plot_wcs`
