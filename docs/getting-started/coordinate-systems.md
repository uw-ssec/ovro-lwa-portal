# Coordinate Systems

OVRO-LWA data uses multiple coordinate systems. Understanding them is essential for analysis.

## Pixel Coordinates (l, m)

The primary coordinate system uses direction cosines:

- **l**: East-West direction cosine (dimensionless)
- **m**: North-South direction cosine (dimensionless)
- Range: typically -1 to +1
- Center: (l=0, m=0) is the phase center

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

# Access pixel data
intensity = ds['SKY'].sel(l=0.1, m=0.2, method='nearest')
```

## World Coordinate System (WCS)

If WCS information is present, you can work with celestial coordinates:

### Check for WCS

```python
# Check if dataset has WCS
if ds.radport.has_wcs():
    print("WCS available")
else:
    print("No WCS information")
```

### Pixel to Sky Coordinates

Convert pixel coordinates to RA/Dec:

```python
# Convert (l, m) to (RA, Dec)
ra, dec = ds.radport.pixel_to_coords(l=0.1, m=0.2)
print(f"RA: {ra:.3f}°, Dec: {dec:.3f}°")
```

### Sky to Pixel Coordinates

Convert RA/Dec to pixel coordinates:

```python
# Convert (RA, Dec) to (l, m)
l, m = ds.radport.coords_to_pixel(ra=180.0, dec=20.0)
print(f"l: {l:.3f}, m: {m:.3f}")
```

### Plotting with WCS

```python
# Plot with WCS overlay
ds.radport.plot_wcs(
    time_idx=0,
    freq_idx=0,
    projection='rectangular'  # or 'aitoff', 'mollweide'
)
```

## Indexing Methods

The accessor provides helper methods to find nearest indices:

### Nearest Frequency

```python
# Find index for frequency closest to 40 MHz
freq_idx = ds.radport.nearest_freq_idx(freq_mhz=40.0)

# Use it
ds.radport.plot(freq_idx=freq_idx)
```

### Nearest Time

```python
# Find index for time closest to MJD
time_idx = ds.radport.nearest_time_idx(mjd=59000.5)

# Use it
ds.radport.plot(time_idx=time_idx)
```

### Nearest Pixel

```python
# Find pixel indices closest to (l, m)
l_idx, m_idx = ds.radport.nearest_lm_idx(l=0.1, m=0.2)

# Extract light curve at that location
lc = ds.radport.light_curve(l_idx=l_idx, m_idx=m_idx)
```

## Time Coordinates

Time is stored as Modified Julian Date (MJD):

```python
# Access time coordinates
times = ds.coords['time']

# Convert to datetime if needed
from astropy.time import Time
times_dt = Time(times.values, format='mjd').datetime
```

## Frequency Coordinates

Frequencies are in Hz:

```python
# Access frequency coordinates
freqs = ds.coords['frequency']

# Convert to MHz for display
freqs_mhz = freqs.values / 1e6
```

## Beam Coordinates

Some datasets include beam information:

```python
# Check for beam
if ds.radport.has_beam():
    beam = ds['BEAM']
    print(f"Beam shape: {beam.shape}")
```

## Example: Multi-Coordinate Analysis

```python
import ovro_lwa_portal as ovro

ds = ovro.open_dataset("path/to/data.zarr")

# Find indices
freq_idx = ds.radport.nearest_freq_idx(freq_mhz=40.0)
time_idx = ds.radport.nearest_time_idx(mjd=59000.5)

# Convert sky position to pixels
l, m = ds.radport.coords_to_pixel(ra=180.0, dec=20.0)
l_idx, m_idx = ds.radport.nearest_lm_idx(l=l, m=m)

# Extract data at that point
value = ds['SKY'].isel(
    time=time_idx,
    frequency=freq_idx,
    l=l_idx,
    m=m_idx
).values

print(f"Intensity at RA=180°, Dec=20°, 40 MHz: {value:.2f}")
```

## Next Steps

- Explore [visualization methods](../user-guide/visualization.md)
- Learn about [WCS coordinates](../user-guide/wcs-coordinates.md)
- Try the [transient analysis tutorial](../tutorials/transient-analysis.md)
