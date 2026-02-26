# radport Accessor

The `radport` accessor extends `xarray.Dataset` with domain-specific methods for
OVRO-LWA radio astronomy data analysis and visualization.

## Usage

The accessor becomes available on any `xarray.Dataset` after importing
`ovro_lwa_portal`:

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")
ds.radport.plot()  # accessor is now available
```

The dataset must contain the required dimensions (`time`, `frequency`,
`polarization`, `l`, `m`) and the `SKY` data variable. An optional `BEAM`
variable enables beam-related functionality.

## Method Categories

| Category              | Methods                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| **Selection**         | `nearest_freq_idx`, `nearest_time_idx`, `nearest_lm_idx`                             |
| **Basic Plotting**    | `plot`, `cutout`, `plot_cutout`                                                      |
| **Dynamic Spectrum**  | `dynamic_spectrum`, `plot_dynamic_spectrum`                                          |
| **Difference Maps**   | `diff`, `plot_diff`                                                                  |
| **Data Quality**      | `find_valid_frame`, `finite_fraction`                                                |
| **Grid Plots**        | `plot_grid`, `plot_frequency_grid`, `plot_time_grid`                                 |
| **Light Curves**      | `light_curve`, `plot_light_curve`                                                    |
| **Spectra**           | `spectrum`, `plot_spectrum`                                                          |
| **Averaging**         | `time_average`, `frequency_average`, `plot_time_average`, `plot_frequency_average`   |
| **WCS Coordinates**   | `has_wcs`, `pixel_to_coords`, `coords_to_pixel`, `plot_wcs`                          |
| **Animation**         | `animate_time`, `animate_frequency`, `export_frames`                                 |
| **Source Detection**  | `rms_map`, `snr_map`, `find_peaks`, `peak_flux_map`, `plot_snr_map`                  |
| **Spectral Analysis** | `spectral_index`, `spectral_index_map`, `integrated_flux`, `plot_spectral_index_map` |

## Full API Reference

::: ovro_lwa_portal.accessor.RadportAccessor
    options:
      show_root_heading: true
      show_root_full_path: false
      members_order: source
