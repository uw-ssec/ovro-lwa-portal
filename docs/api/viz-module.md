# Visualization Module

Interactive Panel/HoloViews visualization framework for OVRO-LWA datasets.
All dependencies are optional — install with
`pip install 'ovro_lwa_portal[visualization]'`.

For usage examples, see the
[Interactive Visualization](../user-guide/interactive-visualization.md) guide.

## Module Entry Points

::: ovro_lwa_portal.viz
    options:
      show_root_heading: false
      members:
        - ImageExplorer
        - DynamicSpectrumExplorer
        - CutoutExplorer
        - SkyViewer
        - create_exploration_dashboard

## Explorer Classes

### ImageExplorer

::: ovro_lwa_portal.viz.explorers.ImageExplorer
    options:
      show_root_heading: true
      members:
        - panel

### DynamicSpectrumExplorer

::: ovro_lwa_portal.viz.explorers.DynamicSpectrumExplorer
    options:
      show_root_heading: true
      members:
        - panel

### CutoutExplorer

::: ovro_lwa_portal.viz.explorers.CutoutExplorer
    options:
      show_root_heading: true
      members:
        - panel

### SkyViewer

::: ovro_lwa_portal.viz.sky_viewer.SkyViewer
    options:
      show_root_heading: true
      members:
        - panel

## Data Bridge

Internal utilities for converting accessor outputs to HoloViews elements.

### PreloadedCube

::: ovro_lwa_portal.viz._data.PreloadedCube
    options:
      show_root_heading: true
      members:
        - image
        - dynamic_spectrum
        - light_curve
        - spectrum
        - nearest_lm_idx
        - bounds

### Element Factories

::: ovro_lwa_portal.viz._data.sky_image_element
    options:
      show_root_heading: true

::: ovro_lwa_portal.viz._data.cutout_image_element
    options:
      show_root_heading: true

::: ovro_lwa_portal.viz._data.dynamic_spectrum_element
    options:
      show_root_heading: true

::: ovro_lwa_portal.viz._data.light_curve_element
    options:
      show_root_heading: true

::: ovro_lwa_portal.viz._data.spectrum_element
    options:
      show_root_heading: true

## Components

::: ovro_lwa_portal.viz.components
    options:
      show_root_heading: false
      members:
        - COLORMAPS
        - style_sky_image
        - style_spectrum_image
        - style_curve

## Dashboards

::: ovro_lwa_portal.viz.dashboards.create_exploration_dashboard
    options:
      show_root_heading: true

## Accessor Integration

The `radport` xarray accessor exposes the visualization framework through
convenience methods:

| Accessor Method | Explorer |
|----------------|----------|
| `ds.radport.explore()` | Full tabbed dashboard |
| `ds.radport.explore_image()` | `ImageExplorer` |
| `ds.radport.explore_dynamic_spectrum()` | `DynamicSpectrumExplorer` |
| `ds.radport.explore_sky()` | `SkyViewer` |
