"""Interactive explorer classes for OVRO-LWA data.

Each explorer uses the optimal data access strategy for its pattern:

- **ImageExplorer**: LRU-cached single slices (one S3 read per frame change)
- **DynamicSpectrumExplorer**: accessor's batched dask.compute for the
  waterfall, accessor's single-frame methods for linked views
- **CutoutExplorer**: LRU-cached single slices with spatial subsetting

Call ``.panel()`` to get a renderable Panel layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import holoviews as hv
import panel as pn
import param

from ovro_lwa_portal.viz._data import (
    PreloadedCube,
    cutout_image_element,
    sky_image_element,
)
from ovro_lwa_portal.viz.components import (
    COLORMAPS,
    style_curve,
    style_sky_image,
    style_spectrum_image,
)

if TYPE_CHECKING:
    import xarray as xr

hv.extension("bokeh")


class ImageExplorer(param.Parameterized):
    """Interactive image explorer with time/frequency/polarization selection.

    Uses an LRU cache — first view of each (time, freq) slice takes one
    S3 read (~3s), subsequent views of the same slice are instant.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.
    """

    time_idx = param.Integer(default=0, bounds=(0, 1), doc="Time step index")
    freq_idx = param.Integer(default=0, bounds=(0, 1), doc="Frequency channel index")
    pol = param.Integer(default=0, bounds=(0, 0), doc="Polarization index")
    var = param.Selector(
        default="SKY", objects=["SKY"], doc="Data variable to display"
    )
    cmap = param.Selector(
        default="inferno", objects=COLORMAPS, doc="Colormap"
    )
    robust = param.Boolean(default=True, doc="Use robust (percentile) color scaling")

    def __init__(self, ds: xr.Dataset, **params: Any) -> None:
        super().__init__(**params)
        self._ds = ds

        n_times = ds.sizes["time"]
        n_freqs = ds.sizes["frequency"]
        n_pols = ds.sizes["polarization"]

        self.param.time_idx.bounds = (0, max(0, n_times - 1))
        self.param.freq_idx.bounds = (0, max(0, n_freqs - 1))
        self.param.pol.bounds = (0, max(0, n_pols - 1))

        available_vars = [v for v in ["SKY", "BEAM"] if v in ds.data_vars]
        self.param.var.objects = available_vars
        if available_vars:
            self.var = available_vars[0]

        # LRU-cached cube — no upfront load, fetches one slice at a time
        self._cube = PreloadedCube(ds, var=self.var, pol=self.pol)

        self._time_labels = {
            i: f"{float(ds.coords['time'].values[i]):.4f}"
            for i in range(n_times)
        }
        self._freq_labels = {
            i: f"{float(ds.coords['frequency'].values[i]) / 1e6:.1f} MHz"
            for i in range(n_freqs)
        }

    @param.depends("time_idx", "freq_idx", "cmap", "robust")
    def _image_view(self) -> hv.Image:
        img = sky_image_element(
            self._cube,
            time_idx=self.time_idx,
            freq_idx=self.freq_idx,
            robust=self.robust,
        )
        return style_sky_image(img, cmap=self.cmap)

    @param.depends("time_idx")
    def _time_label(self) -> str:
        return f"**Time:** {self._time_labels.get(self.time_idx, '?')} MJD"

    @param.depends("freq_idx")
    def _freq_label(self) -> str:
        return f"**Frequency:** {self._freq_labels.get(self.freq_idx, '?')}"

    def panel(self) -> pn.viewable.Viewable:
        """Return the complete Panel layout."""
        time_label = pn.pane.Markdown(self._time_label, width=250)
        freq_label = pn.pane.Markdown(self._freq_label, width=250)

        controls = pn.Column(
            "## Image Explorer",
            pn.widgets.IntSlider.from_param(self.param.time_idx, name="Time Step"),
            time_label,
            pn.widgets.IntSlider.from_param(self.param.freq_idx, name="Frequency Channel"),
            freq_label,
            pn.widgets.Select.from_param(self.param.cmap, name="Colormap"),
            pn.widgets.Checkbox.from_param(self.param.robust, name="Robust Scaling"),
            width=280,
        )

        image_pane = pn.pane.HoloViews(
            hv.DynamicMap(self._image_view), sizing_mode="stretch_both"
        )

        return pn.Row(controls, image_pane, sizing_mode="stretch_width")


class DynamicSpectrumExplorer(param.Parameterized):
    """Interactive dynamic spectrum explorer with linked views.

    Uses the accessor's ``dynamic_spectrum()`` method which batches all
    pixel extractions via ``dask.compute()`` — much faster than 100
    sequential reads. Linked spectrum/light curve use the accessor's
    single-frame methods (one chunk read each).

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.
    """

    l_val = param.Number(default=0.0, bounds=(-1.2, 1.2), step=0.01, doc="l direction cosine")
    m_val = param.Number(default=0.0, bounds=(-1.2, 1.2), step=0.01, doc="m direction cosine")
    cmap = param.Selector(default="inferno", objects=COLORMAPS, doc="Colormap")
    robust = param.Boolean(default=True, doc="Use robust color scaling")

    def __init__(
        self,
        ds: xr.Dataset,
        *,
        l: float | None = None,
        m: float | None = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._ds = ds

        # Set l/m bounds from coordinate range
        l_vals = ds.coords["l"].values
        m_vals = ds.coords["m"].values
        self.param.l_val.bounds = (
            float(min(l_vals[0], l_vals[-1])),
            float(max(l_vals[0], l_vals[-1])),
        )
        self.param.m_val.bounds = (
            float(min(m_vals[0], m_vals[-1])),
            float(max(m_vals[0], m_vals[-1])),
        )

        if l is not None:
            self.l_val = l
        if m is not None:
            self.m_val = m

        self._tap = hv.streams.Tap(x=None, y=None)

        # Cache for dynamic spectra keyed by (l_val, m_val)
        self._dynspec_cache: dict[tuple[float, float], Any] = {}

    def _get_dynspec(self) -> Any:
        """Get dynamic spectrum DataArray, using cache if available."""
        key = (round(self.l_val, 4), round(self.m_val, 4))
        if key not in self._dynspec_cache:
            self._dynspec_cache[key] = self._ds.radport.dynamic_spectrum(
                l=self.l_val, m=self.m_val
            )
        return self._dynspec_cache[key]

    @param.depends("l_val", "m_val", "cmap", "robust")
    def _dynspec_view(self) -> hv.Image:
        from ovro_lwa_portal.viz._data import _ensure_extension

        _ensure_extension()

        dynspec = self._get_dynspec()
        time_mjd = dynspec.coords["time"].values
        freq_mhz = dynspec.coords["frequency"].values / 1e6
        data = dynspec.values

        pixel_l = dynspec.attrs.get("pixel_l", "?")
        pixel_m = dynspec.attrs.get("pixel_m", "?")
        if isinstance(pixel_l, (int, float)):
            title = f"Dynamic Spectrum at l={pixel_l:+.4f}, m={pixel_m:+.4f}"
        else:
            title = f"Dynamic Spectrum at l={self.l_val:+.4f}, m={self.m_val:+.4f}"

        # Add half-step padding so each pixel has visible extent.
        # Without this, a narrow time range (e.g., 0.001 MJD span) can
        # produce bounds where left ≈ right, rendering an empty image.
        if len(time_mjd) > 1:
            dt = float(time_mjd[1] - time_mjd[0]) / 2
        else:
            dt = 0.0001
        if len(freq_mhz) > 1:
            df = float(freq_mhz[1] - freq_mhz[0]) / 2
        else:
            df = 1.0

        bounds = (
            float(time_mjd[0]) - dt, float(freq_mhz[0]) - df,
            float(time_mjd[-1]) + dt, float(freq_mhz[-1]) + df,
        )

        img = hv.Image(
            data, kdims=["Time (MJD)", "Frequency (MHz)"], bounds=bounds,
        ).opts(
            xlabel="Time (MJD)", ylabel="Frequency (MHz)",
            title=title, colorbar=True, clabel="Jy/beam",
        )

        if self.robust:
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                img = img.opts(
                    clim=(float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                )

        img = style_spectrum_image(img, cmap=self.cmap)
        self._tap.source = img
        return img

    def _linked_spectrum(self, x: float | None, y: float | None) -> hv.Curve:
        """Spectrum at the clicked time step — one chunk read."""
        from ovro_lwa_portal.viz._data import _ensure_extension

        _ensure_extension()

        if x is None:
            return hv.Curve([], kdims=["Frequency (MHz)"], vdims=["Intensity (Jy/beam)"]).opts(
                title="Click on dynamic spectrum to show spectrum"
            )

        time_vals = self._ds.coords["time"].values
        time_idx = int(np.abs(time_vals - x).argmin())

        spec = self._ds.radport.spectrum(
            l=self.l_val, m=self.m_val, time_idx=time_idx
        )
        freq_mhz = spec.coords["frequency"].values / 1e6
        time_val = float(time_vals[time_idx])

        curve = hv.Curve(
            (freq_mhz, spec.values),
            kdims=["Frequency (MHz)"], vdims=["Intensity (Jy/beam)"],
        ).opts(title=f"Spectrum at t={time_val:.4f} MJD")
        return style_curve(curve)

    def _linked_light_curve(self, x: float | None, y: float | None) -> hv.Curve:
        """Light curve at the clicked frequency — one chunk read per time step."""
        from ovro_lwa_portal.viz._data import _ensure_extension

        _ensure_extension()

        if y is None:
            return hv.Curve([], kdims=["Time (MJD)"], vdims=["Intensity (Jy/beam)"]).opts(
                title="Click on dynamic spectrum to show light curve"
            )

        freq_hz = self._ds.coords["frequency"].values
        freq_mhz = freq_hz / 1e6
        freq_idx = int(np.abs(freq_mhz - y).argmin())

        lc = self._ds.radport.light_curve(
            l=self.l_val, m=self.m_val, freq_idx=freq_idx
        )

        curve = hv.Curve(
            (lc.coords["time"].values, lc.values),
            kdims=["Time (MJD)"], vdims=["Intensity (Jy/beam)"],
        ).opts(title=f"Light Curve at f={float(freq_mhz[freq_idx]):.1f} MHz")
        return style_curve(curve)

    def panel(self) -> pn.viewable.Viewable:
        """Return the complete Panel layout with linked views."""
        controls = pn.Column(
            "## Dynamic Spectrum Explorer",
            pn.widgets.FloatSlider.from_param(self.param.l_val, name="l", step=0.01),
            pn.widgets.FloatSlider.from_param(self.param.m_val, name="m", step=0.01),
            pn.widgets.Select.from_param(self.param.cmap, name="Colormap"),
            pn.widgets.Checkbox.from_param(self.param.robust, name="Robust Scaling"),
            "---",
            "*Initial load fetches pixel data across all time/freq slices.*",
            width=280,
        )

        dynspec_pane = pn.pane.HoloViews(
            hv.DynamicMap(self._dynspec_view), sizing_mode="stretch_both"
        )
        spectrum_pane = pn.pane.HoloViews(
            hv.DynamicMap(self._linked_spectrum, streams=[self._tap]),
        )
        lightcurve_pane = pn.pane.HoloViews(
            hv.DynamicMap(self._linked_light_curve, streams=[self._tap]),
        )

        return pn.Row(
            controls,
            pn.Column(dynspec_pane, sizing_mode="stretch_both"),
            pn.Column(spectrum_pane, lightcurve_pane),
            sizing_mode="stretch_width",
        )


class CutoutExplorer(param.Parameterized):
    """Interactive cutout explorer with linked light curve and spectrum.

    Uses an LRU-cached cube for spatial subsetting — fast for single frames.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset to explore.
    """

    l_center = param.Number(default=0.0, step=0.01, doc="l center")
    m_center = param.Number(default=0.0, step=0.01, doc="m center")
    dl = param.Number(default=0.1, bounds=(0.01, 1.0), step=0.01, doc="l half-extent")
    dm = param.Number(default=0.1, bounds=(0.01, 1.0), step=0.01, doc="m half-extent")
    time_idx = param.Integer(default=0, bounds=(0, 1), doc="Time step")
    freq_idx = param.Integer(default=0, bounds=(0, 1), doc="Frequency channel")
    cmap = param.Selector(default="inferno", objects=COLORMAPS, doc="Colormap")
    robust = param.Boolean(default=True, doc="Robust scaling")

    def __init__(self, ds: xr.Dataset, **params: Any) -> None:
        super().__init__(**params)
        self._ds = ds

        n_times = ds.sizes["time"]
        n_freqs = ds.sizes["frequency"]
        self.param.time_idx.bounds = (0, max(0, n_times - 1))
        self.param.freq_idx.bounds = (0, max(0, n_freqs - 1))

        var = "SKY" if "SKY" in ds.data_vars else list(ds.data_vars)[0]
        self._cube = PreloadedCube(ds, var=var, pol=0)

        self._tap = hv.streams.Tap(x=None, y=None)

    @param.depends("l_center", "m_center", "dl", "dm", "time_idx", "freq_idx", "cmap", "robust")
    def _cutout_view(self) -> hv.Image:
        img = cutout_image_element(
            self._cube,
            l_center=self.l_center, m_center=self.m_center,
            dl=self.dl, dm=self.dm,
            time_idx=self.time_idx, freq_idx=self.freq_idx,
            robust=self.robust,
        )
        img = style_sky_image(img, cmap=self.cmap)
        self._tap.source = img
        return img

    def _linked_spectrum(self, x: float | None, y: float | None) -> hv.Curve:
        from ovro_lwa_portal.viz._data import _ensure_extension, spectrum_element

        _ensure_extension()

        if x is None or y is None:
            return hv.Curve([], kdims=["Frequency (MHz)"], vdims=["Intensity (Jy/beam)"]).opts(
                title="Click on cutout to show spectrum"
            )
        curve = spectrum_element(self._cube, l=x, m=y, time_idx=self.time_idx)
        curve = curve.opts(title=f"Spectrum at l={x:.4f}, m={y:.4f}")
        return style_curve(curve)

    def _linked_light_curve(self, x: float | None, y: float | None) -> hv.Curve:
        from ovro_lwa_portal.viz._data import _ensure_extension, light_curve_element

        _ensure_extension()

        if x is None or y is None:
            return hv.Curve([], kdims=["Time (MJD)"], vdims=["Intensity (Jy/beam)"]).opts(
                title="Click on cutout to show light curve"
            )
        curve = light_curve_element(self._cube, l=x, m=y, freq_idx=self.freq_idx)
        curve = curve.opts(title=f"Light Curve at l={x:.4f}, m={y:.4f}")
        return style_curve(curve)

    def panel(self) -> pn.viewable.Viewable:
        """Return the complete Panel layout with linked views."""
        controls = pn.Column(
            "## Cutout Explorer",
            pn.widgets.FloatSlider.from_param(self.param.l_center, name="l center", step=0.01),
            pn.widgets.FloatSlider.from_param(self.param.m_center, name="m center", step=0.01),
            pn.widgets.FloatSlider.from_param(self.param.dl, name="dl (half-extent)", step=0.01),
            pn.widgets.FloatSlider.from_param(self.param.dm, name="dm (half-extent)", step=0.01),
            pn.widgets.IntSlider.from_param(self.param.time_idx, name="Time Step"),
            pn.widgets.IntSlider.from_param(self.param.freq_idx, name="Frequency"),
            pn.widgets.Select.from_param(self.param.cmap, name="Colormap"),
            pn.widgets.Checkbox.from_param(self.param.robust, name="Robust Scaling"),
            width=280,
        )

        cutout_pane = pn.pane.HoloViews(
            hv.DynamicMap(self._cutout_view), sizing_mode="stretch_both"
        )
        spectrum_pane = pn.pane.HoloViews(
            hv.DynamicMap(self._linked_spectrum, streams=[self._tap]),
        )
        lightcurve_pane = pn.pane.HoloViews(
            hv.DynamicMap(self._linked_light_curve, streams=[self._tap]),
        )

        return pn.Row(
            controls,
            pn.Column(cutout_pane, sizing_mode="stretch_both"),
            pn.Column(spectrum_pane, lightcurve_pane),
            sizing_mode="stretch_width",
        )
