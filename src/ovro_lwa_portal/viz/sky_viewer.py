"""Interactive sky viewer using ipyaladin (Aladin Lite) and Panel.

Provides a real-time, pannable, zoomable celestial sky view with
OVRO-LWA data overlaid on astronomical survey backgrounds (DSS, WISE,
Planck, etc.). Replaces static matplotlib WCS plots with continuous
visualization that can be explored in real time.

The viewer constructs proper FITS HDUs from the dataset's numpy arrays
and WCS headers, then uses Aladin Lite's ``add_fits`` to overlay them
on the sky with correct celestial projection.

Examples
--------
>>> from ovro_lwa_portal.viz.sky_viewer import SkyViewer
>>> viewer = SkyViewer(ds)
>>> viewer.panel()  # Interactive sky exploration in Jupyter or Panel serve
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import panel as pn
import param

from ovro_lwa_portal.viz.components import COLORMAPS

if TYPE_CHECKING:
    import xarray as xr

# Background survey presets useful for radio astronomy
SURVEY_PRESETS: dict[str, str] = {
    "DSS Color": "CDS/P/DSS2/color",
    "2MASS Color": "CDS/P/2MASS/color",
    "AllWISE Color": "CDS/P/allWISE/color",
    "Planck HFI Color": "CDS/P/PLANCK/R2/HFI/color",
    "SDSS9 Color": "CDS/P/SDSS9/color",
    "Mellinger Color": "CDS/P/Mellinger/color",
    "Fermi Color": "CDS/P/Fermi/color",
    "RASS Soft": "CDS/P/RASS",
    "Haslam 408 MHz": "CDS/P/HI4PI/NHI",
}


def _build_fits_hdu(
    ds: xr.Dataset,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol: int = 0,
    var: Literal["SKY", "BEAM"] = "SKY",
    robust: bool = True,
) -> Any:
    """Build an astropy FITS HDUList from a dataset slice.

    Constructs a PrimaryHDU with the 2D image data and proper WCS
    header so that Aladin Lite can project it onto the sky.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset with WCS header.
    time_idx, freq_idx, pol : int
        Slice indices.
    var : str
        Data variable name.
    robust : bool
        Apply 2nd/98th percentile clipping for better display.

    Returns
    -------
    astropy.io.fits.HDUList
        FITS HDU list ready for ``aladin.add_fits()``.
    """
    from astropy.io.fits import Header, HDUList, PrimaryHDU

    # Extract the 2D image slice
    da = ds[var].isel(time=time_idx, frequency=freq_idx, polarization=pol)

    # Downsample large images for Aladin overlay performance
    max_size = 512
    n_l, n_m = da.sizes.get("l", 0), da.sizes.get("m", 0)
    factor_l = max(1, n_l // max_size)
    factor_m = max(1, n_m // max_size)
    if factor_l > 1 or factor_m > 1:
        trim_l = (n_l // factor_l) * factor_l
        trim_m = (n_m // factor_m) * factor_m
        da = da.isel(l=slice(0, trim_l), m=slice(0, trim_m))
        da = da.coarsen(l=factor_l, m=factor_m, boundary="exact").mean()

    data = da.values.copy().astype(np.float32)

    # Apply robust clipping for display
    if robust:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
            data = np.clip(data, vmin, vmax)

    # Replace NaN with 0 for FITS compatibility (Aladin handles blanks)
    data = np.nan_to_num(data, nan=0.0)

    # Get the WCS header from the dataset
    hdr_str = None
    if var in ds.data_vars:
        hdr_str = ds[var].attrs.get("fits_wcs_header")
    if not hdr_str:
        hdr_str = ds.attrs.get("fits_wcs_header")

    if not hdr_str:
        msg = (
            "No WCS header found in dataset. The sky viewer requires "
            "celestial WCS metadata (fits_wcs_header attribute)."
        )
        raise ValueError(msg)

    header = Header.fromstring(hdr_str, sep="\n")

    # Ensure NAXIS matches our data shape
    # The data is (l, m) where l=NAXIS1, m=NAXIS2
    # FITS stores row-major so we transpose: FITS expects (NAXIS2, NAXIS1)
    fits_data = data.T

    header["NAXIS"] = 2
    header["NAXIS1"] = fits_data.shape[1]
    header["NAXIS2"] = fits_data.shape[0]
    header["BITPIX"] = -32  # float32

    # Adjust WCS for downsampled pixel scale
    if factor_l > 1 or factor_m > 1:
        if "CDELT1" in header:
            header["CDELT1"] = header["CDELT1"] * factor_l
        if "CDELT2" in header:
            header["CDELT2"] = header["CDELT2"] * factor_m
        if "CRPIX1" in header:
            header["CRPIX1"] = (header["CRPIX1"] - 0.5) / factor_l + 0.5
        if "CRPIX2" in header:
            header["CRPIX2"] = (header["CRPIX2"] - 0.5) / factor_m + 0.5

    hdu = PrimaryHDU(data=fits_data, header=header)
    return HDUList([hdu])


def _hdulist_to_bytes(hdul: Any) -> bytes:
    """Serialize an HDUList to bytes for Aladin ingestion."""
    buf = io.BytesIO()
    hdul.writeto(buf, overwrite=True)
    return buf.getvalue()


class SkyViewer(param.Parameterized):
    """Interactive sky viewer with OVRO-LWA data overlaid on survey backgrounds.

    Uses ipyaladin (Aladin Lite) to provide a real-time, pannable,
    zoomable celestial sky view. The OVRO-LWA image is projected onto
    the sky using the dataset's WCS header.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset with WCS header.

    Examples
    --------
    >>> viewer = SkyViewer(ds)
    >>> viewer.panel()  # Launch in notebook or panel serve
    """

    time_idx = param.Integer(default=0, bounds=(0, 1), doc="Time step")
    freq_idx = param.Integer(default=0, bounds=(0, 1), doc="Frequency channel")
    pol = param.Integer(default=0, bounds=(0, 0), doc="Polarization")
    var = param.Selector(default="SKY", objects=["SKY"], doc="Variable")

    survey = param.Selector(
        default="DSS Color",
        objects=list(SURVEY_PRESETS.keys()),
        doc="Background survey",
    )
    overlay_opacity = param.Number(
        default=0.7, bounds=(0.0, 1.0), step=0.05,
        doc="OVRO-LWA overlay opacity",
    )
    colormap = param.Selector(
        default="inferno", objects=COLORMAPS, doc="Overlay colormap"
    )
    stretch = param.Selector(
        default="linear",
        objects=["linear", "log", "sqrt", "pow2"],
        doc="Color stretch function",
    )
    robust = param.Boolean(default=True, doc="Robust percentile clipping")
    fov = param.Number(
        default=180.0, bounds=(0.1, 180.0), step=1.0,
        doc="Field of view (degrees)",
    )

    def __init__(self, ds: xr.Dataset, **params: Any) -> None:
        super().__init__(**params)
        self._ds = ds

        # Set bounds from dataset dimensions
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

        # Precompute display labels
        self._time_labels = {
            i: f"{float(ds.coords['time'].values[i]):.4f}"
            for i in range(n_times)
        }
        self._freq_labels = {
            i: f"{float(ds.coords['frequency'].values[i]) / 1e6:.1f} MHz"
            for i in range(n_freqs)
        }

        # Get the phase center from WCS for initial Aladin target
        self._phase_center_ra = 0.0
        self._phase_center_dec = 0.0
        if ds.radport.has_wcs:
            try:
                wcs = ds.radport._get_wcs()
                self._phase_center_ra = float(wcs.wcs.crval[0])
                self._phase_center_dec = float(wcs.wcs.crval[1])
            except Exception:  # noqa: BLE001
                pass

        # Create the Aladin widget
        from astropy.coordinates import SkyCoord
        from ipyaladin import Aladin

        self._aladin = Aladin(
            target=SkyCoord(
                ra=self._phase_center_ra,
                dec=self._phase_center_dec,
                unit="deg",
                frame="fk5",
            ),
            fov=self.fov,
            survey=SURVEY_PRESETS[self.survey],
            projection="SIN",
            show_coo_grid=True,
            show_coo_grid_control=True,
            show_settings_control=True,
            height=600,
        )

        # Track the current overlay name for removal
        self._current_overlay_name: str | None = None

    def _update_overlay(self) -> None:
        """Update the OVRO-LWA image overlay on Aladin."""
        try:
            hdul = _build_fits_hdu(
                self._ds,
                time_idx=self.time_idx,
                freq_idx=self.freq_idx,
                pol=self.pol,
                var=self.var,
                robust=self.robust,
            )
        except ValueError:
            return  # No WCS header — skip overlay

        time_label = self._time_labels.get(self.time_idx, "?")
        freq_label = self._freq_labels.get(self.freq_idx, "?")
        overlay_name = f"OVRO-LWA {self.var} t={time_label} f={freq_label}"

        self._aladin.add_fits(
            hdul,
            name=overlay_name,
            opacity=self.overlay_opacity,
            colormap=self.colormap,
            stretch=self.stretch,
        )
        self._current_overlay_name = overlay_name

    def _on_param_change(self, event: Any) -> None:
        """React to parameter changes by updating the overlay."""
        if event.name == "survey":
            self._aladin.survey = SURVEY_PRESETS[self.survey]
        elif event.name == "fov":
            self._aladin.fov = self.fov
        elif event.name in (
            "time_idx", "freq_idx", "pol", "var",
            "overlay_opacity", "colormap", "stretch", "robust",
        ):
            self._update_overlay()

    @param.depends("time_idx")
    def _time_label(self) -> str:
        return f"**Time:** {self._time_labels.get(self.time_idx, '?')} MJD"

    @param.depends("freq_idx")
    def _freq_label(self) -> str:
        return f"**Frequency:** {self._freq_labels.get(self.freq_idx, '?')}"

    def panel(self) -> pn.viewable.Viewable:
        """Return the complete Panel layout with sky viewer.

        Returns
        -------
        pn.viewable.Viewable
            Panel layout with Aladin sky viewer and controls.
        """
        # Enable ipywidgets integration for Panel (required for ipyaladin)
        try:
            pn.extension("ipywidgets")
        except Exception:  # noqa: BLE001
            pass  # May fail outside notebook context; widget still works

        # Watch all relevant parameters
        self.param.watch(
            self._on_param_change,
            [
                "time_idx", "freq_idx", "pol", "var",
                "survey", "overlay_opacity", "colormap", "stretch",
                "robust", "fov",
            ],
        )

        time_label = pn.pane.Markdown(self._time_label, width=250)
        freq_label = pn.pane.Markdown(self._freq_label, width=250)

        controls = pn.Column(
            "## Sky Viewer",
            pn.widgets.IntSlider.from_param(self.param.time_idx, name="Time Step"),
            time_label,
            pn.widgets.IntSlider.from_param(self.param.freq_idx, name="Frequency"),
            freq_label,
            pn.widgets.IntSlider.from_param(self.param.pol, name="Polarization"),
            pn.widgets.Select.from_param(self.param.var, name="Variable"),
            "---",
            pn.widgets.Select.from_param(self.param.survey, name="Background Survey"),
            pn.widgets.FloatSlider.from_param(
                self.param.overlay_opacity, name="Overlay Opacity", step=0.05
            ),
            pn.widgets.Select.from_param(self.param.colormap, name="Colormap"),
            pn.widgets.Select.from_param(self.param.stretch, name="Stretch"),
            pn.widgets.Checkbox.from_param(self.param.robust, name="Robust Clipping"),
            pn.widgets.FloatSlider.from_param(
                self.param.fov, name="Field of View (\u00b0)", step=1.0
            ),
            width=280,
        )

        # Display the Aladin widget via ipywidgets Output to avoid
        # conflicts with Panel's comms="default" mode. IPyWidget pane
        # requires ipywidgets comms which may not be active.
        import ipywidgets as ipw

        output = ipw.Output()
        with output:
            from IPython.display import display
            display(self._aladin)

        aladin_pane = pn.pane.IPyWidget(output, width=700, height=600)

        # Load initial overlay after widget renders
        def _load_initial_overlay(event: Any = None) -> None:
            self._update_overlay()

        pn.state.onload(_load_initial_overlay)

        return pn.Row(controls, aladin_pane, sizing_mode="stretch_width", min_height=600)
