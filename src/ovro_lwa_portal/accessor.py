"""xarray accessor for OVRO-LWA radio astronomy datasets.

This module provides the `radport` accessor that extends xarray Datasets
with domain-specific methods for OVRO-LWA data visualization and analysis.

Example
-------
>>> import ovro_lwa_portal
>>> from ovro_lwa_portal import open_dataset
>>> ds = open_dataset("path/to/data.zarr")
>>> ds.radport.plot()  # Create default visualization
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import interpolate

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@xr.register_dataset_accessor("radport")
class RadportAccessor:
    """xarray accessor for OVRO-LWA radio astronomy datasets.

    This accessor provides domain-specific methods for working with
    OVRO-LWA data, including visualization and validation utilities.

    The accessor is automatically available on any xarray Dataset after
    importing the `ovro_lwa_portal` package.

    Parameters
    ----------
    xarray_obj : xr.Dataset
        The xarray Dataset to extend with accessor methods.

    Raises
    ------
    ValueError
        If the dataset is missing required dimensions or variables
        for OVRO-LWA data.

    Example
    -------
    >>> import ovro_lwa_portal
    >>> from ovro_lwa_portal import open_dataset
    >>> ds = open_dataset("path/to/data.zarr")
    >>> ds.radport.plot()
    """

    # Required dimensions for OVRO-LWA datasets
    _required_dims: frozenset[str] = frozenset(
        {"time", "frequency", "polarization", "l", "m"}
    )

    # Required data variables
    _required_vars: frozenset[str] = frozenset({"SKY"})

    # Optional data variables
    _optional_vars: frozenset[str] = frozenset({"BEAM"})

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialize the RadportAccessor.

        Parameters
        ----------
        xarray_obj : xr.Dataset
            The xarray Dataset to extend.

        Raises
        ------
        ValueError
            If the dataset structure is invalid for OVRO-LWA data.
        """
        self._obj = xarray_obj
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate that the dataset has required dimensions and variables.

        Raises
        ------
        ValueError
            If required dimensions or variables are missing, with an
            informative error message listing what is missing.
        """
        # Check for required dimensions
        missing_dims = self._required_dims - set(self._obj.dims)
        if missing_dims:
            raise ValueError(
                f"Dataset is missing required dimensions for OVRO-LWA data: "
                f"{sorted(missing_dims)}. "
                f"Expected dimensions: {sorted(self._required_dims)}. "
                f"Found dimensions: {sorted(self._obj.dims)}."
            )

        # Check for required data variables
        missing_vars = self._required_vars - set(self._obj.data_vars)
        if missing_vars:
            raise ValueError(
                f"Dataset is missing required variables for OVRO-LWA data: "
                f"{sorted(missing_vars)}. "
                f"Expected variables: {sorted(self._required_vars)}. "
                f"Found variables: {sorted(self._obj.data_vars)}."
            )

    @property
    def has_beam(self) -> bool:
        """Check if the dataset contains BEAM data.

        Returns
        -------
        bool
            True if the dataset contains a BEAM variable.
        """
        return "BEAM" in self._obj.data_vars

    # =========================================================================
    # Selection Helper Methods
    # =========================================================================

    def nearest_freq_idx(self, freq_mhz: float) -> int:
        """Find the index of the frequency nearest to the given value in MHz.

        Parameters
        ----------
        freq_mhz : float
            Target frequency in MHz.

        Returns
        -------
        int
            Index of the nearest frequency in the dataset.

        Examples
        --------
        >>> idx = ds.radport.nearest_freq_idx(50.0)  # Find index nearest to 50 MHz
        >>> ds.radport.plot(freq_idx=idx)
        """
        freq_hz = freq_mhz * 1e6
        freq_values = self._obj.coords["frequency"].values
        return int(np.argmin(np.abs(freq_values - freq_hz)))

    def nearest_time_idx(self, mjd: float) -> int:
        """Find the index of the time nearest to the given MJD value.

        Parameters
        ----------
        mjd : float
            Target time in Modified Julian Date (MJD).

        Returns
        -------
        int
            Index of the nearest time in the dataset.

        Examples
        --------
        >>> idx = ds.radport.nearest_time_idx(60000.5)  # Find index nearest to MJD
        >>> ds.radport.plot(time_idx=idx)
        """
        time_values = self._obj.coords["time"].values
        return int(np.argmin(np.abs(time_values - mjd)))

    def nearest_lm_idx(self, l: float, m: float) -> tuple[int, int]:
        """Find the indices of the (l, m) pixel nearest to the given coordinates.

        Parameters
        ----------
        l : float
            Target l direction cosine coordinate.
        m : float
            Target m direction cosine coordinate.

        Returns
        -------
        tuple of int
            (l_idx, m_idx) indices of the nearest pixel.

        Examples
        --------
        >>> l_idx, m_idx = ds.radport.nearest_lm_idx(0.0, 0.0)  # Find center pixel
        """
        l_values = self._obj.coords["l"].values
        m_values = self._obj.coords["m"].values
        l_idx = int(np.argmin(np.abs(l_values - l)))
        m_idx = int(np.argmin(np.abs(m_values - m)))
        return l_idx, m_idx

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Create a visualization of radio data as a 2D image.

        Plots a single snapshot of the data at the specified time, frequency,
        and polarization indices. The resulting image shows intensity values
        in the (l, m) direction cosine coordinate system.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot. Use 'BEAM' only if the dataset contains
            beam data (check with `ds.radport.has_beam`).
        time_idx : int, optional
            Index along the time dimension for the snapshot. Default is 0.
            Ignored if `time_mjd` is provided.
        freq_idx : int, optional
            Index along the frequency dimension for the snapshot. Default is 0.
            Ignored if `freq_mhz` is provided.
        pol : int, default 0
            Index along the polarization dimension.
        freq_mhz : float, optional
            Select frequency by value in MHz. If provided, overrides `freq_idx`.
            Uses the nearest available frequency.
        time_mjd : float, optional
            Select time by MJD value. If provided, overrides `time_idx`.
            Uses the nearest available time.
        cmap : str, default 'inferno'
            Matplotlib colormap name for the image.
        vmin : float, optional
            Minimum value for the color scale. If None and robust=False,
            uses the data minimum.
        vmax : float, optional
            Maximum value for the color scale. If None and robust=False,
            uses the data maximum.
        robust : bool, default False
            If True, compute vmin/vmax using the 2nd and 98th percentiles
            of the data, which is useful for data with outliers.
        mask_radius : int, optional
            If provided, mask pixels outside this radius (in pixels) from
            the image center. Useful for all-sky images where edge pixels
            are invalid. Masked pixels are shown as NaN (transparent).
        figsize : tuple of float, default (8, 6)
            Figure size in inches as (width, height).
        add_colorbar : bool, default True
            Whether to add a colorbar to the plot.
        **kwargs : dict
            Additional keyword arguments passed to `matplotlib.pyplot.imshow`.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the requested variable does not exist in the dataset.

        Examples
        --------
        >>> import ovro_lwa_portal
        >>> ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")

        Plot with default settings (first time, frequency, polarization):

        >>> fig = ds.radport.plot()

        Plot a specific time and frequency with custom colormap:

        >>> fig = ds.radport.plot(time_idx=5, freq_idx=10, cmap='viridis')

        Plot by selecting frequency in MHz (more intuitive):

        >>> fig = ds.radport.plot(freq_mhz=50.0)

        Plot with fixed color scale:

        >>> fig = ds.radport.plot(vmin=-1.0, vmax=16.0)

        Plot with robust color scaling for data with outliers:

        >>> fig = ds.radport.plot(robust=True)

        Plot with circular mask to hide invalid edge pixels:

        >>> fig = ds.radport.plot(mask_radius=1800)
        """
        # Validate the requested variable exists
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            msg = f"Variable '{var}' not found in dataset. Available variables: {available}"
            raise ValueError(msg)

        # Resolve frequency selection: freq_mhz takes precedence over freq_idx
        if freq_mhz is not None:
            freq_idx = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is None:
            freq_idx = 0

        # Resolve time selection: time_mjd takes precedence over time_idx
        if time_mjd is not None:
            time_idx = self.nearest_time_idx(time_mjd)
        elif time_idx is None:
            time_idx = 0

        # Extract the 2D slice for plotting
        da = self._obj[var].isel(
            time=time_idx,
            frequency=freq_idx,
            polarization=pol,
        )

        # Build title with metadata
        title = self._build_plot_title(var, time_idx, freq_idx, pol)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data for plotting (triggers dask computation if needed)
        data = da.values.copy()  # Copy to allow modification for masking

        # Apply circular mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            distance_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[distance_from_center > mask_radius] = np.nan

        # Handle robust scaling (after masking, so we only consider valid pixels)
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate extents for proper axis labeling
        l_vals = da.coords["l"].values
        m_vals = da.coords["m"].values
        extent = [float(l_vals.min()), float(l_vals.max()),
                  float(m_vals.min()), float(m_vals.max())]

        # Plot the image
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Set labels and title
        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(title)

        fig.tight_layout()

        return fig

    def _build_plot_title(
        self,
        var: str,
        time_idx: int,
        freq_idx: int,
        pol: int,
    ) -> str:
        """Build an informative title for the plot.

        Parameters
        ----------
        var : str
            The variable being plotted.
        time_idx : int
            Time index.
        freq_idx : int
            Frequency index.
        pol : int
            Polarization index.

        Returns
        -------
        str
            Formatted title string.
        """
        # Get time value
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f} MJD"
        except (TypeError, ValueError):
            time_str = str(time_val)

        # Get frequency value in MHz
        freq_val = self._obj.coords["frequency"].values[freq_idx]
        freq_mhz = float(freq_val) / 1e6

        return f"{var}: t={time_str}, f={freq_mhz:.2f} MHz, pol={pol}"

    # =========================================================================
    # Spatial Cutout Methods
    # =========================================================================

    def cutout(
        self,
        l_center: float,
        m_center: float,
        dl: float,
        dm: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
    ) -> xr.DataArray:
        """Extract a spatial cutout (rectangular region) from the data.

        Returns a 2D DataArray containing data within the specified (l, m)
        bounding box for a given time, frequency, and polarization.

        Parameters
        ----------
        l_center : float
            Center l coordinate of the cutout region.
        m_center : float
            Center m coordinate of the cutout region.
        dl : float
            Half-width of the cutout in the l direction.
            The cutout spans [l_center - dl, l_center + dl].
        dm : float
            Half-width of the cutout in the m direction.
            The cutout spans [m_center - dm, m_center + dm].
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz (overrides `freq_idx`).
        time_mjd : float, optional
            Select time by MJD value (overrides `time_idx`).

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (l, m) containing the cutout data.
            Includes metadata attributes: cutout_l_center, cutout_m_center,
            cutout_dl, cutout_dm.

        Raises
        ------
        ValueError
            If the requested variable does not exist or cutout is empty.

        Examples
        --------
        >>> # Extract 0.2 x 0.2 region centered at (0, 0)
        >>> cutout = ds.radport.cutout(l_center=0.0, m_center=0.0, dl=0.1, dm=0.1)

        >>> # Extract at specific frequency
        >>> cutout = ds.radport.cutout(0.0, 0.0, 0.1, 0.1, freq_mhz=50.0)

        >>> # Plot the cutout
        >>> cutout.plot()
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        # Resolve indices
        if freq_mhz is not None:
            freq_idx = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is None:
            freq_idx = 0

        if time_mjd is not None:
            time_idx = self.nearest_time_idx(time_mjd)
        elif time_idx is None:
            time_idx = 0

        # Extract 2D slice
        da = self._obj[var].isel(
            time=time_idx,
            frequency=freq_idx,
            polarization=pol,
        )

        # Compute l/m bounds
        l_min, l_max = l_center - dl, l_center + dl
        m_min, m_max = m_center - dm, m_center + dm

        # Handle coordinate ordering (ascending or descending)
        l_coords = da.coords["l"]
        m_coords = da.coords["m"]

        # Determine slice direction based on coordinate ordering
        if float(l_coords[0]) <= float(l_coords[-1]):
            l_slice = slice(l_min, l_max)
        else:
            l_slice = slice(l_max, l_min)

        if float(m_coords[0]) <= float(m_coords[-1]):
            m_slice = slice(m_min, m_max)
        else:
            m_slice = slice(m_max, m_min)

        # Select the cutout region
        cutout = da.sel(l=l_slice, m=m_slice)

        # Check if cutout is empty
        if cutout.size == 0:
            raise ValueError(
                f"Cutout region is empty. Requested l=[{l_min:.3f}, {l_max:.3f}], "
                f"m=[{m_min:.3f}, {m_max:.3f}]. "
                f"Dataset l range: [{float(l_coords.min()):.3f}, {float(l_coords.max()):.3f}], "
                f"m range: [{float(m_coords.min()):.3f}, {float(m_coords.max()):.3f}]."
            )

        # Add metadata attributes
        cutout.attrs["cutout_l_center"] = l_center
        cutout.attrs["cutout_m_center"] = m_center
        cutout.attrs["cutout_dl"] = dl
        cutout.attrs["cutout_dm"] = dm
        cutout.attrs["time_idx"] = time_idx
        cutout.attrs["freq_idx"] = freq_idx
        cutout.attrs["pol"] = pol

        return cutout

    def plot_cutout(
        self,
        l_center: float,
        m_center: float,
        dl: float,
        dm: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (6, 5),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Extract and plot a spatial cutout.

        Convenience method that combines `cutout()` with plotting.

        Parameters
        ----------
        l_center, m_center : float
            Center coordinates of the cutout region.
        dl, dm : float
            Half-widths of the cutout in l and m directions.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        time_idx : int, optional
            Time index. Default is 0.
        freq_idx : int, optional
            Frequency index. Default is 0.
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        time_mjd : float, optional
            Select time by MJD value.
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based color scaling.
        figsize : tuple, default (6, 5)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the cutout plot.

        Examples
        --------
        >>> fig = ds.radport.plot_cutout(0.0, 0.0, 0.1, 0.1, freq_mhz=50.0)
        """
        # Get cutout data
        cutout = self.cutout(
            l_center=l_center,
            m_center=m_center,
            dl=dl,
            dm=dm,
            var=var,
            time_idx=time_idx,
            freq_idx=freq_idx,
            pol=pol,
            freq_mhz=freq_mhz,
            time_mjd=time_mjd,
        )

        # Resolve actual indices for title
        actual_time_idx = cutout.attrs["time_idx"]
        actual_freq_idx = cutout.attrs["freq_idx"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = cutout.values

        # Handle robust scaling
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate extents
        l_vals = cutout.coords["l"].values
        m_vals = cutout.coords["m"].values
        extent = [
            float(l_vals.min()), float(l_vals.max()),
            float(m_vals.min()), float(m_vals.max()),
        ]

        # Plot
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Build title
        title = self._build_plot_title(var, actual_time_idx, actual_freq_idx, pol)
        title += f"\nl=[{l_center-dl:+.2f},{l_center+dl:+.2f}], m=[{m_center-dm:+.2f},{m_center+dm:+.2f}]"

        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(title)

        fig.tight_layout()
        return fig

    # =========================================================================
    # Dynamic Spectrum Methods
    # =========================================================================

    def dynamic_spectrum(
        self,
        l: float,
        m: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.DataArray:
        """Extract a dynamic spectrum (time vs frequency) for a single pixel.

        Returns a 2D DataArray showing how intensity varies across time
        and frequency at the pixel nearest to the specified (l, m) location.

        Parameters
        ----------
        l : float
            Target l coordinate for pixel selection.
        m : float
            Target m coordinate for pixel selection.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (time, frequency).
            Includes metadata: pixel_l, pixel_m, l_idx, m_idx, pol.

        Examples
        --------
        >>> # Get dynamic spectrum at image center
        >>> dynspec = ds.radport.dynamic_spectrum(l=0.0, m=0.0)

        >>> # Plot it
        >>> dynspec.plot(x='time', y='frequency')
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        # Find nearest pixel
        l_idx, m_idx = self.nearest_lm_idx(l, m)

        # Extract (time, frequency) slice at this pixel
        da = self._obj[var].isel(l=l_idx, m=m_idx, polarization=pol)

        # Sort by time and frequency for consistent plotting
        if "time" in da.dims:
            da = da.sortby("time")
        if "frequency" in da.dims:
            da = da.sortby("frequency")

        # Add metadata
        da.attrs["pixel_l"] = float(self._obj.coords["l"].values[l_idx])
        da.attrs["pixel_m"] = float(self._obj.coords["m"].values[m_idx])
        da.attrs["l_idx"] = l_idx
        da.attrs["m_idx"] = m_idx
        da.attrs["pol"] = pol

        return da

    def plot_dynamic_spectrum(
        self,
        l: float,
        m: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (8, 5),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot a dynamic spectrum (time vs frequency) for a single pixel.

        Creates a 2D visualization showing intensity variations across
        time and frequency at a specified (l, m) location.

        Parameters
        ----------
        l : float
            Target l coordinate for pixel selection.
        m : float
            Target m coordinate for pixel selection.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based color scaling.
        figsize : tuple, default (8, 5)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the dynamic spectrum plot.

        Examples
        --------
        >>> fig = ds.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
        """
        # Get dynamic spectrum
        dynspec = self.dynamic_spectrum(l=l, m=m, var=var, pol=pol)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = dynspec.values

        # Handle robust scaling
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate values
        time_vals = dynspec.coords["time"].values
        freq_vals = dynspec.coords["frequency"].values / 1e6  # Convert to MHz

        # Compute extent for imshow
        # extent = [xmin, xmax, ymin, ymax]
        extent = [
            float(time_vals.min()), float(time_vals.max()),
            float(freq_vals.min()), float(freq_vals.max()),
        ]

        # Plot - transpose so time is x-axis and frequency is y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="auto",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Labels and title
        pixel_l = dynspec.attrs["pixel_l"]
        pixel_m = dynspec.attrs["pixel_m"]
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(
            f"{var} Dynamic Spectrum at l={pixel_l:+.4f}, m={pixel_m:+.4f}, pol={pol}"
        )

        fig.tight_layout()
        return fig

    # =========================================================================
    # Difference Map Methods
    # =========================================================================

    def diff(
        self,
        mode: Literal["time", "frequency"] = "time",
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
    ) -> xr.DataArray:
        """Compute a difference map between adjacent time or frequency slices.

        Useful for identifying transient sources or spectral features by
        subtracting consecutive frames.

        Parameters
        ----------
        mode : {'time', 'frequency'}, default 'time'
            Difference mode:
            - 'time': Subtract previous time step from current (at fixed freq)
            - 'frequency': Subtract previous frequency from current (at fixed time)
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to difference.
        time_idx : int, optional
            Time index for the "current" frame. Default is 1.
            For mode='time', differences frame[time_idx] - frame[time_idx-1].
        freq_idx : int, optional
            Frequency index for the "current" frame. Default is 1.
            For mode='frequency', differences frame[freq_idx] - frame[freq_idx-1].
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        time_mjd : float, optional
            Select time by MJD value.

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (l, m) containing the difference.
            Includes metadata: diff_mode, idx1, idx2.

        Raises
        ------
        ValueError
            If indices are out of bounds for differencing.

        Examples
        --------
        >>> # Time difference at fixed frequency
        >>> diff = ds.radport.diff(mode='time', time_idx=5, freq_mhz=50.0)

        >>> # Frequency difference at fixed time
        >>> diff = ds.radport.diff(mode='frequency', freq_idx=10, time_idx=0)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        # Resolve indices
        if freq_mhz is not None:
            freq_idx = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is None:
            freq_idx = 1 if mode == "frequency" else 0

        if time_mjd is not None:
            time_idx = self.nearest_time_idx(time_mjd)
        elif time_idx is None:
            time_idx = 1 if mode == "time" else 0

        # Compute difference based on mode
        if mode == "time":
            if time_idx < 1:
                raise ValueError(
                    f"time_idx must be >= 1 for time differencing. Got {time_idx}."
                )
            n_times = self._obj.sizes["time"]
            if time_idx >= n_times:
                raise ValueError(
                    f"time_idx {time_idx} out of bounds (dataset has {n_times} times)."
                )

            frame_current = self._obj[var].isel(
                time=time_idx, frequency=freq_idx, polarization=pol
            )
            frame_prev = self._obj[var].isel(
                time=time_idx - 1, frequency=freq_idx, polarization=pol
            )
            diff = frame_current - frame_prev

            diff.attrs["diff_mode"] = "time"
            diff.attrs["time_idx_current"] = time_idx
            diff.attrs["time_idx_prev"] = time_idx - 1
            diff.attrs["freq_idx"] = freq_idx

        else:  # mode == "frequency"
            if freq_idx < 1:
                raise ValueError(
                    f"freq_idx must be >= 1 for frequency differencing. Got {freq_idx}."
                )
            n_freqs = self._obj.sizes["frequency"]
            if freq_idx >= n_freqs:
                raise ValueError(
                    f"freq_idx {freq_idx} out of bounds (dataset has {n_freqs} frequencies)."
                )

            frame_current = self._obj[var].isel(
                time=time_idx, frequency=freq_idx, polarization=pol
            )
            frame_prev = self._obj[var].isel(
                time=time_idx, frequency=freq_idx - 1, polarization=pol
            )
            diff = frame_current - frame_prev

            diff.attrs["diff_mode"] = "frequency"
            diff.attrs["freq_idx_current"] = freq_idx
            diff.attrs["freq_idx_prev"] = freq_idx - 1
            diff.attrs["time_idx"] = time_idx

        diff.attrs["pol"] = pol
        return diff

    def plot_diff(
        self,
        mode: Literal["time", "frequency"] = "time",
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric: bool = True,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot a difference map between adjacent time or frequency slices.

        Parameters
        ----------
        mode : {'time', 'frequency'}, default 'time'
            Difference mode.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to difference.
        time_idx : int, optional
            Time index for the "current" frame.
        freq_idx : int, optional
            Frequency index for the "current" frame.
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        time_mjd : float, optional
            Select time by MJD value.
        cmap : str, default 'RdBu_r'
            Colormap (diverging colormaps work well for differences).
        vmin, vmax : float, optional
            Color scale limits.
        symmetric : bool, default True
            If True and vmin/vmax not specified, use symmetric color scale
            centered on zero.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the difference plot.

        Examples
        --------
        >>> fig = ds.radport.plot_diff(mode='time', time_idx=5, freq_mhz=50.0)
        """
        # Get difference data
        diff = self.diff(
            mode=mode,
            var=var,
            time_idx=time_idx,
            freq_idx=freq_idx,
            pol=pol,
            freq_mhz=freq_mhz,
            time_mjd=time_mjd,
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = diff.values

        # Handle symmetric scaling
        if symmetric and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                max_abs = float(np.percentile(np.abs(finite_data), 98))
                vmin, vmax = -max_abs, max_abs

        # Get coordinate extents
        l_vals = diff.coords["l"].values
        m_vals = diff.coords["m"].values
        extent = [
            float(l_vals.min()), float(l_vals.max()),
            float(m_vals.min()), float(m_vals.max()),
        ]

        # Plot
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("ΔJy/beam")

        # Build title
        if mode == "time":
            t_curr = diff.attrs["time_idx_current"]
            t_prev = diff.attrs["time_idx_prev"]
            f_idx = diff.attrs["freq_idx"]
            freq_mhz_val = float(self._obj.coords["frequency"].values[f_idx]) / 1e6
            title = f"{var} Time Diff (t{t_curr} - t{t_prev}) at f={freq_mhz_val:.2f} MHz"
        else:
            f_curr = diff.attrs["freq_idx_current"]
            f_prev = diff.attrs["freq_idx_prev"]
            t_idx = diff.attrs["time_idx"]
            freq_curr_mhz = float(self._obj.coords["frequency"].values[f_curr]) / 1e6
            freq_prev_mhz = float(self._obj.coords["frequency"].values[f_prev]) / 1e6
            time_val = self._obj.coords["time"].values[t_idx]
            title = f"{var} Freq Diff ({freq_curr_mhz:.1f} - {freq_prev_mhz:.1f} MHz) at t={float(time_val):.6f}"

        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(title)

        fig.tight_layout()
        return fig

    # =========================================================================
    # Data Quality Methods
    # =========================================================================

    def find_valid_frame(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        min_finite_fraction: float = 0.1,
    ) -> tuple[int, int]:
        """Find the first (time, freq) frame with sufficient finite data.

        Searches through time and frequency indices to find a frame where
        at least `min_finite_fraction` of pixels contain finite (non-NaN) values.
        Useful for automatically selecting a valid frame for visualization.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to check.
        pol : int, default 0
            Polarization index.
        min_finite_fraction : float, default 0.1
            Minimum fraction of finite pixels required (0 to 1).

        Returns
        -------
        tuple of int
            (time_idx, freq_idx) of the first valid frame.

        Raises
        ------
        ValueError
            If no valid frame is found.

        Examples
        --------
        >>> time_idx, freq_idx = ds.radport.find_valid_frame()
        >>> fig = ds.radport.plot(time_idx=time_idx, freq_idx=freq_idx)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)

        # Compute fraction of finite values for each (time, freq) plane
        finite_frac = np.isfinite(da).mean(dim=("l", "m"))

        # If data is lazy (dask), compute it
        if hasattr(finite_frac, "compute"):
            finite_frac = finite_frac.compute()

        arr = finite_frac.values

        # Search for first valid frame
        for ti in range(arr.shape[0]):
            for fi in range(arr.shape[1]):
                if arr[ti, fi] >= min_finite_fraction:
                    return ti, fi

        raise ValueError(
            f"No valid frame found with at least {min_finite_fraction:.0%} finite pixels. "
            f"Dataset may contain all NaN values."
        )

    def finite_fraction(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.DataArray:
        """Compute the fraction of finite (non-NaN) pixels for each (time, freq).

        Returns a 2D DataArray showing data availability across all
        time and frequency combinations. Useful for identifying which
        frames contain valid data.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to check.
        pol : int, default 0
            Polarization index.

        Returns
        -------
        xr.DataArray
            2D array with dimensions (time, frequency) containing fractions
            from 0 (all NaN) to 1 (all finite).

        Examples
        --------
        >>> frac = ds.radport.finite_fraction()
        >>> frac.plot()  # Visualize data availability
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)
        finite_frac = np.isfinite(da).mean(dim=("l", "m"))

        finite_frac.attrs["variable"] = var
        finite_frac.attrs["pol"] = pol

        return finite_frac

    # =========================================================================
    # Grid Plot Methods
    # =========================================================================

    def plot_grid(
        self,
        time_indices: list[int] | None = None,
        freq_indices: list[int] | None = None,
        freq_mhz_list: list[float] | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        ncols: int = 4,
        panel_size: tuple[float, float] = (3.0, 2.6),
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        share_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Create a grid of plots showing multiple time/frequency combinations.

        Useful for comparing observations across time or frequency in a
        single figure with consistent scaling.

        Parameters
        ----------
        time_indices : list of int, optional
            Time indices to plot. If None, uses all available times.
        freq_indices : list of int, optional
            Frequency indices to plot. If None, uses all available frequencies.
            Ignored if `freq_mhz_list` is provided.
        freq_mhz_list : list of float, optional
            Frequencies in MHz to plot. Overrides `freq_indices`.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        ncols : int, default 4
            Number of columns in the grid.
        panel_size : tuple of float, default (3.0, 2.6)
            Size of each panel in inches as (width, height).
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits. Applied to all panels.
        robust : bool, default True
            If True and vmin/vmax not specified, compute global percentile
            scaling across all panels.
        mask_radius : int, optional
            Circular mask radius in pixels.
        share_colorbar : bool, default True
            If True, show a single shared colorbar for all panels.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the grid of plots.

        Examples
        --------
        >>> # Plot all times at a single frequency
        >>> fig = ds.radport.plot_grid(freq_mhz_list=[50.0])

        >>> # Plot specific times and frequencies
        >>> fig = ds.radport.plot_grid(
        ...     time_indices=[0, 1, 2],
        ...     freq_mhz_list=[46.0, 50.0, 54.0],
        ... )

        >>> # Plot first 4 times at all frequencies
        >>> fig = ds.radport.plot_grid(time_indices=[0, 1, 2, 3])
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Resolve time indices
        if time_indices is None:
            time_indices = list(range(self._obj.sizes["time"]))

        # Resolve frequency indices
        if freq_mhz_list is not None:
            freq_indices = [self.nearest_freq_idx(f) for f in freq_mhz_list]
        elif freq_indices is None:
            freq_indices = list(range(self._obj.sizes["frequency"]))

        # Calculate grid dimensions
        n_panels = len(time_indices) * len(freq_indices)
        if n_panels == 0:
            raise ValueError("No panels to plot. Check time_indices and freq_indices.")

        nrows = int(np.ceil(n_panels / ncols))

        # Create figure
        fig_width = panel_size[0] * ncols
        fig_height = panel_size[1] * nrows
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )

        # Collect all data for global scaling if robust=True
        all_data = []
        panel_data = []

        for ti in time_indices:
            for fi in freq_indices:
                da = self._obj[var].isel(
                    time=ti, frequency=fi, polarization=pol
                )
                data = da.values.copy()

                # Apply mask if requested
                if mask_radius is not None:
                    ny, nx = data.shape
                    cy, cx = ny // 2, nx // 2
                    yy, xx = np.ogrid[:ny, :nx]
                    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                    data[dist > mask_radius] = np.nan

                panel_data.append((ti, fi, data, da))
                if robust and vmin is None and vmax is None:
                    finite = data[np.isfinite(data)]
                    if finite.size > 0:
                        all_data.append(finite)

        # Compute global vmin/vmax if robust
        if robust and vmin is None and vmax is None and all_data:
            all_finite = np.concatenate(all_data)
            vmin = float(np.percentile(all_finite, 2))
            vmax = float(np.percentile(all_finite, 98))

        # Plot each panel
        im = None
        for idx, (ti, fi, data, da) in enumerate(panel_data):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            # Get coordinate extents
            l_vals = da.coords["l"].values
            m_vals = da.coords["m"].values
            extent = [
                float(l_vals.min()), float(l_vals.max()),
                float(m_vals.min()), float(m_vals.max()),
            ]

            # Check if panel has data
            has_data = np.any(np.isfinite(data))

            if has_data:
                im = ax.imshow(
                    data,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    aspect="equal",
                    **kwargs,
                )
            else:
                ax.text(
                    0.5, 0.5, "No Data",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

            # Build panel title
            time_val = self._obj.coords["time"].values[ti]
            freq_val = self._obj.coords["frequency"].values[fi] / 1e6
            try:
                time_str = f"{float(time_val):.6f}"
            except (TypeError, ValueError):
                time_str = str(time_val)

            ax.set_title(f"t={time_str}\nf={freq_val:.2f} MHz", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused panels
        for idx in range(n_panels, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].axis("off")

        # Add shared colorbar
        if share_colorbar and im is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Jy/beam")

        fig.suptitle(f"{var} Grid (pol={pol})", fontsize=12, y=1.02)

        return fig

    def plot_frequency_grid(
        self,
        time_idx: int = 0,
        freq_mhz_list: list[float] | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        ncols: int = 4,
        panel_size: tuple[float, float] = (3.0, 2.6),
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Create a grid showing all frequencies at a fixed time.

        Convenience method for comparing across frequency channels.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for all panels.
        freq_mhz_list : list of float, optional
            Specific frequencies to plot. If None, plots all frequencies.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        ncols : int, default 4
            Number of columns.
        panel_size : tuple, default (3.0, 2.6)
            Size of each panel.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based scaling.
        mask_radius : int, optional
            Circular mask radius.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_frequency_grid(time_idx=0)
        """
        return self.plot_grid(
            time_indices=[time_idx],
            freq_mhz_list=freq_mhz_list,
            var=var,
            pol=pol,
            ncols=ncols,
            panel_size=panel_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            mask_radius=mask_radius,
            **kwargs,
        )

    def plot_time_grid(
        self,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        time_indices: list[int] | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        ncols: int = 4,
        panel_size: tuple[float, float] = (3.0, 2.6),
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Create a grid showing all times at a fixed frequency.

        Convenience method for comparing across time steps (time evolution).

        Parameters
        ----------
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        time_indices : list of int, optional
            Specific time indices to plot. If None, plots all times.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        ncols : int, default 4
            Number of columns.
        panel_size : tuple, default (3.0, 2.6)
            Size of each panel.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based scaling.
        mask_radius : int, optional
            Circular mask radius.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_time_grid(freq_mhz=50.0)
        """
        # Resolve frequency
        if freq_mhz is not None:
            freq_indices = [self.nearest_freq_idx(freq_mhz)]
        elif freq_idx is not None:
            freq_indices = [freq_idx]
        else:
            freq_indices = [0]

        return self.plot_grid(
            time_indices=time_indices,
            freq_indices=freq_indices,
            var=var,
            pol=pol,
            ncols=ncols,
            panel_size=panel_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            mask_radius=mask_radius,
            **kwargs,
        )

    # =========================================================================
    # 1D Analysis Methods
    # =========================================================================

    def light_curve(
        self,
        l: float,
        m: float,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.DataArray:
        """Extract a light curve (time series) at a specific spatial location.

        Returns intensity as a function of time at the pixel nearest to
        the specified (l, m) coordinates and frequency.

        Parameters
        ----------
        l : float
            Direction cosine l coordinate.
        m : float
            Direction cosine m coordinate.
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.

        Returns
        -------
        xr.DataArray
            1D array with dimension 'time' containing the light curve.

        Examples
        --------
        >>> lc = ds.radport.light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        >>> lc.plot()  # Plot intensity vs time
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Find nearest pixel
        l_idx, m_idx = self.nearest_lm_idx(l, m)

        # Extract light curve
        lc = self._obj[var].isel(
            frequency=fi,
            polarization=pol,
            l=l_idx,
            m=m_idx,
        )

        # Add metadata
        freq_hz = float(self._obj.coords["frequency"].values[fi])
        l_val = float(self._obj.coords["l"].values[l_idx])
        m_val = float(self._obj.coords["m"].values[m_idx])

        lc.attrs["variable"] = var
        lc.attrs["freq_idx"] = fi
        lc.attrs["freq_mhz"] = freq_hz / 1e6
        lc.attrs["pol"] = pol
        lc.attrs["l"] = l_val
        lc.attrs["m"] = m_val
        lc.attrs["l_idx"] = l_idx
        lc.attrs["m_idx"] = m_idx

        return lc

    def plot_light_curve(
        self,
        l: float,
        m: float,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        figsize: tuple[float, float] = (10, 4),
        marker: str = "o",
        linestyle: str = "-",
        **kwargs: Any,
    ) -> Figure:
        """Plot a light curve (time series) at a specific spatial location.

        Parameters
        ----------
        l : float
            Direction cosine l coordinate.
        m : float
            Direction cosine m coordinate.
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        figsize : tuple, default (10, 4)
            Figure size in inches.
        marker : str, default 'o'
            Marker style for data points.
        linestyle : str, default '-'
            Line style connecting points.
        **kwargs : dict
            Additional arguments passed to plt.plot.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        """
        lc = self.light_curve(
            l=l, m=m, freq_idx=freq_idx, freq_mhz=freq_mhz, var=var, pol=pol
        )

        fig, ax = plt.subplots(figsize=figsize)

        time_vals = lc.coords["time"].values
        ax.plot(time_vals, lc.values, marker=marker, linestyle=linestyle, **kwargs)

        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel(f"{var} Intensity (Jy/beam)")

        freq_mhz_val = lc.attrs["freq_mhz"]
        l_val = lc.attrs["l"]
        m_val = lc.attrs["m"]
        ax.set_title(
            f"{var} Light Curve at (l={l_val:.3f}, m={m_val:.3f}), "
            f"f={freq_mhz_val:.2f} MHz, pol={pol}"
        )

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig

    def spectrum(
        self,
        l: float,
        m: float,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.DataArray:
        """Extract a frequency spectrum at a specific spatial location and time.

        Returns intensity as a function of frequency at the pixel nearest to
        the specified (l, m) coordinates and time.

        Parameters
        ----------
        l : float
            Direction cosine l coordinate.
        m : float
            Direction cosine m coordinate.
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        time_mjd : float, optional
            Time in MJD (overrides time_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.

        Returns
        -------
        xr.DataArray
            1D array with dimension 'frequency' containing the spectrum.

        Examples
        --------
        >>> spec = ds.radport.spectrum(l=0.0, m=0.0, time_idx=0)
        >>> spec.plot()  # Plot intensity vs frequency
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Resolve time index
        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        elif time_idx is not None:
            ti = time_idx
        else:
            ti = 0

        # Find nearest pixel
        l_idx, m_idx = self.nearest_lm_idx(l, m)

        # Extract spectrum
        spec = self._obj[var].isel(
            time=ti,
            polarization=pol,
            l=l_idx,
            m=m_idx,
        )

        # Add metadata
        time_val = float(self._obj.coords["time"].values[ti])
        l_val = float(self._obj.coords["l"].values[l_idx])
        m_val = float(self._obj.coords["m"].values[m_idx])

        spec.attrs["variable"] = var
        spec.attrs["time_idx"] = ti
        spec.attrs["time_mjd"] = time_val
        spec.attrs["pol"] = pol
        spec.attrs["l"] = l_val
        spec.attrs["m"] = m_val
        spec.attrs["l_idx"] = l_idx
        spec.attrs["m_idx"] = m_idx

        return spec

    def plot_spectrum(
        self,
        l: float,
        m: float,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        figsize: tuple[float, float] = (10, 4),
        marker: str = "o",
        linestyle: str = "-",
        freq_unit: Literal["Hz", "MHz"] = "MHz",
        **kwargs: Any,
    ) -> Figure:
        """Plot a frequency spectrum at a specific spatial location and time.

        Parameters
        ----------
        l : float
            Direction cosine l coordinate.
        m : float
            Direction cosine m coordinate.
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        time_mjd : float, optional
            Time in MJD (overrides time_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        figsize : tuple, default (10, 4)
            Figure size in inches.
        marker : str, default 'o'
            Marker style for data points.
        linestyle : str, default '-'
            Line style connecting points.
        freq_unit : {'Hz', 'MHz'}, default 'MHz'
            Unit for frequency axis.
        **kwargs : dict
            Additional arguments passed to plt.plot.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_spectrum(l=0.0, m=0.0, time_idx=0)
        """
        spec = self.spectrum(
            l=l, m=m, time_idx=time_idx, time_mjd=time_mjd, var=var, pol=pol
        )

        fig, ax = plt.subplots(figsize=figsize)

        freq_vals = spec.coords["frequency"].values
        if freq_unit == "MHz":
            freq_vals = freq_vals / 1e6
            xlabel = "Frequency (MHz)"
        else:
            xlabel = "Frequency (Hz)"

        ax.plot(freq_vals, spec.values, marker=marker, linestyle=linestyle, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{var} Intensity (Jy/beam)")

        time_mjd_val = spec.attrs["time_mjd"]
        l_val = spec.attrs["l"]
        m_val = spec.attrs["m"]
        ax.set_title(
            f"{var} Spectrum at (l={l_val:.3f}, m={m_val:.3f}), "
            f"t={time_mjd_val:.6f} MJD, pol={pol}"
        )

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig

    def time_average(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        time_indices: list[int] | None = None,
    ) -> xr.DataArray:
        """Compute the time-averaged image.

        Averages the data across the time dimension, returning a 3D array
        with dimensions (frequency, l, m).

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to average.
        pol : int, default 0
            Polarization index.
        time_indices : list of int, optional
            Specific time indices to include in the average.
            If None, averages over all times.

        Returns
        -------
        xr.DataArray
            3D array with dimensions (frequency, l, m).

        Examples
        --------
        >>> avg = ds.radport.time_average()
        >>> avg.isel(frequency=0).plot()  # Plot mean image at first frequency
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)

        if time_indices is not None:
            da = da.isel(time=time_indices)

        avg = da.mean(dim="time")

        avg.attrs["variable"] = var
        avg.attrs["pol"] = pol
        avg.attrs["operation"] = "time_average"
        if time_indices is not None:
            avg.attrs["time_indices"] = time_indices
        else:
            avg.attrs["n_times"] = self._obj.sizes["time"]

        return avg

    def frequency_average(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_indices: list[int] | None = None,
        freq_min_mhz: float | None = None,
        freq_max_mhz: float | None = None,
    ) -> xr.DataArray:
        """Compute the frequency-averaged image.

        Averages the data across the frequency dimension, returning a 3D array
        with dimensions (time, l, m).

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to average.
        pol : int, default 0
            Polarization index.
        freq_indices : list of int, optional
            Specific frequency indices to include in the average.
            If None (and freq_min/max not set), averages over all frequencies.
        freq_min_mhz : float, optional
            Minimum frequency in MHz for averaging band.
        freq_max_mhz : float, optional
            Maximum frequency in MHz for averaging band.

        Returns
        -------
        xr.DataArray
            3D array with dimensions (time, l, m).

        Examples
        --------
        >>> avg = ds.radport.frequency_average()
        >>> avg.isel(time=0).plot()  # Plot mean image at first time

        >>> # Average only 45-55 MHz band
        >>> band_avg = ds.radport.frequency_average(freq_min_mhz=45.0, freq_max_mhz=55.0)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)

        # Handle frequency selection
        if freq_min_mhz is not None or freq_max_mhz is not None:
            freq_hz = self._obj.coords["frequency"].values
            freq_mhz = freq_hz / 1e6

            if freq_min_mhz is None:
                freq_min_mhz = freq_mhz.min()
            if freq_max_mhz is None:
                freq_max_mhz = freq_mhz.max()

            mask = (freq_mhz >= freq_min_mhz) & (freq_mhz <= freq_max_mhz)
            freq_indices = list(np.where(mask)[0])

            if len(freq_indices) == 0:
                raise ValueError(
                    f"No frequencies in range [{freq_min_mhz}, {freq_max_mhz}] MHz. "
                    f"Available range: [{freq_mhz.min():.2f}, {freq_mhz.max():.2f}] MHz"
                )

        if freq_indices is not None:
            da = da.isel(frequency=freq_indices)

        avg = da.mean(dim="frequency")

        avg.attrs["variable"] = var
        avg.attrs["pol"] = pol
        avg.attrs["operation"] = "frequency_average"
        if freq_indices is not None:
            avg.attrs["freq_indices"] = freq_indices
        else:
            avg.attrs["n_frequencies"] = self._obj.sizes["frequency"]
        if freq_min_mhz is not None:
            avg.attrs["freq_min_mhz"] = freq_min_mhz
        if freq_max_mhz is not None:
            avg.attrs["freq_max_mhz"] = freq_max_mhz

        return avg

    # =========================================================================
    # Sliding Window Time-Frequency Analysis
    # =========================================================================

    def sliding_window_stacks(
        self,
        l_center: float,
        m_center: float,
        cutout_size: float,
        time_window: int,
        freq_window: int,
        time_step: int = 1,
        freq_step: int = 1,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        min_valid_fraction: float = 0.5,
    ) -> xr.Dataset:
        """Compute sliding window averages over time and frequency.

        Creates averaged image stacks using a sliding kernel that moves across
        both time and frequency dimensions. This is useful for detecting variable
        and transient radio sources that may appear/disappear across different
        time and frequency ranges.

        Parameters
        ----------
        l_center : float
            Center l coordinate of the cutout region.
        m_center : float
            Center m coordinate of the cutout region.
        cutout_size : float
            Half-width of the cutout in both l and m directions.
            The cutout spans [center - size, center + size].
        time_window : int
            Number of time steps to include in each window.
        freq_window : int
            Number of frequency channels to include in each window.
        time_step : int, default 1
            Step size for sliding the time window. Larger values
            reduce the number of output time kernels.
        freq_step : int, default 1
            Step size for sliding the frequency window. Larger values
            reduce the number of output frequency kernels.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to analyze.
        pol : int, default 0
            Polarization index.
        min_valid_fraction : float, default 0.5
            Minimum fraction of valid (non-NaN) pixels required
            for a kernel to be included. Kernels with fewer valid
            pixels are set to NaN.

        Returns
        -------
        xr.Dataset
            Dataset with the following variables:
            - stack : (kernel_time, kernel_freq, l, m) - averaged images
            - rms : (kernel_time, kernel_freq, l, m) - RMS per kernel
            - peak_flux : (kernel_time, kernel_freq) - peak flux per kernel
            - peak_l : (kernel_time, kernel_freq) - l position of peak
            - peak_m : (kernel_time, kernel_freq) - m position of peak
            - n_valid : (kernel_time, kernel_freq) - count of valid pixels

        Raises
        ------
        ValueError
            If window size exceeds data extent or cutout region is empty.

        Examples
        --------
        >>> # Analyze a region with 5-time, 3-frequency sliding windows
        >>> stacks = ds.radport.sliding_window_stacks(
        ...     l_center=0.0, m_center=0.0, cutout_size=0.1,
        ...     time_window=5, freq_window=3
        ... )
        >>> # Plot the first kernel stack
        >>> stacks.stack.isel(kernel_time=0, kernel_freq=0).plot()

        >>> # Use larger steps to reduce computation
        >>> stacks = ds.radport.sliding_window_stacks(
        ...     l_center=0.0, m_center=0.0, cutout_size=0.1,
        ...     time_window=5, freq_window=3,
        ...     time_step=2, freq_step=2
        ... )
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Get data dimensions
        n_times = self._obj.sizes["time"]
        n_freqs = self._obj.sizes["frequency"]

        # Validate window sizes
        if time_window > n_times:
            raise ValueError(
                f"time_window ({time_window}) exceeds number of time steps ({n_times})"
            )
        if freq_window > n_freqs:
            raise ValueError(
                f"freq_window ({freq_window}) exceeds number of frequency "
                f"channels ({n_freqs})"
            )

        # Extract the cutout region across all time and frequency
        da = self._obj[var].isel(polarization=pol)

        # Compute l/m bounds
        l_min, l_max = l_center - cutout_size, l_center + cutout_size
        m_min, m_max = m_center - cutout_size, m_center + cutout_size

        # Handle coordinate ordering
        l_coords = da.coords["l"]
        m_coords = da.coords["m"]

        if float(l_coords[0]) <= float(l_coords[-1]):
            l_slice = slice(l_min, l_max)
        else:
            l_slice = slice(l_max, l_min)

        if float(m_coords[0]) <= float(m_coords[-1]):
            m_slice = slice(m_min, m_max)
        else:
            m_slice = slice(m_max, m_min)

        cutout = da.sel(l=l_slice, m=m_slice)

        if cutout.size == 0:
            raise ValueError(
                f"Cutout region is empty. Requested l=[{l_min:.3f}, {l_max:.3f}], "
                f"m=[{m_min:.3f}, {m_max:.3f}]."
            )

        # Compute kernel positions
        time_starts = list(range(0, n_times - time_window + 1, time_step))
        freq_starts = list(range(0, n_freqs - freq_window + 1, freq_step))

        n_time_kernels = len(time_starts)
        n_freq_kernels = len(freq_starts)

        # Get spatial dimensions
        n_l = cutout.sizes["l"]
        n_m = cutout.sizes["m"]

        # Initialize output arrays
        stack_data = np.full((n_time_kernels, n_freq_kernels, n_l, n_m), np.nan)
        rms_data = np.full((n_time_kernels, n_freq_kernels, n_l, n_m), np.nan)
        peak_flux = np.full((n_time_kernels, n_freq_kernels), np.nan)
        peak_l = np.full((n_time_kernels, n_freq_kernels), np.nan)
        peak_m = np.full((n_time_kernels, n_freq_kernels), np.nan)
        n_valid = np.zeros((n_time_kernels, n_freq_kernels), dtype=int)

        # Compute cutout values once
        cutout_values = cutout.values  # Shape: (time, freq, l, m)

        # Slide the window
        for ti, t_start in enumerate(time_starts):
            t_end = t_start + time_window
            for fi, f_start in enumerate(freq_starts):
                f_end = f_start + freq_window

                # Extract window
                window = cutout_values[t_start:t_end, f_start:f_end, :, :]

                # Compute mean and std over time/freq dimensions
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_img = np.nanmean(window, axis=(0, 1))
                    std_img = np.nanstd(window, axis=(0, 1))

                # Count valid pixels
                valid_count = np.sum(np.isfinite(window), axis=(0, 1))
                total_window_size = time_window * freq_window
                valid_fraction = valid_count / total_window_size

                # Apply validity threshold
                invalid_mask = valid_fraction < min_valid_fraction
                mean_img[invalid_mask] = np.nan
                std_img[invalid_mask] = np.nan

                stack_data[ti, fi] = mean_img
                rms_data[ti, fi] = std_img

                # Find peak position and value
                finite_mask = np.isfinite(mean_img)
                n_valid[ti, fi] = np.sum(finite_mask)

                if np.any(finite_mask):
                    peak_idx = np.nanargmax(mean_img)
                    peak_l_idx, peak_m_idx = np.unravel_index(peak_idx, mean_img.shape)
                    peak_flux[ti, fi] = mean_img[peak_l_idx, peak_m_idx]
                    peak_l[ti, fi] = float(cutout.coords["l"].values[peak_l_idx])
                    peak_m[ti, fi] = float(cutout.coords["m"].values[peak_m_idx])

        # Compute kernel center times and frequencies
        time_vals = self._obj.coords["time"].values
        freq_vals = self._obj.coords["frequency"].values

        kernel_times = np.array([
            (time_vals[t_start] + time_vals[t_start + time_window - 1]) / 2
            for t_start in time_starts
        ])
        kernel_freqs = np.array([
            (freq_vals[f_start] + freq_vals[f_start + freq_window - 1]) / 2
            for f_start in freq_starts
        ])

        # Build output dataset
        result = xr.Dataset(
            {
                "stack": (
                    ["kernel_time", "kernel_freq", "l", "m"],
                    stack_data,
                ),
                "rms": (
                    ["kernel_time", "kernel_freq", "l", "m"],
                    rms_data,
                ),
                "peak_flux": (
                    ["kernel_time", "kernel_freq"],
                    peak_flux,
                ),
                "peak_l": (
                    ["kernel_time", "kernel_freq"],
                    peak_l,
                ),
                "peak_m": (
                    ["kernel_time", "kernel_freq"],
                    peak_m,
                ),
                "n_valid": (
                    ["kernel_time", "kernel_freq"],
                    n_valid,
                ),
            },
            coords={
                "kernel_time": kernel_times,
                "kernel_freq": kernel_freqs,
                "l": cutout.coords["l"].values,
                "m": cutout.coords["m"].values,
            },
        )

        # Add metadata
        result.attrs["variable"] = var
        result.attrs["pol"] = pol
        result.attrs["l_center"] = l_center
        result.attrs["m_center"] = m_center
        result.attrs["cutout_size"] = cutout_size
        result.attrs["time_window"] = time_window
        result.attrs["freq_window"] = freq_window
        result.attrs["time_step"] = time_step
        result.attrs["freq_step"] = freq_step
        result.attrs["min_valid_fraction"] = min_valid_fraction

        return result

    def variability_index(
        self,
        l_center: float,
        m_center: float,
        cutout_size: float,
        metric: Literal[
            "modulation_index", "chi_squared", "peak_to_mean"
        ] = "modulation_index",
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.DataArray:
        """Compute variability index for each pixel across time and frequency.

        Calculates how much each pixel's flux varies across the time and
        frequency dimensions, useful for identifying variable radio sources.

        Parameters
        ----------
        l_center : float
            Center l coordinate of the cutout region.
        m_center : float
            Center m coordinate of the cutout region.
        cutout_size : float
            Half-width of the cutout in both l and m directions.
        metric : {'modulation_index', 'chi_squared', 'peak_to_mean'}, default 'modulation_index'
            Variability metric to compute:
            - 'modulation_index': std(flux) / mean(flux), measures fractional
              variability. Values ~0 indicate steady sources, >0.3 indicates
              significant variability.
            - 'chi_squared': Sum of (flux - mean)² / variance, measures
              deviation from constant flux. Higher values indicate more
              variability.
            - 'peak_to_mean': max(flux) / mean(flux), ratio of peak to average
              flux. Values ~1 indicate steady sources, >2 indicates transients.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to analyze.
        pol : int, default 0
            Polarization index.

        Returns
        -------
        xr.DataArray
            2D array with dimensions (l, m) containing the variability index
            for each pixel. Higher values indicate more variable sources.

        Examples
        --------
        >>> # Compute modulation index for a region
        >>> var_idx = ds.radport.variability_index(
        ...     l_center=0.0, m_center=0.0, cutout_size=0.2
        ... )
        >>> var_idx.plot()

        >>> # Use chi-squared metric for detecting transients
        >>> chi2 = ds.radport.variability_index(
        ...     l_center=0.0, m_center=0.0, cutout_size=0.2,
        ...     metric='chi_squared'
        ... )
        """
        # Validate inputs
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        valid_metrics = ["modulation_index", "chi_squared", "peak_to_mean"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        # Extract cutout across all time and frequency
        da = self._obj[var].isel(polarization=pol)

        # Compute l/m bounds
        l_min, l_max = l_center - cutout_size, l_center + cutout_size
        m_min, m_max = m_center - cutout_size, m_center + cutout_size

        # Handle coordinate ordering
        l_coords = da.coords["l"]
        m_coords = da.coords["m"]

        if float(l_coords[0]) <= float(l_coords[-1]):
            l_slice = slice(l_min, l_max)
        else:
            l_slice = slice(l_max, l_min)

        if float(m_coords[0]) <= float(m_coords[-1]):
            m_slice = slice(m_min, m_max)
        else:
            m_slice = slice(m_max, m_min)

        cutout = da.sel(l=l_slice, m=m_slice)

        if cutout.size == 0:
            raise ValueError(
                f"Cutout region is empty. Requested l=[{l_min:.3f}, {l_max:.3f}], "
                f"m=[{m_min:.3f}, {m_max:.3f}]."
            )

        # Compute statistics over time and frequency
        data = cutout.values  # Shape: (time, freq, l, m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            if metric == "modulation_index":
                # std / mean
                mean_flux = np.nanmean(data, axis=(0, 1))
                std_flux = np.nanstd(data, axis=(0, 1))
                result_data = std_flux / np.abs(mean_flux)
                # Handle negative or zero mean
                result_data[mean_flux <= 0] = np.nan

            elif metric == "chi_squared":
                # Sum of (flux - mean)² / variance
                mean_flux = np.nanmean(data, axis=(0, 1), keepdims=True)
                var_flux = np.nanvar(data, axis=(0, 1), keepdims=True)
                # Avoid division by zero
                var_flux[var_flux == 0] = np.nan
                chi2 = np.nansum((data - mean_flux) ** 2 / var_flux, axis=(0, 1))
                # Normalize by degrees of freedom
                n_obs = np.sum(np.isfinite(data), axis=(0, 1))
                result_data = chi2 / np.maximum(n_obs - 1, 1)

            elif metric == "peak_to_mean":
                # max / mean
                peak_flux = np.nanmax(data, axis=(0, 1))
                mean_flux = np.nanmean(data, axis=(0, 1))
                result_data = peak_flux / np.abs(mean_flux)
                # Handle negative or zero mean
                result_data[mean_flux <= 0] = np.nan

        # Build output DataArray
        result = xr.DataArray(
            result_data,
            dims=["l", "m"],
            coords={
                "l": cutout.coords["l"].values,
                "m": cutout.coords["m"].values,
            },
            attrs={
                "variable": var,
                "pol": pol,
                "metric": metric,
                "l_center": l_center,
                "m_center": m_center,
                "cutout_size": cutout_size,
            },
        )

        return result

    def find_variable_sources(
        self,
        time_window: int,
        freq_window: int,
        snr_threshold: float = 5.0,
        variability_threshold: float = 0.3,
        grid_step: float | None = None,
        exclude_horizon: bool = True,
        max_candidates: int = 100,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.Dataset:
        """Search for variable sources across the full field of view.

        Systematically scans the image to find pixels that show both
        significant SNR and variability above specified thresholds.
        Useful for blind searches for transient and variable radio sources.

        Parameters
        ----------
        time_window : int
            Number of time steps for computing variability statistics.
        freq_window : int
            Number of frequency channels for computing variability statistics.
        snr_threshold : float, default 5.0
            Minimum signal-to-noise ratio for a candidate source.
            SNR is computed relative to local RMS.
        variability_threshold : float, default 0.3
            Minimum modulation index for a candidate to be considered
            variable. Values of 0.3 indicate 30% fractional variability.
        grid_step : float, optional
            Step size for grid search in l/m coordinates. If None, defaults
            to 1/20 of the l coordinate range.
        exclude_horizon : bool, default True
            If True, exclude pixels near the horizon (l² + m² > 0.9).
        max_candidates : int, default 100
            Maximum number of candidate sources to return, sorted by
            variability index.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to analyze.
        pol : int, default 0
            Polarization index.

        Returns
        -------
        xr.Dataset
            Dataset containing variable source candidates with:
            - l, m : (candidate,) - source positions
            - snr : (candidate,) - peak signal-to-noise ratio
            - variability : (candidate,) - modulation index
            - peak_time_idx : (candidate,) - time index of peak flux
            - peak_freq_idx : (candidate,) - frequency index of peak flux
            - peak_flux : (candidate,) - maximum flux value
            - mean_flux : (candidate,) - mean flux value
            - light_curve : (candidate, time) - flux vs time at peak freq
            If no candidates found, returns empty dataset.

        Examples
        --------
        >>> # Search for variable sources
        >>> candidates = ds.radport.find_variable_sources(
        ...     time_window=5, freq_window=3,
        ...     snr_threshold=5.0, variability_threshold=0.3
        ... )
        >>> print(f"Found {candidates.sizes['candidate']} variable sources")

        >>> # Plot light curve of most variable source
        >>> if candidates.sizes['candidate'] > 0:
        ...     candidates.light_curve.isel(candidate=0).plot()
        """
        from scipy.ndimage import maximum_filter, uniform_filter

        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Get data
        da = self._obj[var].isel(polarization=pol)
        data = da.values  # Shape: (time, freq, l, m)

        n_times = data.shape[0]
        n_freqs = data.shape[1]

        # Get coordinates
        l_vals = da.coords["l"].values
        m_vals = da.coords["m"].values

        # Default grid step
        if grid_step is None:
            l_range = float(l_vals.max() - l_vals.min())
            grid_step = l_range / 20.0

        # Compute time-averaged image for SNR calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_image = np.nanmean(data, axis=(0, 1))
            std_image = np.nanstd(data, axis=(0, 1))

        # Compute local RMS using sliding box
        box_size = max(5, int(len(l_vals) / 20))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Compute local mean and RMS
            local_mean = uniform_filter(mean_image, size=box_size, mode="constant")
            local_sq_mean = uniform_filter(
                mean_image**2, size=box_size, mode="constant"
            )
            local_rms = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
            local_rms[local_rms == 0] = np.nan

        # Compute SNR map
        snr_map = mean_image / local_rms

        # Compute variability index (modulation index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variability_map = std_image / np.abs(mean_image)
            variability_map[mean_image <= 0] = np.nan

        # Create horizon mask
        ll, mm = np.meshgrid(l_vals, m_vals, indexing="ij")
        horizon_mask = ll**2 + mm**2 > 0.9

        # Apply thresholds and masks
        candidate_mask = (
            (snr_map >= snr_threshold)
            & (variability_map >= variability_threshold)
            & np.isfinite(snr_map)
            & np.isfinite(variability_map)
        )

        if exclude_horizon:
            candidate_mask &= ~horizon_mask

        # Find local maxima in variability map to avoid duplicate detections
        # from the same source
        max_filter_size = max(3, int(len(l_vals) / 50))
        variability_maxima = variability_map == maximum_filter(
            variability_map, size=max_filter_size, mode="constant"
        )
        candidate_mask &= variability_maxima

        # Get candidate positions
        candidate_indices = np.where(candidate_mask)
        n_raw_candidates = len(candidate_indices[0])

        if n_raw_candidates == 0:
            # Return empty dataset
            return xr.Dataset(
                {
                    "l": ("candidate", np.array([])),
                    "m": ("candidate", np.array([])),
                    "snr": ("candidate", np.array([])),
                    "variability": ("candidate", np.array([])),
                    "peak_time_idx": ("candidate", np.array([], dtype=int)),
                    "peak_freq_idx": ("candidate", np.array([], dtype=int)),
                    "peak_flux": ("candidate", np.array([])),
                    "mean_flux": ("candidate", np.array([])),
                    "light_curve": (
                        ("candidate", "time"),
                        np.empty((0, n_times)),
                    ),
                },
                coords={
                    "time": self._obj.coords["time"].values,
                },
                attrs={
                    "variable": var,
                    "pol": pol,
                    "snr_threshold": snr_threshold,
                    "variability_threshold": variability_threshold,
                },
            )

        # Collect candidate data
        candidates_l = l_vals[candidate_indices[0]]
        candidates_m = m_vals[candidate_indices[1]]
        candidates_snr = snr_map[candidate_indices]
        candidates_var = variability_map[candidate_indices]
        candidates_mean = mean_image[candidate_indices]

        # Sort by variability (descending) and limit
        sort_idx = np.argsort(-candidates_var)[:max_candidates]

        candidates_l = candidates_l[sort_idx]
        candidates_m = candidates_m[sort_idx]
        candidates_snr = candidates_snr[sort_idx]
        candidates_var = candidates_var[sort_idx]
        candidates_mean = candidates_mean[sort_idx]
        candidate_l_idx = candidate_indices[0][sort_idx]
        candidate_m_idx = candidate_indices[1][sort_idx]

        n_candidates = len(candidates_l)

        # Extract light curves and find peak times/frequencies
        light_curves = np.zeros((n_candidates, n_times))
        peak_time_idx = np.zeros(n_candidates, dtype=int)
        peak_freq_idx = np.zeros(n_candidates, dtype=int)
        peak_flux = np.zeros(n_candidates)

        for i in range(n_candidates):
            li, mi = candidate_l_idx[i], candidate_m_idx[i]
            pixel_data = data[:, :, li, mi]  # Shape: (time, freq)

            # Light curve: average over frequency
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                light_curves[i] = np.nanmean(pixel_data, axis=1)

            # Find peak
            if np.any(np.isfinite(pixel_data)):
                peak_idx = np.nanargmax(pixel_data)
                peak_time_idx[i], peak_freq_idx[i] = np.unravel_index(
                    peak_idx, pixel_data.shape
                )
                peak_flux[i] = pixel_data[peak_time_idx[i], peak_freq_idx[i]]

        # Build output dataset
        result = xr.Dataset(
            {
                "l": ("candidate", candidates_l),
                "m": ("candidate", candidates_m),
                "snr": ("candidate", candidates_snr),
                "variability": ("candidate", candidates_var),
                "peak_time_idx": ("candidate", peak_time_idx),
                "peak_freq_idx": ("candidate", peak_freq_idx),
                "peak_flux": ("candidate", peak_flux),
                "mean_flux": ("candidate", candidates_mean),
                "light_curve": (("candidate", "time"), light_curves),
            },
            coords={
                "time": self._obj.coords["time"].values,
            },
            attrs={
                "variable": var,
                "pol": pol,
                "snr_threshold": snr_threshold,
                "variability_threshold": variability_threshold,
                "time_window": time_window,
                "freq_window": freq_window,
                "exclude_horizon": exclude_horizon,
            },
        )

        return result

    def plot_time_average(
        self,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        time_indices: list[int] | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot the time-averaged image at a specific frequency.

        Parameters
        ----------
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        time_indices : list of int, optional
            Specific time indices to include in the average.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use 2nd/98th percentile for scaling.
        mask_radius : int, optional
            Circular mask radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_time_average(freq_mhz=50.0)
        """
        avg = self.time_average(var=var, pol=pol, time_indices=time_indices)

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Select frequency slice
        data = avg.isel(frequency=fi).values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[dist > mask_radius] = np.nan

        # Compute vmin/vmax if robust
        if robust and vmin is None and vmax is None:
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 98))

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        l_vals = avg.coords["l"].values
        m_vals = avg.coords["m"].values
        extent = [
            float(l_vals.min()), float(l_vals.max()),
            float(m_vals.min()), float(m_vals.max()),
        ]

        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        freq_hz = float(self._obj.coords["frequency"].values[fi])
        n_times = len(time_indices) if time_indices else self._obj.sizes["time"]
        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(
            f"{var} Time Average ({n_times} frames) at f={freq_hz/1e6:.2f} MHz, pol={pol}"
        )

        fig.tight_layout()
        return fig

    def plot_frequency_average(
        self,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_indices: list[int] | None = None,
        freq_min_mhz: float | None = None,
        freq_max_mhz: float | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot the frequency-averaged image at a specific time.

        Parameters
        ----------
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        time_mjd : float, optional
            Time in MJD (overrides time_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        freq_indices : list of int, optional
            Specific frequency indices to include in the average.
        freq_min_mhz : float, optional
            Minimum frequency in MHz for averaging band.
        freq_max_mhz : float, optional
            Maximum frequency in MHz for averaging band.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use 2nd/98th percentile for scaling.
        mask_radius : int, optional
            Circular mask radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_frequency_average(time_idx=0)

        >>> # Average 45-55 MHz band
        >>> fig = ds.radport.plot_frequency_average(
        ...     time_idx=0, freq_min_mhz=45.0, freq_max_mhz=55.0
        ... )
        """
        avg = self.frequency_average(
            var=var,
            pol=pol,
            freq_indices=freq_indices,
            freq_min_mhz=freq_min_mhz,
            freq_max_mhz=freq_max_mhz,
        )

        # Resolve time index
        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        elif time_idx is not None:
            ti = time_idx
        else:
            ti = 0

        # Select time slice
        data = avg.isel(time=ti).values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[dist > mask_radius] = np.nan

        # Compute vmin/vmax if robust
        if robust and vmin is None and vmax is None:
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 98))

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        l_vals = avg.coords["l"].values
        m_vals = avg.coords["m"].values
        extent = [
            float(l_vals.min()), float(l_vals.max()),
            float(m_vals.min()), float(m_vals.max()),
        ]

        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        time_val = float(self._obj.coords["time"].values[ti])

        # Build title with frequency info
        if freq_min_mhz is not None and freq_max_mhz is not None:
            freq_info = f"{freq_min_mhz:.1f}-{freq_max_mhz:.1f} MHz"
        elif freq_indices is not None:
            freq_info = f"{len(freq_indices)} channels"
        else:
            freq_info = f"{self._obj.sizes['frequency']} channels"

        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(
            f"{var} Frequency Average ({freq_info}) at t={time_val:.6f} MJD, pol={pol}"
        )

        fig.tight_layout()
        return fig

    # =========================================================================
    # WCS & Coordinate Methods
    # =========================================================================

    def _get_wcs(self, var: Literal["SKY", "BEAM"] = "SKY"):
        """Get WCS object from the dataset.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to get WCS from (checks attrs first).

        Returns
        -------
        astropy.wcs.WCS
            The WCS object for coordinate transformations.

        Raises
        ------
        ImportError
            If astropy is not installed.
        ValueError
            If no WCS header is found in the dataset.
        """
        try:
            from astropy.io.fits import Header
            from astropy.wcs import WCS
        except ImportError as e:
            raise ImportError(
                "astropy is required for WCS functionality. "
                "Install with: pip install astropy"
            ) from e

        # Try to get WCS header string from various locations
        hdr_str = None

        # 1. Check variable attrs
        if var in self._obj.data_vars:
            hdr_str = self._obj[var].attrs.get("fits_wcs_header")

        # 2. Check dataset attrs
        if not hdr_str:
            hdr_str = self._obj.attrs.get("fits_wcs_header")

        # 3. Check wcs_header_str variable
        if not hdr_str and "wcs_header_str" in self._obj:
            val = self._obj["wcs_header_str"].values
            if isinstance(val, np.ndarray):
                val = val.item()
            if isinstance(val, (bytes, bytearray)) or type(val).__name__ == "bytes_":
                hdr_str = val.decode("utf-8", errors="replace")
            else:
                hdr_str = str(val)

        if not hdr_str:
            raise ValueError(
                "No WCS header found in dataset. Expected 'fits_wcs_header' "
                "attribute on variable/dataset or 'wcs_header_str' variable."
            )

        return WCS(Header.fromstring(hdr_str, sep="\n"))

    @property
    def has_wcs(self) -> bool:
        """Check if WCS coordinate information is available.

        Returns
        -------
        bool
            True if WCS header is available in the dataset.

        Example
        -------
        >>> if ds.radport.has_wcs:
        ...     fig = ds.radport.plot_wcs()
        """
        try:
            self._get_wcs()
            return True
        except (ImportError, ValueError):
            return False

    def pixel_to_coords(
        self,
        l_idx: int,
        m_idx: int,
    ) -> tuple[float, float]:
        """Convert pixel indices to celestial coordinates (RA, Dec).

        Parameters
        ----------
        l_idx : int
            Index along the l dimension.
        m_idx : int
            Index along the m dimension.

        Returns
        -------
        tuple of float
            (ra, dec) in degrees. RA is in range [0, 360).

        Raises
        ------
        ValueError
            If WCS is not available or indices are out of bounds.

        Example
        -------
        >>> ra, dec = ds.radport.pixel_to_coords(100, 100)
        >>> print(f"RA={ra:.2f}°, Dec={dec:.2f}°")
        """
        wcs = self._get_wcs()

        # Validate indices
        n_l = self._obj.sizes["l"]
        n_m = self._obj.sizes["m"]
        if not (0 <= l_idx < n_l):
            raise ValueError(f"l_idx={l_idx} out of bounds [0, {n_l})")
        if not (0 <= m_idx < n_m):
            raise ValueError(f"m_idx={m_idx} out of bounds [0, {n_m})")

        # WCS pixel_to_world expects (x, y) which is (l, m) in our convention
        coord = wcs.pixel_to_world(l_idx, m_idx)
        ra = float(coord.ra.wrap_at("360d").deg)
        dec = float(coord.dec.deg)

        return ra, dec

    def coords_to_pixel(
        self,
        ra: float,
        dec: float,
    ) -> tuple[int, int]:
        """Convert celestial coordinates (RA, Dec) to pixel indices.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.

        Returns
        -------
        tuple of int
            (l_idx, m_idx) pixel indices (rounded to nearest integer).

        Raises
        ------
        ValueError
            If WCS is not available or coordinates are outside the image.

        Example
        -------
        >>> l_idx, m_idx = ds.radport.coords_to_pixel(180.0, 45.0)
        """
        try:
            from astropy.coordinates import SkyCoord
            from astropy import units as u
        except ImportError as e:
            raise ImportError(
                "astropy is required for coordinate transformations."
            ) from e

        wcs = self._get_wcs()

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="fk5")
        x, y = wcs.world_to_pixel(coord)

        l_idx = int(round(float(x)))
        m_idx = int(round(float(y)))

        # Validate result is within bounds
        n_l = self._obj.sizes["l"]
        n_m = self._obj.sizes["m"]
        if not (0 <= l_idx < n_l) or not (0 <= m_idx < n_m):
            raise ValueError(
                f"Coordinates (RA={ra}, Dec={dec}) map to pixel ({l_idx}, {m_idx}) "
                f"which is outside image bounds [0, {n_l}) x [0, {n_m})"
            )

        return l_idx, m_idx

    def plot_wcs(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int = 0,
        freq_idx: int = 0,
        freq_mhz: float | None = None,
        pol: int = 0,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (10, 10),
        add_colorbar: bool = True,
        grid_color: str = "white",
        grid_alpha: float = 0.6,
        grid_linestyle: str = ":",
        label_color: str = "white",
        facecolor: str = "black",
        **kwargs: Any,
    ) -> Figure:
        """Plot with WCS projection and celestial coordinate grid.

        Creates a publication-quality plot with RA/Dec coordinate axes
        and optional grid overlay.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        time_idx : int, default 0
            Time index.
        freq_idx : int, default 0
            Frequency index. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        pol : int, default 0
            Polarization index.
        cmap : str, default 'inferno'
            Colormap name.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use 2nd/98th percentile for scaling.
        mask_radius : int, optional
            Circular mask radius in pixels.
        figsize : tuple, default (10, 10)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add colorbar.
        grid_color : str, default 'white'
            Color of coordinate grid lines.
        grid_alpha : float, default 0.6
            Transparency of grid lines.
        grid_linestyle : str, default ':'
            Line style for grid.
        label_color : str, default 'white'
            Color for axis labels and ticks.
        facecolor : str, default 'black'
            Background color for the plot.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If WCS is not available in the dataset.

        Example
        -------
        >>> fig = ds.radport.plot_wcs(freq_mhz=50.0, mask_radius=1800)
        """
        try:
            from astropy import units as u
        except ImportError as e:
            raise ImportError(
                "astropy is required for WCS plotting."
            ) from e

        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        wcs = self._get_wcs(var)

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        else:
            fi = freq_idx

        # Extract data
        da = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        )

        # Ensure proper dimension order (m, l) for imshow
        if set(da.dims) == {"m", "l"}:
            da = da.transpose("m", "l")

        data = da.values.astype(float).copy()

        # Apply mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[dist > mask_radius] = np.nan

        # Compute vmin/vmax if robust
        if robust and vmin is None and vmax is None:
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 98))

        # Set up colormap with bad values as black
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(facecolor, 1.0)

        # Create figure with WCS projection
        fig = plt.figure(figsize=figsize, facecolor=facecolor)
        ax = fig.add_subplot(111, projection=wcs, facecolor=facecolor)

        # Plot image
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        # Configure axes
        ax.set_xlabel("RA", color=label_color, fontsize=12)
        ax.set_ylabel("Dec", color=label_color, fontsize=12)

        # Check if RA needs to be inverted (increases to left in sky)
        try:
            cdelt1 = float(wcs.wcs.cdelt[0])
            if np.isfinite(cdelt1) and cdelt1 > 0:
                ax.invert_xaxis()
        except (AttributeError, IndexError):
            pass

        # Add coordinate grid
        overlay = ax.get_coords_overlay("fk5")
        overlay.grid(color=grid_color, ls=grid_linestyle, lw=1.0, alpha=grid_alpha)

        # Configure tick labels
        for coord in overlay:
            coord.set_ticklabel_visible(True)
            coord.set_ticklabel(color=label_color, size=10)
            coord.tick_params(width=1, color=label_color)

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Jy/beam", color=label_color, fontsize=11)
            cbar.ax.tick_params(color=label_color, labelcolor=label_color)
            cbar.outline.set_edgecolor(label_color)

        # Add title
        freq_hz = float(self._obj.coords["frequency"].values[fi])
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        ax.set_title(
            f"{var} at t={time_str} MJD, f={freq_hz/1e6:.2f} MHz, pol={pol}",
            color=label_color,
            fontsize=12,
            pad=10,
        )

        return fig

    # =========================================================================
    # Phase F: Animation & Export Methods
    # =========================================================================

    def animate_time(
        self,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        output_file: str | None = None,
        fps: int = 5,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 100,
        **kwargs: Any,
    ) -> Any:
        """Create an animation showing time evolution at a fixed frequency.

        Parameters
        ----------
        freq_idx : int, optional
            Frequency index to animate. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to animate ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        output_file : str, optional
            Path to save the animation. Supported formats: .mp4, .gif.
            If None, returns the animation object for display in notebooks.
        fps : int, default 5
            Frames per second for the animation.
        cmap : str, default "inferno"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for color scaling. If None and robust=True,
            uses 2nd percentile across all frames.
        vmax : float, optional
            Maximum value for color scaling. If None and robust=True,
            uses 98th percentile across all frames.
        robust : bool, default True
            Use percentile-based color scaling across all frames.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        dpi : int, default 100
            Resolution for saved animation.
        **kwargs
            Additional arguments passed to FuncAnimation.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object. Can be displayed in notebooks with HTML(anim.to_jshtml())
            or saved to file.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Create animation and save to file
        >>> anim = ds.radport.animate_time(freq_mhz=50.0, output_file="time_evolution.mp4")
        >>>
        >>> # Display in Jupyter notebook
        >>> from IPython.display import HTML
        >>> anim = ds.radport.animate_time(freq_mhz=50.0)
        >>> HTML(anim.to_jshtml())
        """
        from matplotlib.animation import FuncAnimation

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get data for all time steps
        data = self._obj[var].isel(frequency=fi, polarization=pol)
        n_times = len(self._obj.coords["time"])

        # Compute global color scale from all frames
        if vmin is None or vmax is None:
            all_values = data.values.ravel()
            finite_values = all_values[np.isfinite(all_values)]
            if len(finite_values) > 0:
                if robust:
                    computed_vmin = np.percentile(finite_values, 2)
                    computed_vmax = np.percentile(finite_values, 98)
                else:
                    computed_vmin = np.nanmin(finite_values)
                    computed_vmax = np.nanmax(finite_values)
            else:
                computed_vmin, computed_vmax = 0, 1

            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Create mask if requested
        mask = None
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)

        frame_data = data.isel(time=0).values.copy()
        if mask is not None:
            frame_data[mask] = np.nan

        im = ax.imshow(
            frame_data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Jy/beam", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        freq_hz = float(self._obj.coords["frequency"].values[fi])

        def update(frame: int) -> tuple:
            """Update function for animation."""
            frame_data = data.isel(time=frame).values.copy()
            if mask is not None:
                frame_data[mask] = np.nan
            im.set_array(frame_data.T)

            time_val = self._obj.coords["time"].values[frame]
            try:
                time_str = f"{float(time_val):.6f}"
            except (TypeError, ValueError):
                time_str = str(time_val)

            ax.set_title(
                f"{var} at f={freq_hz/1e6:.2f} MHz, pol={pol}\n"
                f"Time: {time_str} MJD (frame {frame + 1}/{n_times})",
                fontsize=11,
            )
            return (im,)

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=n_times,
            interval=1000 // fps,
            blit=True,
            **kwargs,
        )

        # Save if output file specified
        if output_file is not None:
            if output_file.endswith(".gif"):
                anim.save(output_file, writer="pillow", fps=fps, dpi=dpi)
            else:
                anim.save(output_file, writer="ffmpeg", fps=fps, dpi=dpi)
            plt.close(fig)

        return anim

    def animate_frequency(
        self,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        output_file: str | None = None,
        fps: int = 5,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 100,
        **kwargs: Any,
    ) -> Any:
        """Create an animation showing frequency sweep at a fixed time.

        Parameters
        ----------
        time_idx : int, optional
            Time index to animate. Defaults to 0 if neither time_idx
            nor time_mjd is provided.
        time_mjd : float, optional
            Select time by MJD value. Overrides time_idx if provided.
        var : str, default "SKY"
            Data variable to animate ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        output_file : str, optional
            Path to save the animation. Supported formats: .mp4, .gif.
            If None, returns the animation object for display in notebooks.
        fps : int, default 5
            Frames per second for the animation.
        cmap : str, default "inferno"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for color scaling. If None and robust=True,
            uses 2nd percentile across all frames.
        vmax : float, optional
            Maximum value for color scaling. If None and robust=True,
            uses 98th percentile across all frames.
        robust : bool, default True
            Use percentile-based color scaling across all frames.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        dpi : int, default 100
            Resolution for saved animation.
        **kwargs
            Additional arguments passed to FuncAnimation.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object. Can be displayed in notebooks with HTML(anim.to_jshtml())
            or saved to file.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Create animation and save to file
        >>> anim = ds.radport.animate_frequency(time_idx=0, output_file="freq_sweep.gif")
        >>>
        >>> # Display in Jupyter notebook
        >>> from IPython.display import HTML
        >>> anim = ds.radport.animate_frequency(time_idx=0)
        >>> HTML(anim.to_jshtml())
        """
        from matplotlib.animation import FuncAnimation

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve time index
        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        elif time_idx is not None:
            ti = time_idx
        else:
            ti = 0

        # Get data for all frequencies
        data = self._obj[var].isel(time=ti, polarization=pol)
        n_freqs = len(self._obj.coords["frequency"])
        freqs_hz = self._obj.coords["frequency"].values

        # Compute global color scale from all frames
        if vmin is None or vmax is None:
            all_values = data.values.ravel()
            finite_values = all_values[np.isfinite(all_values)]
            if len(finite_values) > 0:
                if robust:
                    computed_vmin = np.percentile(finite_values, 2)
                    computed_vmax = np.percentile(finite_values, 98)
                else:
                    computed_vmin = np.nanmin(finite_values)
                    computed_vmax = np.nanmax(finite_values)
            else:
                computed_vmin, computed_vmax = 0, 1

            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Create mask if requested
        mask = None
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)

        frame_data = data.isel(frequency=0).values.copy()
        if mask is not None:
            frame_data[mask] = np.nan

        im = ax.imshow(
            frame_data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Jy/beam", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        time_val = self._obj.coords["time"].values[ti]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        def update(frame: int) -> tuple:
            """Update function for animation."""
            frame_data = data.isel(frequency=frame).values.copy()
            if mask is not None:
                frame_data[mask] = np.nan
            im.set_array(frame_data.T)

            freq_hz = float(freqs_hz[frame])
            ax.set_title(
                f"{var} at t={time_str} MJD, pol={pol}\n"
                f"Frequency: {freq_hz/1e6:.2f} MHz (channel {frame + 1}/{n_freqs})",
                fontsize=11,
            )
            return (im,)

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=n_freqs,
            interval=1000 // fps,
            blit=True,
            **kwargs,
        )

        # Save if output file specified
        if output_file is not None:
            if output_file.endswith(".gif"):
                anim.save(output_file, writer="pillow", fps=fps, dpi=dpi)
            else:
                anim.save(output_file, writer="ffmpeg", fps=fps, dpi=dpi)
            plt.close(fig)

        return anim

    def animate_sliding_window(
        self,
        stacks: xr.Dataset,
        output_file: str | None = None,
        dimension: Literal["time", "frequency"] = "time",
        fps: int = 5,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 100,
        **kwargs: Any,
    ) -> Any:
        """Create an animation from sliding window stacks.

        Animates through the kernel positions along either the time or
        frequency dimension, showing how the averaged image changes as
        the kernel slides through the data.

        Parameters
        ----------
        stacks : xr.Dataset
            Dataset returned by `sliding_window_stacks()`.
        output_file : str, optional
            Path to save the animation. Supported formats: .mp4, .gif.
            If None, returns the animation object for display in notebooks.
        dimension : {'time', 'frequency'}, default 'time'
            Dimension to animate along:
            - 'time': Animate through kernel_time, fixing kernel_freq to first
            - 'frequency': Animate through kernel_freq, fixing kernel_time to first
        fps : int, default 5
            Frames per second for the animation.
        cmap : str, default 'inferno'
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for color scaling. If None and robust=True,
            uses 2nd percentile across all frames.
        vmax : float, optional
            Maximum value for color scaling. If None and robust=True,
            uses 98th percentile across all frames.
        robust : bool, default True
            Use percentile-based color scaling across all frames.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        dpi : int, default 100
            Resolution for saved animation.
        **kwargs
            Additional arguments passed to FuncAnimation.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object. Can be displayed in notebooks with
            HTML(anim.to_jshtml()) or saved to file.

        Raises
        ------
        ValueError
            If the stacks dataset doesn't have required variables.

        Examples
        --------
        >>> # Create sliding window stacks
        >>> stacks = ds.radport.sliding_window_stacks(
        ...     l_center=0.0, m_center=0.0, cutout_size=0.1,
        ...     time_window=5, freq_window=3
        ... )
        >>> # Animate through time
        >>> anim = ds.radport.animate_sliding_window(stacks, dimension='time')
        >>> # Display in notebook
        >>> from IPython.display import HTML
        >>> HTML(anim.to_jshtml())

        >>> # Save animation to file
        >>> anim = ds.radport.animate_sliding_window(
        ...     stacks, output_file='sliding_window.mp4', fps=10
        ... )
        """
        from matplotlib.animation import FuncAnimation

        # Validate input
        if "stack" not in stacks:
            raise ValueError(
                "stacks dataset must contain 'stack' variable. "
                "Use sliding_window_stacks() to create the input."
            )

        if dimension not in ["time", "frequency"]:
            raise ValueError(f"dimension must be 'time' or 'frequency', got '{dimension}'")

        # Get stack data
        stack_data = stacks["stack"]

        if dimension == "time":
            n_frames = stack_data.sizes["kernel_time"]
            fixed_idx = 0
            frame_dim = "kernel_time"
            fixed_dim = "kernel_freq"
            kernel_values = stacks.coords["kernel_time"].values
            fixed_value = float(stacks.coords["kernel_freq"].values[fixed_idx])
        else:
            n_frames = stack_data.sizes["kernel_freq"]
            fixed_idx = 0
            frame_dim = "kernel_freq"
            fixed_dim = "kernel_time"
            kernel_values = stacks.coords["kernel_freq"].values
            fixed_value = float(stacks.coords["kernel_time"].values[fixed_idx])

        # Get data to animate
        if dimension == "time":
            data = stack_data.isel(kernel_freq=fixed_idx)  # Shape: (kernel_time, l, m)
        else:
            data = stack_data.isel(kernel_time=fixed_idx)  # Shape: (kernel_freq, l, m)

        # Compute global color scale from all frames
        if vmin is None or vmax is None:
            all_values = data.values.ravel()
            finite_values = all_values[np.isfinite(all_values)]
            if len(finite_values) > 0:
                if robust:
                    computed_vmin = np.percentile(finite_values, 2)
                    computed_vmax = np.percentile(finite_values, 98)
                else:
                    computed_vmin = np.nanmin(finite_values)
                    computed_vmax = np.nanmax(finite_values)
            else:
                computed_vmin, computed_vmax = 0, 1

            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)

        # Get coordinate extents
        l_vals = stacks.coords["l"].values
        m_vals = stacks.coords["m"].values
        extent = [
            float(l_vals.min()), float(l_vals.max()),
            float(m_vals.min()), float(m_vals.max()),
        ]

        # Initial frame
        if dimension == "time":
            frame_data = data.isel(kernel_time=0).values
        else:
            frame_data = data.isel(kernel_freq=0).values

        im = ax.imshow(
            frame_data,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Jy/beam", fontsize=11)

        # Labels
        ax.set_xlabel("l (direction cosine)", fontsize=11)
        ax.set_ylabel("m (direction cosine)", fontsize=11)

        # Get metadata
        var = stacks.attrs.get("variable", "SKY")
        time_window = stacks.attrs.get("time_window", "?")
        freq_window = stacks.attrs.get("freq_window", "?")

        def update(frame: int) -> tuple:
            """Update function for animation."""
            if dimension == "time":
                frame_data = data.isel(kernel_time=frame).values
                kernel_val = kernel_values[frame]
                title = (
                    f"{var} Sliding Window Stack\n"
                    f"Kernel time: {kernel_val:.6f} MJD (frame {frame + 1}/{n_frames})\n"
                    f"Window: {time_window}t × {freq_window}f, fixed freq={fixed_value/1e6:.2f} MHz"
                )
            else:
                frame_data = data.isel(kernel_freq=frame).values
                kernel_val = kernel_values[frame]
                title = (
                    f"{var} Sliding Window Stack\n"
                    f"Kernel freq: {kernel_val/1e6:.2f} MHz (frame {frame + 1}/{n_frames})\n"
                    f"Window: {time_window}t × {freq_window}f, fixed time={fixed_value:.6f} MJD"
                )

            im.set_array(frame_data)
            ax.set_title(title, fontsize=10)
            return (im,)

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000 // fps,
            blit=True,
            **kwargs,
        )

        # Save if output file specified
        if output_file is not None:
            if output_file.endswith(".gif"):
                anim.save(output_file, writer="pillow", fps=fps, dpi=dpi)
            else:
                anim.save(output_file, writer="ffmpeg", fps=fps, dpi=dpi)
            plt.close(fig)

        return anim

    def export_frames(
        self,
        output_dir: str,
        var: str = "SKY",
        pol: int = 0,
        time_indices: list[int] | None = None,
        freq_indices: list[int] | None = None,
        format: str = "png",
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 150,
        filename_template: str = "{var}_t{time_idx:04d}_f{freq_idx:04d}.{format}",
    ) -> list[str]:
        """Export all (time, freq) frames as individual image files.

        Parameters
        ----------
        output_dir : str
            Directory to save the image files. Will be created if it doesn't exist.
        var : str, default "SKY"
            Data variable to export ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        time_indices : list of int, optional
            Time indices to export. If None, exports all times.
        freq_indices : list of int, optional
            Frequency indices to export. If None, exports all frequencies.
        format : str, default "png"
            Image format (e.g., "png", "jpg", "pdf").
        cmap : str, default "inferno"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for color scaling. If None and robust=True,
            uses 2nd percentile across all exported frames.
        vmax : float, optional
            Maximum value for color scaling. If None and robust=True,
            uses 98th percentile across all exported frames.
        robust : bool, default True
            Use percentile-based color scaling across all exported frames.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        dpi : int, default 150
            Resolution for saved images.
        filename_template : str, default "{var}_t{time_idx:04d}_f{freq_idx:04d}.{format}"
            Template for filenames. Available placeholders: {var}, {time_idx},
            {freq_idx}, {time_mjd}, {freq_mhz}, {format}.

        Returns
        -------
        list of str
            List of paths to the saved image files.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Export all frames
        >>> files = ds.radport.export_frames("./frames")
        >>> print(f"Exported {len(files)} frames")
        >>>
        >>> # Export specific time/frequency combinations
        >>> files = ds.radport.export_frames(
        ...     "./frames",
        ...     time_indices=[0, 1, 2],
        ...     freq_indices=[0, 5, 10],
        ... )
        """
        import os

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get indices to export
        if time_indices is None:
            time_indices = list(range(len(self._obj.coords["time"])))
        if freq_indices is None:
            freq_indices = list(range(len(self._obj.coords["frequency"])))

        # Get coordinate values for labels
        time_values = self._obj.coords["time"].values
        freq_values = self._obj.coords["frequency"].values

        # Compute global color scale from all frames to export
        if vmin is None or vmax is None:
            all_values = []
            for ti in time_indices:
                for fi in freq_indices:
                    frame_data = self._obj[var].isel(
                        time=ti, frequency=fi, polarization=pol
                    ).values
                    all_values.extend(frame_data.ravel())

            all_values = np.array(all_values)
            finite_values = all_values[np.isfinite(all_values)]
            if len(finite_values) > 0:
                if robust:
                    computed_vmin = np.percentile(finite_values, 2)
                    computed_vmax = np.percentile(finite_values, 98)
                else:
                    computed_vmin = np.nanmin(finite_values)
                    computed_vmax = np.nanmax(finite_values)
            else:
                computed_vmin, computed_vmax = 0, 1

            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Create mask if requested
        mask = None
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius

        # Export frames
        exported_files = []

        for ti in time_indices:
            for fi in freq_indices:
                # Get frame data
                frame_data = self._obj[var].isel(
                    time=ti, frequency=fi, polarization=pol
                ).values.copy()

                if mask is not None:
                    frame_data[mask] = np.nan

                # Create figure
                fig, ax = plt.subplots(figsize=figsize)

                im = ax.imshow(
                    frame_data.T,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Jy/beam", fontsize=11)

                # Labels
                ax.set_xlabel("l index", fontsize=11)
                ax.set_ylabel("m index", fontsize=11)

                # Title
                time_val = time_values[ti]
                freq_hz = float(freq_values[fi])
                try:
                    time_str = f"{float(time_val):.6f}"
                except (TypeError, ValueError):
                    time_str = str(time_val)

                ax.set_title(
                    f"{var} at t={time_str} MJD, f={freq_hz/1e6:.2f} MHz, pol={pol}",
                    fontsize=11,
                )

                # Generate filename
                try:
                    time_mjd = float(time_val)
                except (TypeError, ValueError):
                    time_mjd = 0.0

                filename = filename_template.format(
                    var=var,
                    time_idx=ti,
                    freq_idx=fi,
                    time_mjd=time_mjd,
                    freq_mhz=freq_hz / 1e6,
                    format=format,
                )
                filepath = os.path.join(output_dir, filename)

                # Save figure
                fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
                plt.close(fig)

                exported_files.append(filepath)

        return exported_files

    # =========================================================================
    # Phase G: Source Detection Methods
    # =========================================================================

    def rms_map(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        box_size: int = 50,
    ) -> xr.DataArray:
        """Compute local RMS noise estimate map using a sliding box.

        The RMS is computed using a uniform filter approach where each pixel's
        RMS is estimated from the surrounding box_size x box_size region.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.

        Returns
        -------
        xr.DataArray
            2D array of local RMS values with dimensions (l, m).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> rms = ds.radport.rms_map(freq_mhz=50.0, box_size=100)
        >>> rms.plot()
        """
        from scipy.ndimage import uniform_filter

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get frame data
        data = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        ).values.astype(float)

        # Replace NaN with 0 for filtering (we'll handle NaN regions later)
        nan_mask = ~np.isfinite(data)
        data_filled = np.where(nan_mask, 0.0, data)

        # Compute local mean and mean of squares
        local_mean = uniform_filter(data_filled, size=box_size, mode="constant")
        local_mean_sq = uniform_filter(data_filled**2, size=box_size, mode="constant")

        # Count valid pixels in each box
        valid_count = uniform_filter(
            (~nan_mask).astype(float), size=box_size, mode="constant"
        )
        valid_count = np.maximum(valid_count, 1e-10)  # Avoid division by zero

        # Correct for the fact that we filled NaN with 0
        local_mean = local_mean / valid_count * (box_size**2)
        local_mean_sq = local_mean_sq / valid_count * (box_size**2)

        # Compute variance: E[X^2] - E[X]^2
        local_var = local_mean_sq - local_mean**2
        local_var = np.maximum(local_var, 0.0)  # Ensure non-negative

        # RMS is sqrt of variance
        rms = np.sqrt(local_var)

        # Restore NaN where original was NaN
        rms[nan_mask] = np.nan

        # Create DataArray with coordinates
        return xr.DataArray(
            rms,
            dims=["l", "m"],
            coords={
                "l": self._obj.coords["l"],
                "m": self._obj.coords["m"],
            },
            name="rms",
            attrs={
                "long_name": "Local RMS noise estimate",
                "units": "Jy/beam",
                "box_size": box_size,
            },
        )

    def snr_map(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        box_size: int = 50,
    ) -> xr.DataArray:
        """Compute signal-to-noise ratio map.

        The SNR is computed as the signal divided by the local RMS noise
        estimate from a sliding box.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.

        Returns
        -------
        xr.DataArray
            2D array of SNR values with dimensions (l, m).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> snr = ds.radport.snr_map(freq_mhz=50.0)
        >>> # Find pixels with SNR > 5
        >>> significant = snr.where(snr > 5)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get signal
        signal = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        ).values.astype(float)

        # Get RMS map
        rms = self.rms_map(
            time_idx=time_idx,
            freq_idx=fi,
            var=var,
            pol=pol,
            box_size=box_size,
        ).values

        # Compute SNR (avoiding division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = signal / rms
            snr[~np.isfinite(snr)] = np.nan

        # Create DataArray with coordinates
        return xr.DataArray(
            snr,
            dims=["l", "m"],
            coords={
                "l": self._obj.coords["l"],
                "m": self._obj.coords["m"],
            },
            name="snr",
            attrs={
                "long_name": "Signal-to-noise ratio",
                "units": "",
                "box_size": box_size,
            },
        )

    def find_peaks(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        threshold_sigma: float = 5.0,
        box_size: int = 50,
        min_separation: int = 5,
    ) -> list[dict]:
        """Find peaks above threshold in the image.

        Identifies local maxima that exceed the specified SNR threshold.
        Uses local maximum detection with minimum separation between peaks.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        threshold_sigma : float, default 5.0
            Minimum SNR threshold for peak detection.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.
        min_separation : int, default 5
            Minimum separation between peaks in pixels.

        Returns
        -------
        list of dict
            List of detected peaks, each with keys:
            - l: l coordinate value
            - m: m coordinate value
            - l_idx: l pixel index
            - m_idx: m pixel index
            - flux: peak flux value (Jy/beam)
            - snr: signal-to-noise ratio

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> peaks = ds.radport.find_peaks(freq_mhz=50.0, threshold_sigma=5.0)
        >>> print(f"Found {len(peaks)} peaks")
        >>> for p in peaks[:5]:
        ...     print(f"  l={p['l']:.3f}, m={p['m']:.3f}, flux={p['flux']:.2f}, SNR={p['snr']:.1f}")
        """
        from scipy.ndimage import maximum_filter

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get signal and SNR maps
        signal = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        ).values.astype(float)

        snr = self.snr_map(
            time_idx=time_idx,
            freq_idx=fi,
            var=var,
            pol=pol,
            box_size=box_size,
        ).values

        # Find local maxima using maximum filter
        # A pixel is a local max if it equals the max in its neighborhood
        local_max = maximum_filter(signal, size=min_separation * 2 + 1)
        is_local_max = (signal == local_max) & np.isfinite(signal)

        # Apply SNR threshold
        is_peak = is_local_max & (snr >= threshold_sigma)

        # Get peak locations
        l_indices, m_indices = np.where(is_peak)

        # Get coordinate values
        l_coords = self._obj.coords["l"].values
        m_coords = self._obj.coords["m"].values

        # Build list of peaks sorted by SNR (descending)
        peaks = []
        for l_idx, m_idx in zip(l_indices, m_indices):
            peaks.append(
                {
                    "l": float(l_coords[l_idx]),
                    "m": float(m_coords[m_idx]),
                    "l_idx": int(l_idx),
                    "m_idx": int(m_idx),
                    "flux": float(signal[l_idx, m_idx]),
                    "snr": float(snr[l_idx, m_idx]),
                }
            )

        # Sort by SNR descending
        peaks.sort(key=lambda p: p["snr"], reverse=True)

        return peaks

    def peak_flux_map(
        self,
        var: str = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
    ) -> xr.DataArray:
        """Compute peak flux at each pixel across all times.

        For each (l, m) pixel, finds the maximum flux value across
        all time steps at the specified frequency.

        Parameters
        ----------
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Frequency index. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.

        Returns
        -------
        xr.DataArray
            2D array of peak flux values with dimensions (l, m).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Find brightest emission at each pixel across all times
        >>> peak_map = ds.radport.peak_flux_map(freq_mhz=50.0)
        >>> peak_map.plot()
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get data for all times at this frequency
        data = self._obj[var].isel(frequency=fi, polarization=pol)

        # Compute max across time dimension
        peak_flux = data.max(dim="time", skipna=True)

        # Update attributes
        peak_flux.name = "peak_flux"
        peak_flux.attrs = {
            "long_name": "Peak flux across time",
            "units": "Jy/beam",
        }

        return peak_flux

    def plot_snr_map(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        box_size: int = 50,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        symmetric: bool = True,
    ) -> "Figure":
        """Plot the signal-to-noise ratio map.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        var : str, default "SKY"
            Data variable to analyze.
        pol : int, default 0
            Polarization index.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.
        cmap : str, default "RdBu_r"
            Colormap (diverging recommended for SNR).
        vmin : float, optional
            Minimum value for color scaling.
        vmax : float, optional
            Maximum value for color scaling.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        symmetric : bool, default True
            Use symmetric color scale centered at zero.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Example
        -------
        >>> fig = ds.radport.plot_snr_map(freq_mhz=50.0, mask_radius=1800)
        """
        # Get SNR map
        snr = self.snr_map(
            time_idx=time_idx,
            freq_idx=freq_idx,
            freq_mhz=freq_mhz,
            var=var,
            pol=pol,
            box_size=box_size,
        )

        snr_values = snr.values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius
            snr_values[mask] = np.nan

        # Compute color scale
        if symmetric and vmin is None and vmax is None:
            finite_vals = snr_values[np.isfinite(snr_values)]
            if len(finite_vals) > 0:
                max_abs = np.percentile(np.abs(finite_vals), 98)
                vmin = -max_abs
                vmax = max_abs

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            snr_values.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("SNR (σ)", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        # Get frequency for title
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        freq_hz = float(self._obj.coords["frequency"].values[fi])
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        ax.set_title(
            f"SNR Map at t={time_str} MJD, f={freq_hz/1e6:.2f} MHz\n"
            f"(box_size={box_size})",
            fontsize=11,
        )

        return fig

    # =========================================================================
    # Phase H: Spectral Analysis Methods
    # =========================================================================

    def spectral_index(
        self,
        l: float,
        m: float,
        time_idx: int = 0,
        pol: int = 0,
        freq1_mhz: float | None = None,
        freq2_mhz: float | None = None,
        freq1_idx: int | None = None,
        freq2_idx: int | None = None,
        var: str = "SKY",
    ) -> float:
        """Compute spectral index (power-law slope) between two frequencies.

        The spectral index α is defined by the power-law relationship S ∝ ν^α,
        computed as: α = log(S2/S1) / log(ν2/ν1)

        Parameters
        ----------
        l : float
            The l coordinate value for the pixel location.
        m : float
            The m coordinate value for the pixel location.
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq1_mhz : float, optional
            First frequency in MHz. If not provided, uses freq1_idx or first channel.
        freq2_mhz : float, optional
            Second frequency in MHz. If not provided, uses freq2_idx or last channel.
        freq1_idx : int, optional
            First frequency index. Overridden by freq1_mhz if provided.
        freq2_idx : int, optional
            Second frequency index. Overridden by freq2_mhz if provided.
        var : str, default "SKY"
            Data variable to analyze.

        Returns
        -------
        float
            Spectral index α where S ∝ ν^α. Returns NaN if calculation
            is not possible (e.g., non-positive flux values).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Compute spectral index at image center between 46 and 54 MHz
        >>> alpha = ds.radport.spectral_index(
        ...     l=0.0, m=0.0,
        ...     freq1_mhz=46.0,
        ...     freq2_mhz=54.0,
        ... )
        >>> print(f"Spectral index: {alpha:.2f}")

        Notes
        -----
        - Assumes power-law spectrum: S ∝ ν^α
        - Returns NaN for non-positive flux values (cannot take log)
        - Typical radio sources have α ≈ -0.7 (synchrotron emission)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Get pixel indices
        l_idx, m_idx = self.nearest_lm_idx(l, m)

        # Resolve frequency indices
        if freq1_mhz is not None:
            fi1 = self.nearest_freq_idx(freq1_mhz)
        elif freq1_idx is not None:
            fi1 = freq1_idx
        else:
            fi1 = 0

        if freq2_mhz is not None:
            fi2 = self.nearest_freq_idx(freq2_mhz)
        elif freq2_idx is not None:
            fi2 = freq2_idx
        else:
            fi2 = len(self._obj.coords["frequency"]) - 1

        # Get flux values at both frequencies
        s1 = float(
            self._obj[var]
            .isel(time=time_idx, frequency=fi1, polarization=pol, l=l_idx, m=m_idx)
            .values
        )
        s2 = float(
            self._obj[var]
            .isel(time=time_idx, frequency=fi2, polarization=pol, l=l_idx, m=m_idx)
            .values
        )

        # Get frequency values in Hz
        nu1 = float(self._obj.coords["frequency"].values[fi1])
        nu2 = float(self._obj.coords["frequency"].values[fi2])

        # Compute spectral index: α = log(S2/S1) / log(ν2/ν1)
        # Handle non-positive flux values
        if s1 <= 0 or s2 <= 0 or nu1 <= 0 or nu2 <= 0 or nu1 == nu2:
            return float("nan")

        alpha = np.log(s2 / s1) / np.log(nu2 / nu1)
        return float(alpha)

    def spectral_index_map(
        self,
        time_idx: int = 0,
        pol: int = 0,
        freq1_mhz: float | None = None,
        freq2_mhz: float | None = None,
        freq1_idx: int | None = None,
        freq2_idx: int | None = None,
        var: str = "SKY",
    ) -> xr.DataArray:
        """Compute spectral index map across the image.

        Computes the spectral index α at each pixel, where S ∝ ν^α.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq1_mhz : float, optional
            First frequency in MHz. If not provided, uses freq1_idx or first channel.
        freq2_mhz : float, optional
            Second frequency in MHz. If not provided, uses freq2_idx or last channel.
        freq1_idx : int, optional
            First frequency index. Overridden by freq1_mhz if provided.
        freq2_idx : int, optional
            Second frequency index. Overridden by freq2_mhz if provided.
        var : str, default "SKY"
            Data variable to analyze.

        Returns
        -------
        xr.DataArray
            2D array of spectral index values with dimensions (l, m).
            NaN values indicate pixels where the calculation was not possible.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Compute spectral index map between first and last frequency
        >>> alpha_map = ds.radport.spectral_index_map()
        >>> alpha_map.plot(vmin=-3, vmax=1, cmap="RdBu_r")
        >>>
        >>> # Compute between specific frequencies
        >>> alpha_map = ds.radport.spectral_index_map(
        ...     freq1_mhz=46.0,
        ...     freq2_mhz=54.0,
        ... )
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency indices
        if freq1_mhz is not None:
            fi1 = self.nearest_freq_idx(freq1_mhz)
        elif freq1_idx is not None:
            fi1 = freq1_idx
        else:
            fi1 = 0

        if freq2_mhz is not None:
            fi2 = self.nearest_freq_idx(freq2_mhz)
        elif freq2_idx is not None:
            fi2 = freq2_idx
        else:
            fi2 = len(self._obj.coords["frequency"]) - 1

        # Get flux arrays at both frequencies
        s1 = self._obj[var].isel(
            time=time_idx, frequency=fi1, polarization=pol
        ).values.astype(float)
        s2 = self._obj[var].isel(
            time=time_idx, frequency=fi2, polarization=pol
        ).values.astype(float)

        # Get frequency values in Hz
        nu1 = float(self._obj.coords["frequency"].values[fi1])
        nu2 = float(self._obj.coords["frequency"].values[fi2])

        # Compute spectral index: α = log(S2/S1) / log(ν2/ν1)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Mask non-positive values
            valid_mask = (s1 > 0) & (s2 > 0)
            alpha = np.full_like(s1, np.nan)
            alpha[valid_mask] = (
                np.log(s2[valid_mask] / s1[valid_mask]) / np.log(nu2 / nu1)
            )

        # Create DataArray with coordinates
        return xr.DataArray(
            alpha,
            dims=["l", "m"],
            coords={
                "l": self._obj.coords["l"],
                "m": self._obj.coords["m"],
            },
            name="spectral_index",
            attrs={
                "long_name": "Spectral index",
                "units": "",
                "freq1_hz": nu1,
                "freq2_hz": nu2,
                "freq1_mhz": nu1 / 1e6,
                "freq2_mhz": nu2 / 1e6,
            },
        )

    def integrated_flux(
        self,
        l: float,
        m: float,
        time_idx: int = 0,
        pol: int = 0,
        freq_min_mhz: float | None = None,
        freq_max_mhz: float | None = None,
        freq_indices: list[int] | None = None,
        var: str = "SKY",
    ) -> float:
        """Compute integrated flux density over a frequency band.

        Integrates the flux density across the specified frequency range
        using the trapezoidal rule.

        Parameters
        ----------
        l : float
            The l coordinate value for the pixel location.
        m : float
            The m coordinate value for the pixel location.
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq_min_mhz : float, optional
            Minimum frequency in MHz. If not provided, uses full range.
        freq_max_mhz : float, optional
            Maximum frequency in MHz. If not provided, uses full range.
        freq_indices : list of int, optional
            Specific frequency indices to include. Overrides freq_min/max_mhz.
        var : str, default "SKY"
            Data variable to analyze.

        Returns
        -------
        float
            Integrated flux density in Jy·Hz. Divide by bandwidth to get
            average flux density.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Compute integrated flux at image center across all frequencies
        >>> flux = ds.radport.integrated_flux(l=0.0, m=0.0)
        >>> print(f"Integrated flux: {flux:.2e} Jy·Hz")
        >>>
        >>> # Compute over specific band
        >>> flux = ds.radport.integrated_flux(
        ...     l=0.0, m=0.0,
        ...     freq_min_mhz=45.0,
        ...     freq_max_mhz=55.0,
        ... )

        Notes
        -----
        Uses trapezoidal integration over the frequency axis.
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Get pixel indices
        l_idx, m_idx = self.nearest_lm_idx(l, m)

        # Get all frequency values
        freq_hz = self._obj.coords["frequency"].values

        # Determine which frequencies to include
        if freq_indices is not None:
            indices = freq_indices
        else:
            if freq_min_mhz is not None:
                min_idx = self.nearest_freq_idx(freq_min_mhz)
            else:
                min_idx = 0

            if freq_max_mhz is not None:
                max_idx = self.nearest_freq_idx(freq_max_mhz)
            else:
                max_idx = len(freq_hz) - 1

            # Ensure proper ordering
            if min_idx > max_idx:
                min_idx, max_idx = max_idx, min_idx

            indices = list(range(min_idx, max_idx + 1))

        if len(indices) < 2:
            # Need at least 2 points for integration
            if len(indices) == 1:
                # Return single point value (no integration possible)
                return float(
                    self._obj[var]
                    .isel(
                        time=time_idx,
                        frequency=indices[0],
                        polarization=pol,
                        l=l_idx,
                        m=m_idx,
                    )
                    .values
                )
            return 0.0

        # Get flux values at selected frequencies
        flux_values = []
        freq_values = []
        for fi in indices:
            flux = float(
                self._obj[var]
                .isel(time=time_idx, frequency=fi, polarization=pol, l=l_idx, m=m_idx)
                .values
            )
            flux_values.append(flux)
            freq_values.append(float(freq_hz[fi]))

        flux_values = np.array(flux_values)
        freq_values = np.array(freq_values)

        # Integrate using trapezoidal rule
        integrated = np.trapezoid(flux_values, freq_values)

        return float(integrated)

    def plot_spectral_index_map(
        self,
        time_idx: int = 0,
        pol: int = 0,
        freq1_mhz: float | None = None,
        freq2_mhz: float | None = None,
        freq1_idx: int | None = None,
        freq2_idx: int | None = None,
        var: str = "SKY",
        cmap: str = "RdBu_r",
        vmin: float | None = -3.0,
        vmax: float | None = 1.0,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
    ) -> "Figure":
        """Plot the spectral index map.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq1_mhz : float, optional
            First frequency in MHz.
        freq2_mhz : float, optional
            Second frequency in MHz.
        freq1_idx : int, optional
            First frequency index.
        freq2_idx : int, optional
            Second frequency index.
        var : str, default "SKY"
            Data variable to analyze.
        cmap : str, default "RdBu_r"
            Colormap (diverging recommended for spectral index).
        vmin : float, default -3.0
            Minimum value for color scaling.
        vmax : float, default 1.0
            Maximum value for color scaling.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Example
        -------
        >>> fig = ds.radport.plot_spectral_index_map(
        ...     freq1_mhz=46.0,
        ...     freq2_mhz=54.0,
        ...     mask_radius=1800,
        ... )
        """
        # Get spectral index map
        alpha_map = self.spectral_index_map(
            time_idx=time_idx,
            pol=pol,
            freq1_mhz=freq1_mhz,
            freq2_mhz=freq2_mhz,
            freq1_idx=freq1_idx,
            freq2_idx=freq2_idx,
            var=var,
        )

        alpha_values = alpha_map.values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius
            alpha_values[mask] = np.nan

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            alpha_values.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Spectral Index (α)", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        # Title
        freq1_hz = alpha_map.attrs.get("freq1_hz", 0)
        freq2_hz = alpha_map.attrs.get("freq2_hz", 0)
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        ax.set_title(
            f"Spectral Index Map at t={time_str} MJD\n"
            f"({freq1_hz/1e6:.1f} - {freq2_hz/1e6:.1f} MHz)",
            fontsize=11,
        )

        return fig

    # =========================================================================
    # Dispersion Measure Correction Methods
    # =========================================================================

    # Dispersion constant in MHz^2 pc^-1 cm^3 s
    # Reference: Lorimer & Kramer (2004), Handbook of Pulsar Astronomy
    K_DM = 4.148808e3  # MHz^2 pc^-1 cm^3 s

    def dispersion_delay(
        self,
        dm: float,
        freq_mhz: float | np.ndarray | None = None,
        freq_ref_mhz: float | None = None,
    ) -> float | np.ndarray:
        """Calculate dispersion delay for a given DM and frequency.

        Radio signals experience frequency-dependent delays when propagating
        through the ionized interstellar medium. Lower frequencies arrive
        later than higher frequencies. This method computes the time delay
        using the cold plasma dispersion relation.

        Parameters
        ----------
        dm : float
            Dispersion measure in pc cm^-3. Must be non-negative.
        freq_mhz : float or np.ndarray, optional
            Frequency or array of frequencies in MHz at which to compute
            delays. If None, uses all frequencies in the dataset.
        freq_ref_mhz : float, optional
            Reference frequency in MHz (typically the highest frequency).
            Delays are computed relative to this frequency.
            If None, uses the highest frequency in the dataset.

        Returns
        -------
        float or np.ndarray
            Time delay(s) in seconds. Positive values indicate the signal
            arrives later at lower frequencies. Returns the same shape as
            freq_mhz input.

        Raises
        ------
        ValueError
            If dm is negative.

        Notes
        -----
        The dispersion delay is computed using:

            Δt = K_DM × DM × (f_lo^-2 - f_hi^-2)

        where:
        - K_DM = 4.148808 × 10^3 MHz^2 pc^-1 cm^3 s (dispersion constant)
        - DM is the dispersion measure in pc cm^-3
        - f_lo, f_hi are frequencies in MHz

        Example
        -------
        >>> # Crab pulsar DM = 56.8 pc cm^-3
        >>> dm = 56.8
        >>> delay = ds.radport.dispersion_delay(dm=dm, freq_mhz=46.0)
        >>> print(f"Delay at 46 MHz: {delay:.3f} seconds")

        >>> # Get delays at all dataset frequencies
        >>> delays = ds.radport.dispersion_delay(dm=56.8)

        References
        ----------
        .. [1] Lorimer & Kramer (2004), "Handbook of Pulsar Astronomy"
        """
        # Validate DM
        if dm < 0:
            raise ValueError(f"DM must be non-negative, got {dm}")

        # Get frequencies
        if freq_mhz is None:
            freq_mhz = self._obj.coords["frequency"].values / 1e6

        freq_mhz = np.asarray(freq_mhz)

        # Get reference frequency (highest frequency by default)
        if freq_ref_mhz is None:
            freq_ref_mhz = float(self._obj.coords["frequency"].values.max() / 1e6)

        # Validate reference frequency
        if freq_ref_mhz <= 0:
            raise ValueError(f"Reference frequency must be positive, got {freq_ref_mhz}")

        # Validate input frequencies
        if np.any(freq_mhz <= 0):
            raise ValueError("All frequencies must be positive")

        # Compute delay: Δt = K_DM × DM × (f^-2 - f_ref^-2)
        delay = self.K_DM * dm * (freq_mhz**-2 - freq_ref_mhz**-2)

        return delay

    def dynamic_spectrum_dedispersed(
        self,
        l: float,
        m: float,
        dm: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        method: Literal["shift", "interpolate"] = "shift",
        fill_value: float = np.nan,
        trim: bool = False,
    ) -> xr.DataArray:
        """Extract a dedispersed dynamic spectrum for a single pixel.

        Corrects for interstellar dispersion by shifting or interpolating
        frequency channels according to the dispersion delay. This is essential
        for analyzing dispersed radio transients like pulsars and FRBs.

        Parameters
        ----------
        l : float
            Target l coordinate for pixel selection.
        m : float
            Target m coordinate for pixel selection.
        dm : float
            Dispersion measure in pc cm^-3. Must be non-negative.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.
        method : {'shift', 'interpolate'}, default 'shift'
            Dedispersion method:
            - 'shift': Fast integer-sample shifting (approximate).
              Rounds delays to nearest time sample.
            - 'interpolate': Slower but precise sub-sample interpolation.
              Uses linear interpolation for accurate delay correction.
        fill_value : float, default np.nan
            Value to use for samples shifted outside the time range.
        trim : bool, default False
            If True, trim the time axis to only include valid data
            (removes NaN edges from shifting). If False, returns full
            time axis with NaN-filled edges.

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (time, frequency) containing
            the dedispersed dynamic spectrum. Time axis represents
            arrival time at the reference frequency.
            Includes metadata: pixel_l, pixel_m, dm, method, freq_ref_mhz.

        Raises
        ------
        ValueError
            If dm is negative, variable doesn't exist, or method is invalid.

        Warns
        -----
        UserWarning
            If the maximum dispersion shift exceeds 50% of the time span.

        Notes
        -----
        The dedispersion aligns all frequency channels to a common reference
        time (typically the highest frequency). Lower frequency channels are
        shifted backwards in time to compensate for the dispersion delay.

        For the 'shift' method, delays are rounded to the nearest integer
        number of time samples, which introduces quantization error. For
        precise analysis, use 'interpolate'.

        Example
        -------
        >>> # Dedisperse at Crab pulsar DM
        >>> dynspec = ds.radport.dynamic_spectrum_dedispersed(
        ...     l=0.0, m=0.0, dm=56.8, method="interpolate"
        ... )

        >>> # Fast approximate dedispersion
        >>> dynspec_fast = ds.radport.dynamic_spectrum_dedispersed(
        ...     l=0.0, m=0.0, dm=56.8, method="shift", trim=True
        ... )

        See Also
        --------
        dispersion_delay : Compute dispersion delays for given frequencies.
        dynamic_spectrum : Extract uncorrected dynamic spectrum.
        plot_dynamic_spectrum : Plot dynamic spectrum with optional dedispersion.
        """
        # Validate inputs
        if dm < 0:
            raise ValueError(f"DM must be non-negative, got {dm}")

        if method not in ("shift", "interpolate"):
            raise ValueError(
                f"Method must be 'shift' or 'interpolate', got '{method}'"
            )

        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        # Get the uncorrected dynamic spectrum
        dynspec = self.dynamic_spectrum(l=l, m=m, var=var, pol=pol)

        # If DM is zero, return the original spectrum
        if dm == 0:
            dynspec.attrs["dm"] = 0.0
            dynspec.attrs["method"] = method
            dynspec.attrs["freq_ref_mhz"] = float(
                self._obj.coords["frequency"].values.max() / 1e6
            )
            return dynspec

        # Get coordinates
        time_vals = dynspec.coords["time"].values
        freq_vals = dynspec.coords["frequency"].values  # Hz
        freq_mhz = freq_vals / 1e6

        # Compute reference frequency (highest)
        freq_ref_mhz = float(freq_mhz.max())

        # Compute dispersion delays for each frequency channel
        delays = self.dispersion_delay(dm=dm, freq_mhz=freq_mhz, freq_ref_mhz=freq_ref_mhz)

        # Get time resolution
        if len(time_vals) < 2:
            raise ValueError("Need at least 2 time samples for dedispersion")

        dt = float(time_vals[1] - time_vals[0])  # Time resolution in MJD
        dt_seconds = dt * 86400.0  # Convert to seconds

        # Convert delays to time samples
        delay_samples = delays / dt_seconds

        # Check for excessive delays
        max_delay_samples = np.abs(delay_samples).max()
        if max_delay_samples > 0.5 * len(time_vals):
            warnings.warn(
                f"Maximum dispersion shift ({max_delay_samples:.1f} samples) "
                f"exceeds 50% of time span ({len(time_vals)} samples). "
                "Consider using a smaller DM or longer observation.",
                UserWarning,
                stacklevel=2,
            )

        # Get data values
        data = dynspec.values.copy()  # Shape: (time, frequency)
        n_time, n_freq = data.shape

        # Create output array
        dedispersed = np.full_like(data, fill_value)

        if method == "shift":
            # Integer sample shifting (fast, approximate)
            for i_freq in range(n_freq):
                shift = int(np.round(delay_samples[i_freq]))

                if shift == 0:
                    dedispersed[:, i_freq] = data[:, i_freq]
                elif shift > 0:
                    # Signal arrives later at lower freq, shift backwards
                    if shift < n_time:
                        dedispersed[:-shift, i_freq] = data[shift:, i_freq]
                else:
                    # Negative shift (shouldn't happen for positive DM)
                    shift = abs(shift)
                    if shift < n_time:
                        dedispersed[shift:, i_freq] = data[:-shift, i_freq]

        else:  # method == "interpolate"
            # Sub-sample interpolation (slower, precise)
            for i_freq in range(n_freq):
                delay_mjd = delays[i_freq] / 86400.0  # Convert to MJD

                # Create interpolator for this frequency channel
                interp_func = interpolate.interp1d(
                    time_vals,
                    data[:, i_freq],
                    kind="linear",
                    bounds_error=False,
                    fill_value=fill_value,
                )

                # Interpolate at shifted times
                # To correct for dispersion, we sample at time + delay
                shifted_times = time_vals + delay_mjd
                dedispersed[:, i_freq] = interp_func(shifted_times)

        # Trim if requested
        if trim:
            # Find valid time range (where all frequencies have data)
            valid_mask = ~np.all(np.isnan(dedispersed), axis=1)
            if np.any(valid_mask):
                first_valid = np.argmax(valid_mask)
                last_valid = len(valid_mask) - np.argmax(valid_mask[::-1]) - 1
                dedispersed = dedispersed[first_valid:last_valid + 1, :]
                time_vals = time_vals[first_valid:last_valid + 1]

        # Create output DataArray
        result = xr.DataArray(
            dedispersed,
            dims=["time", "frequency"],
            coords={
                "time": time_vals,
                "frequency": freq_vals,
            },
            name=f"{var}_dedispersed",
            attrs={
                "pixel_l": dynspec.attrs["pixel_l"],
                "pixel_m": dynspec.attrs["pixel_m"],
                "l_idx": dynspec.attrs["l_idx"],
                "m_idx": dynspec.attrs["m_idx"],
                "pol": pol,
                "dm": dm,
                "method": method,
                "freq_ref_mhz": freq_ref_mhz,
                "long_name": f"Dedispersed {var} (DM={dm:.2f} pc/cm³)",
                "units": "Jy/beam",
            },
        )

        return result

    def plot_dynamic_spectrum_dedispersed(
        self,
        l: float,
        m: float,
        dm: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        method: Literal["shift", "interpolate"] = "shift",
        trim: bool = False,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (10, 5),
        add_colorbar: bool = True,
        show_delay_curve: bool = False,
        **kwargs: Any,
    ) -> "Figure":
        """Plot a dedispersed dynamic spectrum for a single pixel.

        Creates a 2D visualization showing intensity variations across
        time and frequency after correcting for interstellar dispersion.

        Parameters
        ----------
        l : float
            Target l coordinate for pixel selection.
        m : float
            Target m coordinate for pixel selection.
        dm : float
            Dispersion measure in pc cm^-3.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        method : {'shift', 'interpolate'}, default 'shift'
            Dedispersion method ('shift' for fast, 'interpolate' for precise).
        trim : bool, default False
            If True, trim time axis to valid data only.
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based color scaling.
        figsize : tuple, default (10, 5)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        show_delay_curve : bool, default False
            If True, overlay the dispersion delay curve on the plot.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the dedispersed dynamic spectrum plot.

        Example
        -------
        >>> fig = ds.radport.plot_dynamic_spectrum_dedispersed(
        ...     l=0.0, m=0.0, dm=56.8, method="interpolate"
        ... )
        """
        # Get dedispersed dynamic spectrum
        dynspec = self.dynamic_spectrum_dedispersed(
            l=l, m=m, dm=dm, var=var, pol=pol, method=method, trim=trim
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = dynspec.values

        # Handle robust scaling
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate values
        time_vals = dynspec.coords["time"].values
        freq_vals = dynspec.coords["frequency"].values / 1e6  # Convert to MHz

        # Compute extent for imshow
        extent = [
            float(time_vals.min()), float(time_vals.max()),
            float(freq_vals.min()), float(freq_vals.max()),
        ]

        # Plot - transpose so time is x-axis and frequency is y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="auto",
            **kwargs,
        )

        # Optionally show dispersion delay curve
        if show_delay_curve and dm > 0:
            delays = self.dispersion_delay(dm=dm, freq_mhz=freq_vals)
            # Convert delays to MJD offset from reference
            delay_mjd = delays / 86400.0
            # Plot as time offset from center of time range
            t_center = (time_vals.min() + time_vals.max()) / 2
            ax.plot(
                t_center - delay_mjd,
                freq_vals,
                "w--",
                linewidth=1.5,
                alpha=0.7,
                label="Dispersion curve",
            )
            ax.legend(loc="upper right")

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Labels and title
        pixel_l = dynspec.attrs["pixel_l"]
        pixel_m = dynspec.attrs["pixel_m"]
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(
            f"{var} Dedispersed Dynamic Spectrum\n"
            f"l={pixel_l:+.4f}, m={pixel_m:+.4f}, pol={pol}, "
            f"DM={dm:.2f} pc/cm³ ({method})"
        )

        fig.tight_layout()
        return fig
