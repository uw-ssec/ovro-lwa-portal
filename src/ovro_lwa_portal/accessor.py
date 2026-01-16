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

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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
