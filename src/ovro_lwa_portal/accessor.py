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

import contextlib
import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import interpolate

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.figure import Figure


# Results smaller than this are eagerly loaded to avoid dask graph
# scheduling overhead.  10 MB covers all realistic OVRO-LWA single-pixel
# extractions (e.g. 1000 times x 1000 freqs x 8 bytes = 8 MB) while
# keeping pathologically large results lazy.
_EAGER_LOAD_THRESHOLD = 10 * 1024 * 1024  # bytes


def _maybe_load(da: xr.DataArray) -> xr.DataArray:
    """Eagerly load a dask-backed DataArray if it is below the size threshold."""
    if hasattr(da, "nbytes") and da.nbytes < _EAGER_LOAD_THRESHOLD:
        if hasattr(da, "load"):
            da = da.load()
    return da


@contextlib.contextmanager
def _dask_progress(label: str = "Computing") -> Generator[None, None, None]:
    """Show a dask progress bar when dask is available.

    Falls back to a simple print when dask.diagnostics is unavailable.
    """
    try:
        from dask.diagnostics import ProgressBar

        with ProgressBar(dt=1.0, minimum=2.0):
            print(f"{label}...")  # noqa: T201
            yield
    except ImportError:
        yield


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
        self._lst_cache: dict[tuple, np.ndarray] = {}
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

    def _compute_pixel_track(
        self,
        ra: float,
        dec: float,
        observatory: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-time-step pixel indices for a fixed (RA, Dec).

        Uses the closed-form SIN projection equations and LST derived from
        MJD timestamps to track a celestial source across time steps.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees (FK5/J2000).
        dec : float
            Declination in degrees (FK5/J2000).
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location. Defaults to OVRO-LWA.

        Returns
        -------
        l_indices : np.ndarray of int, shape (n_times,)
            Pixel l-index at each time step. -1 where source is below horizon.
        m_indices : np.ndarray of int, shape (n_times,)
            Pixel m-index at each time step. -1 where source is below horizon.
        visible : np.ndarray of bool, shape (n_times,)
            True where source is above the horizon.
        """
        from astropy.coordinates import EarthLocation
        from astropy.time import Time
        from astropy.utils.iers import conf as iers_conf

        # OVRO-LWA location (default)
        if observatory is None:
            from astropy import units as u

            observatory = EarthLocation(
                lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
            )

        # Get MJD timestamps from dataset, ensuring float64 dtype
        mjd_array = np.asarray(self._obj.coords["time"].values, dtype=np.float64)
        # Handle scalar (single time step) by ensuring 1-d
        mjd_array = np.atleast_1d(mjd_array)

        # Vectorized LST computation — cached per (times, longitude) pair.
        # The LST only depends on the MJD timestamps and the observatory
        # longitude, so reuse across calls with different (RA, Dec).
        # Use bundled IERS-B data to avoid network downloads.
        # Mean sidereal time has ~1 arcsec error — negligible at
        # OVRO-LWA's ~6-arcmin beam resolution.
        lon_deg = float(observatory.lon.deg)
        cache_key = (mjd_array.tobytes(), lon_deg)

        if cache_key in self._lst_cache:
            lst_deg = self._lst_cache[cache_key]
        else:
            orig_auto_download = iers_conf.auto_download
            try:
                iers_conf.auto_download = False
                t = Time(mjd_array, format="mjd", scale="utc")
                lst_deg = t.sidereal_time("mean", longitude=observatory.lon).deg
            finally:
                iers_conf.auto_download = orig_auto_download
            self._lst_cache[cache_key] = lst_deg

        # Closed-form SIN projection (vectorized)
        ha_rad = np.deg2rad(lst_deg - ra)
        dec_rad = np.deg2rad(dec)
        lat_rad = np.deg2rad(observatory.lat.deg)

        l_vals = np.cos(dec_rad) * np.sin(ha_rad)
        m_vals = np.sin(dec_rad) * np.cos(lat_rad) - np.cos(dec_rad) * np.sin(
            lat_rad
        ) * np.cos(ha_rad)

        # Check visibility: source above horizon when sin(altitude) > 0
        # sin(alt) = sin(dec)*sin(lat) + cos(dec)*cos(lat)*cos(ha)
        sin_alt = np.sin(dec_rad) * np.sin(lat_rad) + np.cos(
            dec_rad
        ) * np.cos(lat_rad) * np.cos(ha_rad)
        visible = sin_alt > 0

        # Get coordinate arrays for pixel lookup
        l_coords = self._obj.coords["l"].values
        m_coords = self._obj.coords["m"].values

        # Vectorized nearest-neighbor pixel lookup using searchsorted.
        # argsort provides the mapping from sorted→original positions so
        # the returned indices are valid for isel() against the dataset.
        l_order = np.argsort(l_coords)
        m_order = np.argsort(m_coords)
        l_sorted = l_coords[l_order]
        m_sorted = m_coords[m_order]

        l_insert = np.searchsorted(l_sorted, l_vals)
        m_insert = np.searchsorted(m_sorted, m_vals)

        # Clamp to valid range
        l_insert = np.clip(l_insert, 0, len(l_sorted) - 1)
        m_insert = np.clip(m_insert, 0, len(m_sorted) - 1)

        # Choose nearest neighbor (searchsorted gives insertion point,
        # check if previous index is closer)
        l_prev = np.clip(l_insert - 1, 0, len(l_sorted) - 1)
        l_sorted_idx = np.where(
            np.abs(l_sorted[l_insert] - l_vals) <= np.abs(l_sorted[l_prev] - l_vals),
            l_insert,
            l_prev,
        )

        m_prev = np.clip(m_insert - 1, 0, len(m_sorted) - 1)
        m_sorted_idx = np.where(
            np.abs(m_sorted[m_insert] - m_vals) <= np.abs(m_sorted[m_prev] - m_vals),
            m_insert,
            m_prev,
        )

        # Map sorted indices back to original coordinate order
        l_indices = l_order[l_sorted_idx]
        m_indices = m_order[m_sorted_idx]

        # Check pixel bounds (source may be visible but outside image FOV)
        in_bounds = (
            (l_vals >= l_sorted[0])
            & (l_vals <= l_sorted[-1])
            & (m_vals >= m_sorted[0])
            & (m_vals <= m_sorted[-1])
        )
        visible = visible & in_bounds

        # Warn if source is never visible during the observation
        if not np.any(visible):
            warnings.warn(
                f"Source (RA={ra}°, Dec={dec}°) is never above the horizon "
                f"during this observation. All output values will be NaN.",
                stacklevel=3,
            )

        # Mark below-horizon / out-of-bounds time steps with out-of-range
        # sentinel (n_l, not -1, to avoid silent last-element extraction
        # if a caller forgets to check the visible mask).
        n_l = len(l_coords)
        n_m = len(m_coords)
        l_indices[~visible] = n_l
        m_indices[~visible] = n_m

        return l_indices.astype(int), m_indices.astype(int), visible

    def _compute_pixel_at_time(
        self,
        ra: float,
        dec: float,
        time_idx: int,
        observatory: Any = None,
    ) -> tuple[int, int]:
        """Compute the pixel index for (RA, Dec) at a single time step.

        This is a lightweight alternative to `_compute_pixel_track` for
        single-frame methods (spectrum, cutout, spectral_index, etc.)
        that only need one time step.  It computes the LST for just the
        requested timestamp rather than the full time axis.

        Parameters
        ----------
        ra, dec : float
            Source coordinates in degrees (FK5/J2000).
        time_idx : int
            Index into the dataset's time dimension.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location. Defaults to OVRO-LWA.

        Returns
        -------
        tuple[int, int]
            ``(l_idx, m_idx)`` pixel indices.

        Raises
        ------
        ValueError
            If the source is below the horizon or outside the image FOV
            at the requested time step.
        """
        from astropy.coordinates import EarthLocation
        from astropy.time import Time
        from astropy.utils.iers import conf as iers_conf

        if observatory is None:
            from astropy import units as u

            observatory = EarthLocation(
                lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
            )

        mjd = float(self._obj.coords["time"].values[time_idx])
        lon_deg = float(observatory.lon.deg)

        # Check the full-array LST cache first — if a prior
        # _compute_pixel_track call already computed all LSTs, reuse it.
        all_mjd = np.asarray(self._obj.coords["time"].values, dtype=np.float64)
        all_mjd = np.atleast_1d(all_mjd)
        full_key = (all_mjd.tobytes(), lon_deg)

        if full_key in self._lst_cache:
            lst_deg = float(self._lst_cache[full_key][time_idx])
        else:
            # Compute LST for just this single timestamp.
            single_key = (np.float64(mjd).tobytes(), lon_deg)
            if single_key in self._lst_cache:
                lst_deg = float(self._lst_cache[single_key])
            else:
                orig = iers_conf.auto_download
                try:
                    iers_conf.auto_download = False
                    t = Time(mjd, format="mjd", scale="utc")
                    lst_deg = float(
                        t.sidereal_time("mean", longitude=observatory.lon).deg
                    )
                finally:
                    iers_conf.auto_download = orig
                self._lst_cache[single_key] = np.array(lst_deg)

        # SIN projection for the single time step
        ha_rad = np.deg2rad(lst_deg - ra)
        dec_rad = np.deg2rad(dec)
        lat_rad = np.deg2rad(observatory.lat.deg)

        l_val = np.cos(dec_rad) * np.sin(ha_rad)
        m_val = np.sin(dec_rad) * np.cos(lat_rad) - np.cos(dec_rad) * np.sin(
            lat_rad
        ) * np.cos(ha_rad)

        # Visibility check
        sin_alt = np.sin(dec_rad) * np.sin(lat_rad) + np.cos(
            dec_rad
        ) * np.cos(lat_rad) * np.cos(ha_rad)
        if sin_alt <= 0:
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) is below the horizon "
                f"at time index {time_idx}."
            )

        # Nearest-neighbor pixel lookup
        l_coords = self._obj.coords["l"].values
        m_coords = self._obj.coords["m"].values

        l_idx = int(np.argmin(np.abs(l_coords - l_val)))
        m_idx = int(np.argmin(np.abs(m_coords - m_val)))

        # Bounds check
        if not (l_coords.min() <= l_val <= l_coords.max()):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) maps to l={l_val:.4f} which "
                f"is outside the image FOV at time index {time_idx}."
            )
        if not (m_coords.min() <= m_val <= m_coords.max()):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) maps to m={m_val:.4f} which "
                f"is outside the image FOV at time index {time_idx}."
            )

        return l_idx, m_idx

    def _resolve_coordinates(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        observatory: Any = None,
    ) -> (
        tuple[int, int]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """Validate coordinate input and return pixel indices.

        Dispatches to either the fixed-pixel path (l/m) or the per-time
        tracking path (ra/dec).

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees.
        dec : float, optional
            Declination in degrees.
        l : float, optional
            Direction cosine l coordinate.
        m : float, optional
            Direction cosine m coordinate.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.

        Returns
        -------
        tuple[int, int]
            Fixed pixel indices ``(l_idx, m_idx)`` when l/m provided.
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Per-time ``(l_indices, m_indices, visible)`` when ra/dec provided.

        Raises
        ------
        ValueError
            If input is ambiguous (both pairs, neither pair, or partial pair).
        """
        has_radec = ra is not None or dec is not None
        has_lm = l is not None or m is not None

        if has_radec and has_lm:
            raise ValueError("Provide either (ra, dec) or (l, m), not both.")

        if not has_radec and not has_lm:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

        if has_radec:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            return self._compute_pixel_track(ra, dec, observatory=observatory)

        # l/m path
        if l is None or m is None:
            raise ValueError("Both l and m must be provided together.")
        return self.nearest_lm_idx(l, m)

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
        extent = [float(l_vals[0]), float(l_vals[-1]),
                  float(m_vals[0]), float(m_vals[-1])]

        # Plot the image.
        # Transpose: xarray dims are (l, m) where l=NAXIS1 (RA/x) and
        # m=NAXIS2 (Dec/y).  imshow maps axis 0→y, axis 1→x, so .T
        # puts l on x and m on y.
        im = ax.imshow(
            data.T,
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
        *,
        ra_center: float | None = None,
        dec_center: float | None = None,
        l_center: float | None = None,
        m_center: float | None = None,
        dl: float | None = None,
        dm: float | None = None,
        dra: float | None = None,
        ddec: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
    ) -> xr.DataArray:
        """Extract a spatial cutout (rectangular region) from the data.

        Returns a 2D DataArray containing data within the specified
        bounding box for a given time, frequency, and polarization.

        Parameters
        ----------
        ra_center : float, optional
            Center Right Ascension in degrees. Requires ``dec_center``.
            Converted to l/m at the specified time step using WCS.
        dec_center : float, optional
            Center Declination in degrees. Requires ``ra_center``.
        l_center : float, optional
            Center l coordinate of the cutout region. Requires ``m_center``.
        m_center : float, optional
            Center m coordinate of the cutout region. Requires ``l_center``.
        dl : float, optional
            Half-width of the cutout in the l direction (direction cosine).
            The cutout spans [l_center - dl, l_center + dl].
            Use with l_center/m_center.
        dm : float, optional
            Half-width of the cutout in the m direction (direction cosine).
            The cutout spans [m_center - dm, m_center + dm].
            Use with l_center/m_center.
        dra : float, optional
            Half-width of the cutout in RA in degrees.
            Use with ra_center/dec_center. Converted to dl internally
            using the SIN projection at the center declination.
        ddec : float, optional
            Half-width of the cutout in Dec in degrees.
            Use with ra_center/dec_center. Converted to dm internally.
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
            cutout_dl, cutout_dm. When using RA/Dec, also includes
            cutout_ra_center, cutout_dec_center, cutout_dra, cutout_ddec.

        Raises
        ------
        ValueError
            If the requested variable does not exist or cutout is empty.

        Examples
        --------
        >>> # Extract 0.2 x 0.2 region centered at (0, 0) in direction cosines
        >>> cutout = ds.radport.cutout(l_center=0.0, m_center=0.0, dl=0.1, dm=0.1)

        >>> # Extract using celestial coordinates with extent in degrees
        >>> cutout = ds.radport.cutout(ra_center=180.0, dec_center=45.0, dra=5.0, ddec=5.0)
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

        # Resolve center coordinates
        has_radec = ra_center is not None or dec_center is not None
        has_lm = l_center is not None or m_center is not None

        if has_radec and has_lm:
            raise ValueError(
                "Provide either (ra_center, dec_center) or (l_center, m_center), not both."
            )
        if not has_radec and not has_lm:
            raise ValueError(
                "Must provide either (ra_center, dec_center) or (l_center, m_center)."
            )

        # Track whether we're in celestial mode for metadata/plotting
        celestial_mode = False

        if has_radec:
            if ra_center is None or dec_center is None:
                raise ValueError(
                    "Both ra_center and dec_center must be provided together."
                )
            celestial_mode = True

            # Convert dra/ddec (degrees) to dl/dm (direction cosines)
            if dra is not None or ddec is not None:
                if dra is None or ddec is None:
                    raise ValueError(
                        "Both dra and ddec must be provided together."
                    )
                dec_rad = np.deg2rad(dec_center)
                cos_dec = np.cos(dec_rad)

                # Near the celestial poles, RA extent maps to ~zero
                # extent in l. The pixel scale sets the minimum
                # meaningful dl.
                l_coords = self._obj.coords["l"].values
                pixel_scale_l = float(np.abs(np.median(np.diff(np.sort(l_coords)))))
                dl = np.deg2rad(dra) * cos_dec
                if dl < pixel_scale_l:
                    raise ValueError(
                        f"At Dec={dec_center}°, the requested dra={dra}° "
                        f"maps to dl={dl:.6f} in direction cosines, which "
                        f"is smaller than the pixel scale ({pixel_scale_l:.6f}). "
                        f"Near the celestial poles, RA extent degenerates. "
                        f"Use dl/dm directly for cutouts near the poles."
                    )
                dm = np.deg2rad(ddec)
            elif dl is None or dm is None:
                raise ValueError(
                    "Must provide cutout extent: either (dra, ddec) in degrees "
                    "or (dl, dm) in direction cosines."
                )

            # Convert RA/Dec to pixel at the resolved time step so the
            # cutout is centred on the correct frame.
            pix_l, pix_m = self.coords_to_pixel(
                ra_center, dec_center, time_idx=time_idx
            )
            l_center = float(self._obj.coords["l"].values[pix_l])
            m_center = float(self._obj.coords["m"].values[pix_m])
        else:
            if l_center is None or m_center is None:
                raise ValueError(
                    "Both l_center and m_center must be provided together."
                )
            if dl is None or dm is None:
                raise ValueError(
                    "Both dl and dm must be provided when using l_center/m_center."
                )

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
        cutout.attrs["celestial_mode"] = celestial_mode

        if celestial_mode:
            cutout.attrs["cutout_ra_center"] = ra_center
            cutout.attrs["cutout_dec_center"] = dec_center
            if dra is not None:
                cutout.attrs["cutout_dra"] = dra
                cutout.attrs["cutout_ddec"] = ddec

        return cutout

    def plot_cutout(
        self,
        *,
        ra_center: float | None = None,
        dec_center: float | None = None,
        l_center: float | None = None,
        m_center: float | None = None,
        dl: float | None = None,
        dm: float | None = None,
        dra: float | None = None,
        ddec: float | None = None,
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
        ra_center : float, optional
            Center Right Ascension in degrees. Requires ``dec_center``.
        dec_center : float, optional
            Center Declination in degrees. Requires ``ra_center``.
        l_center : float, optional
            Center l coordinate. Requires ``m_center``.
        m_center : float, optional
            Center m coordinate. Requires ``l_center``.
        dl, dm : float, optional
            Half-widths of the cutout in l and m directions (direction cosines).
        dra, ddec : float, optional
            Half-widths of the cutout in degrees.
            Use with ra_center/dec_center for consistent units.
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
        >>> fig = ds.radport.plot_cutout(l_center=0.0, m_center=0.0, dl=0.1, dm=0.1, freq_mhz=50.0)
        >>> fig = ds.radport.plot_cutout(ra_center=180.0, dec_center=45.0, dra=5.0, ddec=5.0)
        """
        # Get cutout data
        cutout = self.cutout(
            ra_center=ra_center,
            dec_center=dec_center,
            l_center=l_center,
            m_center=m_center,
            dl=dl,
            dm=dm,
            dra=dra,
            ddec=ddec,
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
        celestial_mode = cutout.attrs.get("celestial_mode", False)

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

        # Get coordinate extents and set axis labels
        l_vals = cutout.coords["l"].values
        m_vals = cutout.coords["m"].values

        if celestial_mode and "cutout_dra" in cutout.attrs:
            # Use the requested RA/Dec extent directly — this avoids
            # non-linear SIN projection artifacts when converting l/m
            # pixel corners back to RA/Dec.
            ra_c = cutout.attrs["cutout_ra_center"]
            dec_c = cutout.attrs["cutout_dec_center"]
            dra_val = cutout.attrs["cutout_dra"]
            ddec_val = cutout.attrs["cutout_ddec"]

            # Clamp Dec to physical range and wrap RA to [0, 360)
            ra_min = (ra_c - dra_val) % 360.0
            ra_max = (ra_c + dra_val) % 360.0
            dec_min = max(-90.0, dec_c - ddec_val)
            dec_max = min(90.0, dec_c + ddec_val)

            # If RA wraps across 0°/360°, keep the raw values for
            # a continuous axis (matplotlib handles negative/360+ fine)
            if ra_min > ra_max:
                ra_min = ra_c - dra_val
                ra_max = ra_c + dra_val

            # RA increases to the left on sky images (reversed x-axis)
            extent = [ra_max, ra_min, dec_min, dec_max]

            ax.set_xlabel("RA (degrees)")
            ax.set_ylabel("Dec (degrees)")
        elif celestial_mode:
            # Celestial mode with dl/dm (no dra/ddec) — display in l/m
            # since we can't reliably invert the SIN projection for
            # display extents.
            extent = [
                float(l_vals[0]), float(l_vals[-1]),
                float(m_vals[0]), float(m_vals[-1]),
            ]
            ax.set_xlabel("l (direction cosine)")
            ax.set_ylabel("m (direction cosine)")
        else:
            extent = [
                float(l_vals[0]), float(l_vals[-1]),
                float(m_vals[0]), float(m_vals[-1]),
            ]
            ax.set_xlabel("l (direction cosine)")
            ax.set_ylabel("m (direction cosine)")

        # Plot — transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
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
        if celestial_mode:
            ra_c = cutout.attrs["cutout_ra_center"]
            dec_c = cutout.attrs["cutout_dec_center"]
            title += f"\nRA={ra_c:.4f}°, Dec={dec_c:.4f}°"
        else:
            resolved_l = cutout.attrs["cutout_l_center"]
            resolved_m = cutout.attrs["cutout_m_center"]
            dl_val = cutout.attrs["cutout_dl"]
            dm_val = cutout.attrs["cutout_dm"]
            title += (
                f"\nl=[{resolved_l - dl_val:+.2f},{resolved_l + dl_val:+.2f}], "
                f"m=[{resolved_m - dm_val:+.2f},{resolved_m + dm_val:+.2f}]"
            )

        ax.set_title(title)

        fig.tight_layout()
        return fig

    # =========================================================================
    # Dynamic Spectrum Methods
    # =========================================================================

    def dynamic_spectrum(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        observatory: Any = None,
    ) -> xr.DataArray:
        """Extract a dynamic spectrum (time vs frequency) for a single pixel.

        Returns a 2D DataArray showing how intensity varies across time
        and frequency at the pixel nearest to the specified location.

        When celestial coordinates (ra, dec) are provided, the pixel is
        tracked across time steps as the source drifts due to Earth rotation.
        Time steps where the source is below the horizon are NaN-filled.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Target l direction cosine coordinate. Requires ``m``.
        m : float, optional
            Target m direction cosine coordinate. Requires ``l``.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (time, frequency).
            Includes metadata: pixel_l, pixel_m, l_idx, m_idx, pol.
            When ra/dec is used, also includes ra, dec, and tracking=True.

        Examples
        --------
        >>> # Get dynamic spectrum at image center (direction cosines)
        >>> dynspec = ds.radport.dynamic_spectrum(l=0.0, m=0.0)

        >>> # Track a celestial source across time
        >>> dynspec = ds.radport.dynamic_spectrum(ra=180.0, dec=45.0)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        result = self._resolve_coordinates(
            ra=ra, dec=dec, l=l, m=m, observatory=observatory
        )

        if isinstance(result, tuple) and len(result) == 2:
            # Fixed pixel path (l/m) — single pixel across all times/freqs.
            # Eagerly load small results to avoid dask graph overhead.
            l_idx, m_idx = result
            da = _maybe_load(
                self._obj[var].isel(l=l_idx, m=m_idx, polarization=pol)
            )

            if "time" in da.dims:
                da = da.sortby("time")
            if "frequency" in da.dims:
                da = da.sortby("frequency")

            da.attrs["pixel_l"] = float(self._obj.coords["l"].values[l_idx])
            da.attrs["pixel_m"] = float(self._obj.coords["m"].values[m_idx])
            da.attrs["l_idx"] = l_idx
            da.attrs["m_idx"] = m_idx
            da.attrs["pol"] = pol

            return da

        # Per-time tracking path (ra/dec)
        # Do NOT sortby("time") here — _compute_pixel_track returns
        # positional indices matching the original dataset time order.
        # Sorting data_var would misalign positional indices with data.
        l_indices, m_indices, visible = result
        data_var = self._obj[var].isel(polarization=pol)

        n_times = self._obj.sizes["time"]
        n_freqs = self._obj.sizes["frequency"]

        time_coords = self._obj.coords["time"].values
        freq_coords = self._obj.coords["frequency"].values

        # Build output array, NaN-filled
        out = np.full((n_times, n_freqs), np.nan)

        # Extract per-time pixel values.
        # Each time step may map to a different (l, m) pixel, so we
        # select the exact pixel per time step rather than loading the
        # full spatial grid (which would be e.g. 4096x4096 per step).
        vis_mask = visible
        if np.any(vis_mask):
            vis_times = np.where(vis_mask)[0]
            vis_l = l_indices[vis_mask]
            vis_m = m_indices[vis_mask]

            # Build per-time-step pixel selections (each is shape n_freqs)
            pixel_arrays = [
                data_var.isel(time=int(t), l=int(li), m=int(mi))
                for t, li, mi in zip(vis_times, vis_l, vis_m)
            ]

            # Compute all pixels in one pass — dask deduplicates
            # shared chunk reads automatically.
            if hasattr(data_var, "chunks") and data_var.chunks is not None:
                import dask

                with _dask_progress("Extracting tracked pixels"):
                    results = dask.compute(*pixel_arrays)
            else:
                results = [p.values for p in pixel_arrays]

            for i, ti in enumerate(vis_times):
                out[ti] = np.asarray(results[i])

        da = xr.DataArray(
            out,
            dims=["time", "frequency"],
            coords={"time": time_coords, "frequency": freq_coords},
            attrs={
                "pixel_l": "tracked",
                "pixel_m": "tracked",
                "l_idx": "tracked",
                "m_idx": "tracked",
                "pol": pol,
                "ra": ra,
                "dec": dec,
                "tracking": True,
            },
        )

        return da

    def plot_dynamic_spectrum(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (8, 5),
        add_colorbar: bool = True,
        observatory: Any = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot a dynamic spectrum (time vs frequency) for a single pixel.

        Creates a 2D visualization showing intensity variations across
        time and frequency at a specified location.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Target l direction cosine coordinate. Requires ``m``.
        m : float, optional
            Target m direction cosine coordinate. Requires ``l``.
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
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the dynamic spectrum plot.

        Examples
        --------
        >>> fig = ds.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
        >>> fig = ds.radport.plot_dynamic_spectrum(ra=180.0, dec=45.0)
        """
        # Get dynamic spectrum
        dynspec = self.dynamic_spectrum(
            ra=ra, dec=dec, l=l, m=m, var=var, pol=pol, observatory=observatory
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
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")

        if dynspec.attrs.get("tracking"):
            ra_val = dynspec.attrs["ra"]
            dec_val = dynspec.attrs["dec"]
            ax.set_title(
                f"{var} Dynamic Spectrum (tracked) at RA={ra_val:.4f}°, "
                f"Dec={dec_val:.4f}°, pol={pol}"
            )
        else:
            pixel_l = dynspec.attrs["pixel_l"]
            pixel_m = dynspec.attrs["pixel_m"]
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
            float(l_vals[0]), float(l_vals[-1]),
            float(m_vals[0]), float(m_vals[-1]),
        ]

        # Plot — transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
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
                float(l_vals[0]), float(l_vals[-1]),
                float(m_vals[0]), float(m_vals[-1]),
            ]

            # Check if panel has data
            has_data = np.any(np.isfinite(data))

            if has_data:
                im = ax.imshow(
                    data.T,
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
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        observatory: Any = None,
    ) -> xr.DataArray:
        """Extract a light curve (time series) at a specific spatial location.

        Returns intensity as a function of time at the pixel nearest to
        the specified coordinates and frequency.

        When celestial coordinates (ra, dec) are provided, the pixel is
        tracked across time steps as the source drifts due to Earth rotation.
        Time steps where the source is below the horizon are NaN-filled.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.

        Returns
        -------
        xr.DataArray
            1D array with dimension 'time' containing the light curve.

        Examples
        --------
        >>> lc = ds.radport.light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        >>> lc = ds.radport.light_curve(ra=180.0, dec=45.0, freq_mhz=50.0)
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

        result = self._resolve_coordinates(
            ra=ra, dec=dec, l=l, m=m, observatory=observatory
        )

        freq_hz = float(self._obj.coords["frequency"].values[fi])

        if isinstance(result, tuple) and len(result) == 2:
            # Fixed pixel path (l/m) — eagerly load small results
            l_idx, m_idx = result
            lc = _maybe_load(
                self._obj[var].isel(
                    frequency=fi, polarization=pol, l=l_idx, m=m_idx
                )
            )

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

        # Per-time tracking path (ra/dec)
        l_indices, m_indices, visible = result
        data_var = self._obj[var].isel(frequency=fi, polarization=pol)

        n_times = self._obj.sizes["time"]
        out = np.full(n_times, np.nan)

        vis_mask = visible
        if np.any(vis_mask):
            vis_times = np.where(vis_mask)[0]
            vis_l = l_indices[vis_mask]
            vis_m = m_indices[vis_mask]

            # Select the exact pixel per time step rather than loading
            # the full spatial grid (which can be e.g. 4096x4096).
            pixel_arrays = [
                data_var.isel(time=int(t), l=int(li), m=int(mi))
                for t, li, mi in zip(vis_times, vis_l, vis_m)
            ]

            if hasattr(data_var, "chunks") and data_var.chunks is not None:
                import dask

                with _dask_progress("Extracting tracked pixels"):
                    results = dask.compute(*pixel_arrays)
            else:
                results = [p.values for p in pixel_arrays]

            for i, ti in enumerate(vis_times):
                out[ti] = float(results[i])

        time_coords = self._obj.coords["time"].values
        lc = xr.DataArray(
            out,
            dims=["time"],
            coords={"time": time_coords},
            attrs={
                "variable": var,
                "freq_idx": fi,
                "freq_mhz": freq_hz / 1e6,
                "pol": pol,
                "l": "tracked",
                "m": "tracked",
                "l_idx": "tracked",
                "m_idx": "tracked",
                "ra": ra,
                "dec": dec,
                "tracking": True,
            },
        )

        return lc

    def plot_light_curve(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        figsize: tuple[float, float] = (10, 4),
        marker: str = "o",
        linestyle: str = "-",
        observatory: Any = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot a light curve (time series) at a specific spatial location.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
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
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.
        **kwargs : dict
            Additional arguments passed to plt.plot.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        >>> fig = ds.radport.plot_light_curve(ra=180.0, dec=45.0, freq_mhz=50.0)
        """
        lc = self.light_curve(
            ra=ra, dec=dec, l=l, m=m,
            freq_idx=freq_idx, freq_mhz=freq_mhz, var=var, pol=pol,
            observatory=observatory,
        )

        fig, ax = plt.subplots(figsize=figsize)

        time_vals = lc.coords["time"].values
        ax.plot(time_vals, lc.values, marker=marker, linestyle=linestyle, **kwargs)

        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel(f"{var} Intensity (Jy/beam)")

        freq_mhz_val = lc.attrs["freq_mhz"]
        if lc.attrs.get("tracking"):
            ra_val = lc.attrs["ra"]
            dec_val = lc.attrs["dec"]
            ax.set_title(
                f"{var} Light Curve (tracked) at RA={ra_val:.4f}°, "
                f"Dec={dec_val:.4f}°, f={freq_mhz_val:.2f} MHz, pol={pol}"
            )
        else:
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
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.DataArray:
        """Extract a frequency spectrum at a specific spatial location and time.

        Returns intensity as a function of frequency at the pixel nearest to
        the specified coordinates and time.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
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
        >>> spec = ds.radport.spectrum(ra=180.0, dec=45.0, time_idx=0)
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

        # Resolve coordinates
        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            l_idx, m_idx = self.coords_to_pixel(ra, dec, time_idx=ti)
        elif l is not None or m is not None:
            if l is None or m is None:
                raise ValueError("Both l and m must be provided together.")
            l_idx, m_idx = self.nearest_lm_idx(l, m)
        else:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

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
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
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
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
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
        >>> fig = ds.radport.plot_spectrum(ra=180.0, dec=45.0, time_idx=0)
        """
        spec = self.spectrum(
            ra=ra, dec=dec, l=l, m=m,
            time_idx=time_idx, time_mjd=time_mjd, var=var, pol=pol,
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
            float(l_vals[0]), float(l_vals[-1]),
            float(m_vals[0]), float(m_vals[-1]),
        ]

        # Transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
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
            float(l_vals[0]), float(l_vals[-1]),
            float(m_vals[0]), float(m_vals[-1]),
        ]

        # Transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
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
        *,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        observatory: Any = None,
    ) -> tuple[int, int]:
        """Convert celestial coordinates (RA, Dec) to pixel indices.

        A fixed (RA, Dec) maps to different pixel positions at different
        times due to Earth rotation. When ``time_idx`` or ``time_mjd`` is
        provided, the conversion uses the SIN projection at that specific
        time step. Without a time argument, the static WCS header is used
        (valid only for the reference time of the dataset).

        Parameters
        ----------
        ra : float
            Right Ascension in degrees (FK5/J2000).
        dec : float
            Declination in degrees (FK5/J2000).
        time_idx : int, optional
            Time index for time-aware conversion.
        time_mjd : float, optional
            MJD value for time-aware conversion (overrides ``time_idx``).
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location. Defaults to OVRO-LWA.

        Returns
        -------
        tuple of int
            (l_idx, m_idx) pixel indices (rounded to nearest integer).

        Raises
        ------
        ValueError
            If WCS is not available (static mode), coordinates are outside
            the image, or the source is below the horizon at the given time.

        Example
        -------
        >>> l_idx, m_idx = ds.radport.coords_to_pixel(180.0, 45.0)
        >>> l_idx, m_idx = ds.radport.coords_to_pixel(180.0, 45.0, time_idx=10)
        """
        # Time-aware path: compute LST for just the single requested
        # timestamp rather than the full time axis (~10x faster).
        if time_idx is not None or time_mjd is not None:
            if time_mjd is not None:
                time_idx = self.nearest_time_idx(time_mjd)

            return self._compute_pixel_at_time(
                ra, dec, time_idx, observatory=observatory
            )

        # Static WCS path (original behavior)
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

        # WCS returns NaN for coordinates outside the projection domain
        # (e.g., sources below the horizon or outside the SIN projection)
        if np.isnan(x) or np.isnan(y):
            # Compute angular distance from phase center to give context
            crval1 = wcs.wcs.crval[0]
            crval2 = wcs.wcs.crval[1]
            phase_center = SkyCoord(ra=crval1 * u.deg, dec=crval2 * u.deg, frame="fk5")
            sep = coord.separation(phase_center).deg
            raise ValueError(
                f"Source (RA={ra}°, Dec={dec}°) is outside the visible sky "
                f"for this dataset's SIN projection. The source is {sep:.1f}° "
                f"from the phase center (RA={crval1:.1f}°, Dec={crval2:.1f}°). "
                f"The SIN projection covers a hemisphere centered on the phase "
                f"center — sources beyond ~90° cannot be mapped to pixels."
            )

        l_idx = int(round(float(x)))
        m_idx = int(round(float(y)))

        # Validate result is within bounds
        n_l = self._obj.sizes["l"]
        n_m = self._obj.sizes["m"]
        if not (0 <= l_idx < n_l) or not (0 <= m_idx < n_m):
            raise ValueError(
                f"Coordinates (RA={ra}°, Dec={dec}°) map to pixel "
                f"({l_idx}, {m_idx}) which is outside the image bounds "
                f"[0, {n_l}) x [0, {n_m}). The source may be in the "
                f"visible sky but outside the field of view captured "
                f"by this dataset."
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
            - ra: Right Ascension in degrees (None if WCS unavailable
              or pixel is outside the projection domain)
            - dec: Declination in degrees (None if WCS unavailable)

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
            peak: dict[str, Any] = {
                "l": float(l_coords[l_idx]),
                "m": float(m_coords[m_idx]),
                "l_idx": int(l_idx),
                "m_idx": int(m_idx),
                "flux": float(signal[l_idx, m_idx]),
                "snr": float(snr[l_idx, m_idx]),
            }
            # Always include ra/dec keys for consistent caller interface.
            # Set to None when WCS is unavailable or conversion fails
            # (e.g., edge pixels outside the SIN projection domain).
            peak["ra"] = None
            peak["dec"] = None
            if self.has_wcs:
                try:
                    ra_val, dec_val = self.pixel_to_coords(int(l_idx), int(m_idx))
                    # WCS returns NaN for pixels outside the SIN
                    # projection domain (near l²+m²≈1).  Keep None.
                    if np.isfinite(ra_val) and np.isfinite(dec_val):
                        peak["ra"] = ra_val
                        peak["dec"] = dec_val
                except (ValueError, ImportError):
                    pass
            peaks.append(peak)

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
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
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
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            The l coordinate value for the pixel location. Requires ``m``.
        m : float, optional
            The m coordinate value for the pixel location. Requires ``l``.
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

        # Resolve coordinates
        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            l_idx, m_idx = self.coords_to_pixel(ra, dec, time_idx=time_idx)
        elif l is not None or m is not None:
            if l is None or m is None:
                raise ValueError("Both l and m must be provided together.")
            l_idx, m_idx = self.nearest_lm_idx(l, m)
        else:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

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
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
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
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            The l coordinate value for the pixel location. Requires ``m``.
        m : float, optional
            The m coordinate value for the pixel location. Requires ``l``.
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

        # Resolve coordinates
        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            l_idx, m_idx = self.coords_to_pixel(ra, dec, time_idx=time_idx)
        elif l is not None or m is not None:
            if l is None or m is None:
                raise ValueError("Both l and m must be provided together.")
            l_idx, m_idx = self.nearest_lm_idx(l, m)
        else:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

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
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        dm: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        method: Literal["shift", "interpolate"] = "shift",
        fill_value: float = np.nan,
        trim: bool = False,
        observatory: Any = None,
    ) -> xr.DataArray:
        """Extract a dedispersed dynamic spectrum for a single pixel.

        Corrects for interstellar dispersion by shifting or interpolating
        frequency channels according to the dispersion delay. This is essential
        for analyzing dispersed radio transients like pulsars and FRBs.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Target l direction cosine coordinate. Requires ``m``.
        m : float, optional
            Target m direction cosine coordinate. Requires ``l``.
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
        dynspec = self.dynamic_spectrum(
            ra=ra, dec=dec, l=l, m=m, var=var, pol=pol, observatory=observatory
        )

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
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
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
        observatory: Any = None,
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
            ra=ra, dec=dec, l=l, m=m, dm=dm, var=var, pol=pol,
            method=method, trim=trim, observatory=observatory,
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
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")

        if dynspec.attrs.get("tracking"):
            ra_val = dynspec.attrs["ra"]
            dec_val = dynspec.attrs["dec"]
            ax.set_title(
                f"{var} Dedispersed Dynamic Spectrum (tracked)\n"
                f"RA={ra_val:.4f}°, Dec={dec_val:.4f}°, pol={pol}, "
                f"DM={dm:.2f} pc/cm³ ({method})"
            )
        else:
            pixel_l = dynspec.attrs["pixel_l"]
            pixel_m = dynspec.attrs["pixel_m"]
            ax.set_title(
                f"{var} Dedispersed Dynamic Spectrum\n"
                f"l={pixel_l:+.4f}, m={pixel_m:+.4f}, pol={pol}, "
                f"DM={dm:.2f} pc/cm³ ({method})"
            )

        fig.tight_layout()
        return fig
