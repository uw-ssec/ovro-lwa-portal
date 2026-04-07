"""Data bridge: convert accessor outputs to HoloViews elements.

This module encapsulates all OVRO-LWA coordinate conventions (transposition,
extent ordering, robust scaling) so that explorer classes work with clean
HoloViews elements without knowing the underlying data layout.

Performance strategy: :class:`PreloadedCube` loads a single spatial slice
at a time (one time×freq pair) with strided downsampling, and caches
recently accessed slices in an LRU cache. This means:

- First access of a slice: one S3 chunk read (~64 MB), then stride → ~1 MB
- Subsequent access of the same slice: instant (cache hit)
- Slider changes to time/freq: one new slice fetch, old ones stay cached
- l/m changes (dynamic spectrum, light curve): instant (reads from cache)
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import holoviews as hv
    import xarray as xr


def _ensure_extension() -> None:
    """Load the HoloViews bokeh extension if not already loaded."""
    import holoviews as hv

    if not hv.Store.renderers:
        hv.extension("bokeh")


# Maximum pixels per side for interactive display.
_MAX_DISPLAY_SIZE = 512

# Number of slices to keep in the LRU cache.
_CACHE_SIZE = 32


class PreloadedCube:
    """Cached, spatially downsampled accessor for OVRO-LWA datasets.

    Instead of loading the full data cube into memory, this class loads
    individual 2D slices on demand with strided downsampling and caches
    them. For the dynamic spectrum (all times x freqs at one pixel),
    it precomputes a small downsampled cube lazily.

    Parameters
    ----------
    ds : xr.Dataset
        OVRO-LWA dataset (may be dask-backed / remote).
    var : str
        Data variable to use (default ``"SKY"``).
    pol : int
        Polarization index.
    max_size : int
        Maximum spatial dimension size after downsampling.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        var: str = "SKY",
        pol: int = 0,
        max_size: int = _MAX_DISPLAY_SIZE,
    ) -> None:
        self._ds = ds
        self.var = var
        self.pol = pol
        self._max_size = max_size

        # Compute stride factors
        n_l = ds.sizes["l"]
        n_m = ds.sizes["m"]
        self.stride_l = max(1, -(-n_l // max_size))
        self.stride_m = max(1, -(-n_m // max_size))

        # Cache coordinate arrays (strided to match display resolution)
        self.l_vals = ds.coords["l"].values[::self.stride_l]
        self.m_vals = ds.coords["m"].values[::self.stride_m]
        self.time_vals = ds.coords["time"].values
        self.freq_vals = ds.coords["frequency"].values
        self.freq_mhz = self.freq_vals / 1e6
        self.n_times = len(self.time_vals)
        self.n_freqs = len(self.freq_vals)

        # Dynamic spectrum cache (lazily computed)
        self._dynspec_cache: dict[tuple[int, int], np.ndarray] = {}

        # Per-instance LRU cache for _load_slice (avoids caching self
        # in a module-level lru_cache, which would pin the instance and
        # prevent garbage collection).
        self._load_slice = lru_cache(maxsize=_CACHE_SIZE)(self._load_slice_impl)

        print(  # noqa: T201
            f"PreloadedCube ready: {var} "
            f"({self.n_times}t x {self.n_freqs}f, "
            f"display {len(self.l_vals)}x{len(self.m_vals)} "
            f"from {n_l}x{n_m}, stride {self.stride_l}x{self.stride_m})"
        )

    def _load_slice_impl(self, time_idx: int, freq_idx: int) -> np.ndarray:
        """Load and downsample a single (l, m) slice.

        Returns
        -------
        np.ndarray
            2D array of shape (n_l_display, n_m_display), float32.
        """
        da = self._ds[self.var].isel(
            time=time_idx, frequency=freq_idx, polarization=self.pol
        )
        # Stride-based downsampling — only reads the strided elements,
        # which for contiguous chunks still reads the full chunk but
        # the resulting array is small and fast to work with.
        data = da.values[::self.stride_l, ::self.stride_m]
        return data.astype(np.float32, copy=False)

    def image(self, time_idx: int = 0, freq_idx: int = 0) -> np.ndarray:
        """Get a 2D image slice, transposed for display (m, l) → (y, x)."""
        return self._load_slice(time_idx, freq_idx).T

    def dynamic_spectrum(self, l_idx: int, m_idx: int) -> np.ndarray:
        """Get a 2D (time, freq) dynamic spectrum at a display pixel.

        Loads all slices not yet in cache, then extracts the pixel.
        """
        key = (l_idx, m_idx)
        if key not in self._dynspec_cache:
            out = np.empty((self.n_times, self.n_freqs), dtype=np.float32)
            for ti in range(self.n_times):
                for fi in range(self.n_freqs):
                    slc = self._load_slice(ti, fi)
                    out[ti, fi] = slc[l_idx, m_idx]
            self._dynspec_cache[key] = out
        return self._dynspec_cache[key]

    def light_curve(self, l_idx: int, m_idx: int, freq_idx: int) -> np.ndarray:
        """Get a 1D time series at a display pixel and frequency."""
        out = np.empty(self.n_times, dtype=np.float32)
        for ti in range(self.n_times):
            slc = self._load_slice(ti, freq_idx)
            out[ti] = slc[l_idx, m_idx]
        return out

    def spectrum(self, l_idx: int, m_idx: int, time_idx: int) -> np.ndarray:
        """Get a 1D frequency spectrum at a display pixel and time."""
        out = np.empty(self.n_freqs, dtype=np.float32)
        for fi in range(self.n_freqs):
            slc = self._load_slice(time_idx, fi)
            out[fi] = slc[l_idx, m_idx]
        return out

    def nearest_lm_idx(self, l: float, m: float) -> tuple[int, int]:
        """Find nearest display pixel indices for given l, m values."""
        l_idx = int(np.argmin(np.abs(self.l_vals - l)))
        m_idx = int(np.argmin(np.abs(self.m_vals - m)))
        return l_idx, m_idx

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Image bounds as (l_left, m_bottom, l_right, m_top)."""
        return (
            float(self.l_vals[0]),
            float(self.m_vals[0]),
            float(self.l_vals[-1]),
            float(self.m_vals[-1]),
        )


def sky_image_element(
    cube: PreloadedCube,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    robust: bool = True,
    mask_radius: int | None = None,
) -> hv.Image:
    """Create an hv.Image from the cached cube.

    Parameters
    ----------
    cube : PreloadedCube
        Data cube.
    time_idx, freq_idx : int
        Slice indices.
    robust : bool
        Use 2nd/98th percentile for color limits.
    mask_radius : int, optional
        Circular mask radius in pixels.

    Returns
    -------
    hv.Image
    """
    import holoviews as hv

    _ensure_extension()

    data = cube.image(time_idx, freq_idx)

    if mask_radius is not None:
        data = data.copy()
        ny, nx = data.shape
        cy, cx = ny // 2, nx // 2
        yy, xx = np.ogrid[:ny, :nx]
        distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        data[distance > mask_radius] = np.nan

    time_val = float(cube.time_vals[time_idx])
    freq_mhz = float(cube.freq_mhz[freq_idx])

    img = hv.Image(data, kdims=["l", "m"], bounds=cube.bounds).opts(
        xlabel="l (direction cosine)",
        ylabel="m (direction cosine)",
        title=f"{cube.var} t={time_val:.4f} MJD, f={freq_mhz:.1f} MHz, pol={cube.pol}",
        aspect="equal",
        colorbar=True,
        clabel="Jy/beam",
    )

    if robust:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            img = img.opts(
                clim=(float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
            )

    return img


def dynamic_spectrum_element(
    cube: PreloadedCube,
    *,
    l: float = 0.0,
    m: float = 0.0,
    robust: bool = True,
) -> hv.Image:
    """Create an hv.Image of a dynamic spectrum from the cached cube.

    Parameters
    ----------
    cube : PreloadedCube
        Data cube.
    l, m : float
        Direction cosine coordinates.
    robust : bool
        Use 2nd/98th percentile for color limits.

    Returns
    -------
    hv.Image
    """
    import holoviews as hv

    _ensure_extension()

    l_idx, m_idx = cube.nearest_lm_idx(l, m)
    data = cube.dynamic_spectrum(l_idx, m_idx)

    pixel_l = float(cube.l_vals[l_idx])
    pixel_m = float(cube.m_vals[m_idx])

    bounds = (
        float(cube.time_vals[0]),
        float(cube.freq_mhz[0]),
        float(cube.time_vals[-1]),
        float(cube.freq_mhz[-1]),
    )

    img = hv.Image(
        data,
        kdims=["Time (MJD)", "Frequency (MHz)"],
        bounds=bounds,
    ).opts(
        xlabel="Time (MJD)",
        ylabel="Frequency (MHz)",
        title=f"{cube.var} Dynamic Spectrum at l={pixel_l:+.4f}, m={pixel_m:+.4f}",
        colorbar=True,
        clabel="Jy/beam",
    )

    if robust:
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            img = img.opts(
                clim=(float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
            )

    return img


def light_curve_element(
    cube: PreloadedCube,
    *,
    l: float = 0.0,
    m: float = 0.0,
    freq_idx: int = 0,
) -> hv.Curve:
    """Create an hv.Curve of a light curve from the cached cube.

    Parameters
    ----------
    cube : PreloadedCube
        Data cube.
    l, m : float
        Direction cosine coordinates.
    freq_idx : int
        Frequency index.

    Returns
    -------
    hv.Curve
    """
    import holoviews as hv

    _ensure_extension()

    l_idx, m_idx = cube.nearest_lm_idx(l, m)
    values = cube.light_curve(l_idx, m_idx, freq_idx)
    freq_mhz = float(cube.freq_mhz[freq_idx])

    return hv.Curve(
        (cube.time_vals, values),
        kdims=["Time (MJD)"],
        vdims=["Intensity (Jy/beam)"],
    ).opts(
        xlabel="Time (MJD)",
        ylabel="Intensity (Jy/beam)",
        title=f"Light Curve at f={freq_mhz:.1f} MHz",
    )


def spectrum_element(
    cube: PreloadedCube,
    *,
    l: float = 0.0,
    m: float = 0.0,
    time_idx: int = 0,
) -> hv.Curve:
    """Create an hv.Curve of a frequency spectrum from the cached cube.

    Parameters
    ----------
    cube : PreloadedCube
        Data cube.
    l, m : float
        Direction cosine coordinates.
    time_idx : int
        Time index.

    Returns
    -------
    hv.Curve
    """
    import holoviews as hv

    _ensure_extension()

    l_idx, m_idx = cube.nearest_lm_idx(l, m)
    values = cube.spectrum(l_idx, m_idx, time_idx)
    time_val = float(cube.time_vals[time_idx])

    return hv.Curve(
        (cube.freq_mhz, values),
        kdims=["Frequency (MHz)"],
        vdims=["Intensity (Jy/beam)"],
    ).opts(
        xlabel="Frequency (MHz)",
        ylabel="Intensity (Jy/beam)",
        title=f"Spectrum at t={time_val:.4f} MJD",
    )


def cutout_image_element(
    cube: PreloadedCube,
    *,
    l_center: float = 0.0,
    m_center: float = 0.0,
    dl: float = 0.1,
    dm: float = 0.1,
    time_idx: int = 0,
    freq_idx: int = 0,
    robust: bool = True,
) -> hv.Image:
    """Create an hv.Image of a spatial cutout from the cached cube.

    Parameters
    ----------
    cube : PreloadedCube
        Data cube.
    l_center, m_center : float
        Center coordinates.
    dl, dm : float
        Half-extent.
    time_idx, freq_idx : int
        Slice indices.
    robust : bool
        Use robust color scaling.

    Returns
    -------
    hv.Image
    """
    import holoviews as hv

    _ensure_extension()

    l_min, l_max = l_center - dl, l_center + dl
    m_min, m_max = m_center - dm, m_center + dm

    # Get the full display-resolution slice
    full = cube._load_slice(time_idx, freq_idx)
    l_vals = cube.l_vals
    m_vals = cube.m_vals

    # Mask for the cutout region (handles ascending or descending coords)
    lo_l, hi_l = min(l_min, l_max), max(l_min, l_max)
    lo_m, hi_m = min(m_min, m_max), max(m_min, m_max)
    l_mask = (l_vals >= lo_l) & (l_vals <= hi_l)
    m_mask = (m_vals >= lo_m) & (m_vals <= hi_m)

    if not np.any(l_mask) or not np.any(m_mask):
        return hv.Image(
            np.zeros((2, 2)), kdims=["l", "m"],
            bounds=(l_min, m_min, l_max, m_max),
        ).opts(title="Empty cutout — adjust center/extent")

    sub_data = full[np.ix_(l_mask, m_mask)].T
    sub_l = l_vals[l_mask]
    sub_m = m_vals[m_mask]
    bounds = (float(sub_l[0]), float(sub_m[0]), float(sub_l[-1]), float(sub_m[-1]))

    img = hv.Image(sub_data, kdims=["l", "m"], bounds=bounds).opts(
        xlabel="l (direction cosine)",
        ylabel="m (direction cosine)",
        title=f"{cube.var} Cutout at l={l_center:.4f}, m={m_center:.4f}",
        aspect="equal",
        colorbar=True,
        clabel="Jy/beam",
    )

    if robust:
        finite = sub_data[np.isfinite(sub_data)]
        if finite.size > 0:
            img = img.opts(
                clim=(float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
            )

    return img
