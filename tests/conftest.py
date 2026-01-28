"""Shared pytest fixtures for ovro_lwa_portal tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def valid_ovro_dataset() -> xr.Dataset:
    """Create a valid OVRO-LWA dataset with all required dimensions and variables.

    Returns
    -------
    xr.Dataset
        A dataset with SKY variable and all required dimensions:
        time, frequency, polarization, l, m.
    """
    np.random.seed(42)
    return xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 3, 2, 50, 50) * 10,
            ),
        },
        coords={
            "time": [60000.0, 60000.1],  # MJD values
            "frequency": [46e6, 50e6, 54e6],  # Hz
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )


@pytest.fixture
def valid_ovro_dataset_with_beam() -> xr.Dataset:
    """Create a valid OVRO-LWA dataset with both SKY and BEAM variables.

    Returns
    -------
    xr.Dataset
        A dataset with SKY and BEAM variables and all required dimensions.
    """
    np.random.seed(42)
    shape = (2, 3, 2, 50, 50)
    dims = ["time", "frequency", "polarization", "l", "m"]
    return xr.Dataset(
        data_vars={
            "SKY": (dims, np.random.rand(*shape) * 10),
            "BEAM": (dims, np.random.rand(*shape)),
        },
        coords={
            "time": [60000.0, 60000.1],
            "frequency": [46e6, 50e6, 54e6],
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )


@pytest.fixture
def dataset_missing_dimensions() -> xr.Dataset:
    """Create a dataset missing required OVRO-LWA dimensions.

    Returns
    -------
    xr.Dataset
        A dataset with SKY variable but wrong dimensions (x, y instead of l, m).
    """
    return xr.Dataset(
        data_vars={
            "SKY": (["x", "y"], np.random.rand(10, 10)),
        },
        coords={
            "x": np.arange(10),
            "y": np.arange(10),
        },
    )


@pytest.fixture
def dataset_missing_sky_variable() -> xr.Dataset:
    """Create a dataset with correct dimensions but missing SKY variable.

    Returns
    -------
    xr.Dataset
        A dataset with all required dimensions but OTHER variable instead of SKY.
    """
    return xr.Dataset(
        data_vars={
            "OTHER": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 3, 2, 10, 10),
            ),
        },
        coords={
            "time": [0, 1],
            "frequency": [1e6, 2e6, 3e6],
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 10),
            "m": np.linspace(-1, 1, 10),
        },
    )


@pytest.fixture
def valid_ovro_dataset_with_wcs() -> xr.Dataset:
    """Create a valid OVRO-LWA dataset with WCS coordinate information.

    Returns
    -------
    xr.Dataset
        A dataset with SKY variable and WCS header for coordinate transforms.
    """
    np.random.seed(42)

    # Simple FITS WCS header for a 50x50 image centered at RA=180, Dec=45
    wcs_header = """NAXIS   =                    2
NAXIS1  =                   50
NAXIS2  =                   50
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CRPIX1  =                 25.0
CRPIX2  =                 25.0
CRVAL1  =                180.0
CRVAL2  =                 45.0
CDELT1  =                 -1.0
CDELT2  =                  1.0
CUNIT1  = 'deg'
CUNIT2  = 'deg'
RADESYS = 'FK5'
EQUINOX =               2000.0"""

    ds = xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 3, 2, 50, 50) * 10,
            ),
        },
        coords={
            "time": [60000.0, 60000.1],
            "frequency": [46e6, 50e6, 54e6],
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )

    # Add WCS header to variable attrs
    ds["SKY"].attrs["fits_wcs_header"] = wcs_header

    return ds


@pytest.fixture
def variable_source_dataset() -> xr.Dataset:
    """Create a dataset with a synthetic variable source for testing.

    This fixture creates a dataset with:
    - 10 time steps
    - 5 frequency channels
    - 50x50 spatial grid
    - A sinusoidally varying source at the center
    - Background noise

    Returns
    -------
    xr.Dataset
        A dataset with a synthetic variable source injected at the center.
    """
    np.random.seed(42)

    n_times = 10
    n_freqs = 5
    n_pols = 2
    n_l = 50
    n_m = 50

    # Create base noise
    data = np.random.rand(n_times, n_freqs, n_pols, n_l, n_m) * 0.5

    # Create coordinate grids
    l_vals = np.linspace(-1, 1, n_l)
    m_vals = np.linspace(-1, 1, n_m)
    ll, mm = np.meshgrid(l_vals, m_vals, indexing="ij")

    # Inject a variable source at center (l=0, m=0)
    # Gaussian spatial profile
    source_l, source_m = 0.0, 0.0
    sigma = 0.1
    spatial_profile = np.exp(-((ll - source_l) ** 2 + (mm - source_m) ** 2) / (2 * sigma**2))

    # Time-varying amplitude (sinusoidal)
    time_vals = np.linspace(60000.0, 60000.9, n_times)
    time_amplitude = 5.0 + 3.0 * np.sin(2 * np.pi * np.arange(n_times) / n_times)

    # Frequency dependence (slight power law)
    freq_vals = np.array([46e6, 48e6, 50e6, 52e6, 54e6])
    freq_factor = (freq_vals / 50e6) ** (-0.7)

    # Inject source into data
    for t in range(n_times):
        for f in range(n_freqs):
            for p in range(n_pols):
                data[t, f, p, :, :] += time_amplitude[t] * freq_factor[f] * spatial_profile

    return xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                data,
            ),
        },
        coords={
            "time": time_vals,
            "frequency": freq_vals,
            "polarization": [0, 1],
            "l": l_vals,
            "m": m_vals,
        },
    )


@pytest.fixture
def sliding_window_dataset() -> xr.Dataset:
    """Create a dataset suitable for sliding window analysis testing.

    This fixture creates a dataset with sufficient dimensions for
    sliding window operations (more time steps and frequencies).

    Returns
    -------
    xr.Dataset
        A dataset with 10 time steps, 8 frequencies, and 30x30 spatial grid.
    """
    np.random.seed(42)

    n_times = 10
    n_freqs = 8
    n_pols = 2
    n_l = 30
    n_m = 30

    data = np.random.rand(n_times, n_freqs, n_pols, n_l, n_m) * 10

    return xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                data,
            ),
        },
        coords={
            "time": np.linspace(60000.0, 60000.9, n_times),
            "frequency": np.linspace(46e6, 54e6, n_freqs),
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, n_l),
            "m": np.linspace(-1, 1, n_m),
        },
    )
