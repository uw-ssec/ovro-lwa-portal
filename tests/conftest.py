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
