"""Shared fixtures for visualization tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pn = pytest.importorskip("panel")
hv = pytest.importorskip("holoviews")


@pytest.fixture
def viz_dataset() -> xr.Dataset:
    """Create a small OVRO-LWA dataset for visualization tests.

    Same structure as the main conftest ``valid_ovro_dataset`` fixture
    but kept here so viz tests are self-contained.
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
            "time": [60000.0, 60000.1],
            "frequency": [46e6, 50e6, 54e6],
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )


@pytest.fixture
def viz_dataset_with_beam() -> xr.Dataset:
    """Create a dataset with both SKY and BEAM for testing variable selection."""
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
