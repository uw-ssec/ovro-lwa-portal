"""Integration tests for the io module with real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ovro_lwa_portal import open_dataset


@pytest.fixture
def sample_zarr_store(tmp_path: Path) -> Path:
    """Create a sample OVRO-LWA-like zarr store for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Returns
    -------
    Path
        Path to the created zarr store.
    """
    zarr_path = tmp_path / "sample_observation.zarr"

    # Create a realistic OVRO-LWA dataset structure
    time = np.arange(10)
    frequency = np.linspace(40e6, 80e6, 20)  # 40-80 MHz
    l = np.arange(512)
    m = np.arange(512)

    ds = xr.Dataset(
        {
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(10, 20, 1, 512, 512).astype(np.float32),
            ),
            "BEAM": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(10, 20, 1, 512, 512).astype(np.float32),
            ),
        },
        coords={
            "time": time,
            "frequency": frequency,
            "polarization": [0],
            "l": l,
            "m": m,
            "right_ascension": (["m", "l"], np.random.rand(512, 512) * 360),
            "declination": (["m", "l"], np.random.rand(512, 512) * 180 - 90),
        },
        attrs={
            "instrument": "OVRO-LWA",
            "observation_date": "2024-05-24",
        },
    )

    # Add WCS header string (simplified)
    wcs_header = "WCSAXES = 2\nCTYPE1  = 'RA---SIN'\nCTYPE2  = 'DEC--SIN'\n"
    ds.attrs["fits_wcs_header"] = wcs_header

    # Save to zarr
    ds.to_zarr(zarr_path, mode="w")

    return zarr_path


class TestOpenDatasetIntegration:
    """Integration tests for open_dataset with realistic data."""

    def test_load_sample_dataset(self, sample_zarr_store: Path) -> None:
        """Test loading a sample OVRO-LWA dataset."""
        ds = open_dataset(sample_zarr_store)

        assert isinstance(ds, xr.Dataset)
        assert "SKY" in ds.data_vars
        assert "BEAM" in ds.data_vars
        assert set(ds.dims.keys()) == {"time", "frequency", "polarization", "l", "m"}

    def test_load_with_validation(self, sample_zarr_store: Path) -> None:
        """Test loading with validation enabled."""
        ds = open_dataset(sample_zarr_store, validate=True)

        assert isinstance(ds, xr.Dataset)
        # Should pass validation without warnings for this dataset

    def test_load_without_validation(self, sample_zarr_store: Path) -> None:
        """Test loading without validation."""
        ds = open_dataset(sample_zarr_store, validate=False)

        assert isinstance(ds, xr.Dataset)

    def test_load_with_custom_chunks(self, sample_zarr_store: Path) -> None:
        """Test loading with custom chunking."""
        ds = open_dataset(
            sample_zarr_store,
            chunks={"time": 5, "frequency": 10, "l": 256, "m": 256},
        )

        assert isinstance(ds, xr.Dataset)
        # Verify chunking
        assert hasattr(ds["SKY"].data, "chunks")

    def test_load_with_auto_chunks(self, sample_zarr_store: Path) -> None:
        """Test loading with automatic chunking."""
        ds = open_dataset(sample_zarr_store, chunks="auto")

        assert isinstance(ds, xr.Dataset)
        assert hasattr(ds["SKY"].data, "chunks")

    def test_load_without_chunks(self, sample_zarr_store: Path) -> None:
        """Test loading entire dataset into memory."""
        ds = open_dataset(sample_zarr_store, chunks=None)

        assert isinstance(ds, xr.Dataset)
        # Data should be numpy array
        assert isinstance(ds["SKY"].data, np.ndarray)

    def test_dataset_attributes_preserved(self, sample_zarr_store: Path) -> None:
        """Test that dataset attributes are preserved."""
        ds = open_dataset(sample_zarr_store, validate=False)

        assert "instrument" in ds.attrs
        assert ds.attrs["instrument"] == "OVRO-LWA"
        assert "fits_wcs_header" in ds.attrs

    def test_coordinates_preserved(self, sample_zarr_store: Path) -> None:
        """Test that coordinates are preserved."""
        ds = open_dataset(sample_zarr_store, validate=False)

        assert "right_ascension" in ds.coords
        assert "declination" in ds.coords
        assert "time" in ds.coords
        assert "frequency" in ds.coords

    def test_data_values_preserved(self, sample_zarr_store: Path) -> None:
        """Test that data values are correctly loaded."""
        # Load original dataset
        ds_original = xr.open_zarr(sample_zarr_store)

        # Load via open_dataset
        ds_loaded = open_dataset(sample_zarr_store, validate=False, chunks=None)

        # Compare values
        np.testing.assert_array_equal(
            ds_original["SKY"].values,
            ds_loaded["SKY"].values,
        )

    def test_subset_selection(self, sample_zarr_store: Path) -> None:
        """Test selecting subsets of data."""
        ds = open_dataset(sample_zarr_store)

        # Select subset
        subset = ds.sel(time=slice(0, 5), frequency=slice(40e6, 60e6))

        assert len(subset.time) <= 5
        assert subset.frequency.min() >= 40e6
        assert subset.frequency.max() <= 60e6

    def test_computation_on_loaded_data(self, sample_zarr_store: Path) -> None:
        """Test performing computations on loaded data."""
        ds = open_dataset(sample_zarr_store)

        # Compute mean
        mean_sky = ds.SKY.mean(dim=["l", "m"]).compute()

        assert mean_sky.shape == (10, 20, 1)
        assert not np.isnan(mean_sky.values).any()

    def test_multiple_loads_same_store(self, sample_zarr_store: Path) -> None:
        """Test loading the same store multiple times."""
        ds1 = open_dataset(sample_zarr_store, validate=False)
        ds2 = open_dataset(sample_zarr_store, validate=False)

        # Both should load successfully
        assert isinstance(ds1, xr.Dataset)
        assert isinstance(ds2, xr.Dataset)

        # Values should be identical
        np.testing.assert_array_equal(
            ds1["SKY"].values,
            ds2["SKY"].values,
        )

    def test_pathlib_path_input(self, sample_zarr_store: Path) -> None:
        """Test that Path objects work as input."""
        # sample_zarr_store is already a Path object
        ds = open_dataset(sample_zarr_store)

        assert isinstance(ds, xr.Dataset)

    def test_string_path_input(self, sample_zarr_store: Path) -> None:
        """Test that string paths work as input."""
        ds = open_dataset(str(sample_zarr_store))

        assert isinstance(ds, xr.Dataset)
