"""Tests for the radport xarray accessor."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

# Set non-interactive backend before importing accessor
matplotlib.use("Agg")

# Import to register the accessor
import ovro_lwa_portal  # noqa: F401
from ovro_lwa_portal.accessor import RadportAccessor


class TestRadportAccessorRegistration:
    """Tests for accessor registration and availability."""

    def test_accessor_available_after_import(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Accessor 'radport' is available on xarray Datasets after importing."""
        assert hasattr(valid_ovro_dataset, "radport")

    def test_accessor_returns_radport_accessor(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Accessor returns RadportAccessor instance."""
        assert isinstance(valid_ovro_dataset.radport, RadportAccessor)

    def test_accessor_cached_on_dataset(self, valid_ovro_dataset: xr.Dataset) -> None:
        """Accessor instance is cached (same object on repeated access)."""
        accessor1 = valid_ovro_dataset.radport
        accessor2 = valid_ovro_dataset.radport
        assert accessor1 is accessor2


class TestRadportValidation:
    """Tests for dataset validation during accessor initialization."""

    def test_valid_dataset_passes_validation(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Valid OVRO-LWA dataset passes validation without error."""
        # Should not raise
        _ = valid_ovro_dataset.radport

    def test_missing_dimensions_raises_value_error(
        self, dataset_missing_dimensions: xr.Dataset
    ) -> None:
        """Missing required dimensions raises ValueError with informative message."""
        with pytest.raises(ValueError, match="missing required dimensions"):
            _ = dataset_missing_dimensions.radport

    def test_missing_dimensions_lists_what_is_missing(
        self, dataset_missing_dimensions: xr.Dataset
    ) -> None:
        """Error message lists the specific missing dimensions."""
        with pytest.raises(ValueError) as exc_info:
            _ = dataset_missing_dimensions.radport

        error_msg = str(exc_info.value)
        # Should mention the missing dimensions
        assert "frequency" in error_msg
        assert "polarization" in error_msg
        assert "time" in error_msg
        assert "l" in error_msg
        assert "m" in error_msg

    def test_missing_sky_variable_raises_value_error(
        self, dataset_missing_sky_variable: xr.Dataset
    ) -> None:
        """Missing SKY variable raises ValueError with informative message."""
        with pytest.raises(ValueError, match="missing required variables"):
            _ = dataset_missing_sky_variable.radport

    def test_missing_sky_variable_lists_what_is_missing(
        self, dataset_missing_sky_variable: xr.Dataset
    ) -> None:
        """Error message lists the missing SKY variable."""
        with pytest.raises(ValueError) as exc_info:
            _ = dataset_missing_sky_variable.radport

        error_msg = str(exc_info.value)
        assert "SKY" in error_msg


class TestRadportHasBeam:
    """Tests for has_beam property."""

    def test_has_beam_false_when_no_beam(self, valid_ovro_dataset: xr.Dataset) -> None:
        """has_beam returns False when dataset has no BEAM variable."""
        assert not valid_ovro_dataset.radport.has_beam

    def test_has_beam_true_when_beam_present(
        self, valid_ovro_dataset_with_beam: xr.Dataset
    ) -> None:
        """has_beam returns True when dataset has BEAM variable."""
        assert valid_ovro_dataset_with_beam.radport.has_beam


class TestRadportPlot:
    """Tests for plot() method."""

    def test_plot_returns_figure(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() returns a matplotlib Figure object."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_default_parameters(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() works with default parameters."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            assert isinstance(fig, plt.Figure)
            # Should have one axes
            assert len(fig.axes) >= 1
        finally:
            plt.close(fig)

    def test_plot_custom_time_index(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom time_idx parameter."""
        fig = valid_ovro_dataset.radport.plot(time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_freq_index(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom freq_idx parameter."""
        fig = valid_ovro_dataset.radport.plot(freq_idx=2)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_polarization(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom pol parameter."""
        fig = valid_ovro_dataset.radport.plot(pol=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_colormap(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom cmap parameter."""
        fig = valid_ovro_dataset.radport.plot(cmap="viridis")
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_vmin_vmax(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts vmin and vmax parameters."""
        fig = valid_ovro_dataset.radport.plot(vmin=0.0, vmax=10.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_robust_scaling(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts robust parameter for percentile-based scaling."""
        fig = valid_ovro_dataset.radport.plot(robust=True)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_figsize(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom figsize parameter."""
        fig = valid_ovro_dataset.radport.plot(figsize=(10, 8))
        try:
            assert fig.get_figwidth() == 10.0
            assert fig.get_figheight() == 8.0
        finally:
            plt.close(fig)

    def test_plot_without_colorbar(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts add_colorbar=False."""
        fig = valid_ovro_dataset.radport.plot(add_colorbar=False)
        try:
            assert isinstance(fig, plt.Figure)
            # Should have only one axes (no colorbar)
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    def test_plot_with_colorbar(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() with add_colorbar=True adds colorbar."""
        fig = valid_ovro_dataset.radport.plot(add_colorbar=True)
        try:
            # Should have two axes (main plot + colorbar)
            assert len(fig.axes) == 2
        finally:
            plt.close(fig)

    def test_plot_beam_variable(
        self, valid_ovro_dataset_with_beam: xr.Dataset
    ) -> None:
        """plot() can plot BEAM variable when present."""
        fig = valid_ovro_dataset_with_beam.radport.plot(var="BEAM")
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_invalid_variable_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found in dataset"):
            valid_ovro_dataset.radport.plot(var="BEAM")

    def test_plot_invalid_variable_lists_available(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Error message lists available variables."""
        with pytest.raises(ValueError) as exc_info:
            valid_ovro_dataset.radport.plot(var="NONEXISTENT")

        error_msg = str(exc_info.value)
        assert "SKY" in error_msg

    def test_plot_title_contains_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Plot title contains time, frequency, and polarization info."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            # Should contain variable name
            assert "SKY" in title
            # Should contain frequency in MHz
            assert "MHz" in title
            # Should contain polarization
            assert "pol=" in title
        finally:
            plt.close(fig)

    def test_plot_axis_labels(self, valid_ovro_dataset: xr.Dataset) -> None:
        """Plot has proper axis labels for l and m coordinates."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            ax = fig.axes[0]
            assert "l" in ax.get_xlabel().lower()
            assert "m" in ax.get_ylabel().lower()
        finally:
            plt.close(fig)


class TestRadportSelectionHelpers:
    """Tests for selection helper methods."""

    def test_nearest_freq_idx_exact_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx returns correct index for exact frequency match."""
        # Dataset has frequencies [46e6, 50e6, 54e6] Hz
        idx = valid_ovro_dataset.radport.nearest_freq_idx(50.0)  # 50 MHz
        assert idx == 1

    def test_nearest_freq_idx_nearest_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx returns nearest index for non-exact frequency."""
        # 49 MHz is closer to 50 MHz (index 1) than 46 MHz (index 0)
        idx = valid_ovro_dataset.radport.nearest_freq_idx(49.0)
        assert idx == 1

    def test_nearest_freq_idx_lower_bound(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx handles frequencies below range."""
        idx = valid_ovro_dataset.radport.nearest_freq_idx(10.0)  # Below 46 MHz
        assert idx == 0  # Should return first index

    def test_nearest_freq_idx_upper_bound(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx handles frequencies above range."""
        idx = valid_ovro_dataset.radport.nearest_freq_idx(100.0)  # Above 54 MHz
        assert idx == 2  # Should return last index

    def test_nearest_time_idx_exact_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_time_idx returns correct index for exact MJD match."""
        # Dataset has times [60000.0, 60000.1] MJD
        idx = valid_ovro_dataset.radport.nearest_time_idx(60000.0)
        assert idx == 0

    def test_nearest_time_idx_nearest_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_time_idx returns nearest index for non-exact MJD."""
        # 60000.08 is closer to 60000.1 (index 1) than 60000.0 (index 0)
        idx = valid_ovro_dataset.radport.nearest_time_idx(60000.08)
        assert idx == 1

    def test_nearest_lm_idx_center(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_lm_idx returns center indices for (0, 0)."""
        # Dataset has l and m from -1 to 1 with 50 points
        l_idx, m_idx = valid_ovro_dataset.radport.nearest_lm_idx(0.0, 0.0)
        # Center should be around index 24 or 25 for 50 points
        assert 23 <= l_idx <= 26
        assert 23 <= m_idx <= 26

    def test_nearest_lm_idx_corner(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_lm_idx returns corner indices for extreme values."""
        l_idx, m_idx = valid_ovro_dataset.radport.nearest_lm_idx(-1.0, 1.0)
        assert l_idx == 0  # -1 is at index 0
        assert m_idx == 49  # 1 is at index 49

    def test_nearest_lm_idx_returns_tuple(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_lm_idx returns a tuple of two integers."""
        result = valid_ovro_dataset.radport.nearest_lm_idx(0.5, -0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)


class TestRadportPlotFrequencySelection:
    """Tests for plot() frequency selection by MHz."""

    def test_plot_freq_mhz_parameter(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_freq_mhz_overrides_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """freq_mhz takes precedence over freq_idx."""
        # freq_idx=0 would select 46 MHz, but freq_mhz=54 should select index 2
        fig = valid_ovro_dataset.radport.plot(freq_idx=0, freq_mhz=54.0)
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            # Title should show 54.00 MHz, not 46.00 MHz
            assert "54.00 MHz" in title
        finally:
            plt.close(fig)

    def test_plot_time_mjd_parameter(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts time_mjd parameter."""
        fig = valid_ovro_dataset.radport.plot(time_mjd=60000.1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_mjd_overrides_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_mjd takes precedence over time_idx."""
        # time_idx=0 would select 60000.0, but time_mjd=60000.1 should select index 1
        fig = valid_ovro_dataset.radport.plot(time_idx=0, time_mjd=60000.1)
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            # Title should show 60000.1 MJD, not 60000.0 MJD
            assert "60000.1" in title
        finally:
            plt.close(fig)


class TestRadportPlotMasking:
    """Tests for plot() circular masking functionality."""

    def test_plot_mask_radius_parameter(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_mask_radius_creates_masked_values(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """mask_radius creates masked/NaN values outside the specified radius."""
        # Get the plotted data by accessing the image
        fig = valid_ovro_dataset.radport.plot(mask_radius=10)
        try:
            ax = fig.axes[0]
            im = ax.images[0]
            data = im.get_array()
            # With mask_radius=10, corner pixels should be masked or NaN
            # matplotlib may return a masked array
            if hasattr(data, "mask"):
                # Check that some values are masked
                assert np.any(data.mask)
            else:
                # Check for NaN values
                assert np.any(np.isnan(data))
        finally:
            plt.close(fig)

    def test_plot_mask_radius_preserves_center(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """mask_radius preserves data within the specified radius."""
        fig = valid_ovro_dataset.radport.plot(mask_radius=25)
        try:
            ax = fig.axes[0]
            im = ax.images[0]
            data = im.get_array()
            # Center pixels should not be NaN
            center = data.shape[0] // 2
            assert not np.isnan(data[center, center])
        finally:
            plt.close(fig)


class TestRadportPlotWithNaN:
    """Tests for plot() method with datasets containing NaN values."""

    @pytest.fixture
    def dataset_with_nan(self) -> xr.Dataset:
        """Create a dataset with some NaN values."""
        np.random.seed(42)
        data = np.random.rand(2, 3, 2, 50, 50) * 10
        # Add some NaN values
        data[0, 0, 0, :10, :10] = np.nan
        return xr.Dataset(
            data_vars={
                "SKY": (
                    ["time", "frequency", "polarization", "l", "m"],
                    data,
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

    def test_plot_handles_nan_values(self, dataset_with_nan: xr.Dataset) -> None:
        """plot() handles datasets with NaN values without error."""
        fig = dataset_with_nan.radport.plot()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_robust_with_nan(self, dataset_with_nan: xr.Dataset) -> None:
        """plot() with robust=True handles NaN values correctly."""
        fig = dataset_with_nan.radport.plot(robust=True)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)
