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


# =============================================================================
# Phase B Tests: Cutout, Dynamic Spectrum, Difference Maps
# =============================================================================


class TestRadportCutout:
    """Tests for cutout() method."""

    def test_cutout_returns_dataarray(self, valid_ovro_dataset: xr.Dataset) -> None:
        """cutout() returns an xarray DataArray."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3
        )
        assert isinstance(cutout, xr.DataArray)

    def test_cutout_has_correct_dimensions(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() returns 2D DataArray with l and m dimensions."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3
        )
        assert set(cutout.dims) == {"l", "m"}

    def test_cutout_smaller_than_full_image(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() returns smaller region than full image."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.2, dm=0.2
        )
        full_size = valid_ovro_dataset.sizes["l"] * valid_ovro_dataset.sizes["m"]
        assert cutout.size < full_size

    def test_cutout_with_freq_mhz(self, valid_ovro_dataset: xr.Dataset) -> None:
        """cutout() accepts freq_mhz parameter."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3, freq_mhz=50.0
        )
        assert cutout.attrs["freq_idx"] == 1  # 50 MHz is index 1

    def test_cutout_metadata_attrs(self, valid_ovro_dataset: xr.Dataset) -> None:
        """cutout() adds metadata attributes."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.1, m_center=-0.1, dl=0.2, dm=0.3
        )
        assert cutout.attrs["cutout_l_center"] == 0.1
        assert cutout.attrs["cutout_m_center"] == -0.1
        assert cutout.attrs["cutout_dl"] == 0.2
        assert cutout.attrs["cutout_dm"] == 0.3

    def test_cutout_invalid_variable_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.cutout(
                l_center=0.0, m_center=0.0, dl=0.1, dm=0.1, var="BEAM"
            )

    def test_cutout_out_of_bounds_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() raises ValueError when region is outside data bounds."""
        with pytest.raises(ValueError, match="empty"):
            valid_ovro_dataset.radport.cutout(
                l_center=5.0, m_center=5.0, dl=0.1, dm=0.1  # Outside [-1, 1] range
            )


class TestRadportPlotCutout:
    """Tests for plot_cutout() method."""

    def test_plot_cutout_returns_figure(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_cutout() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_cutout_with_options(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_cutout() accepts customization options."""
        fig = valid_ovro_dataset.radport.plot_cutout(
            l_center=0.0,
            m_center=0.0,
            dl=0.3,
            dm=0.3,
            freq_mhz=50.0,
            cmap="viridis",
            figsize=(8, 8),
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_cutout_title_contains_bounds(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_cutout() title includes cutout bounds."""
        fig = valid_ovro_dataset.radport.plot_cutout(
            l_center=0.0, m_center=0.0, dl=0.1, dm=0.1
        )
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            assert "l=" in title
            assert "m=" in title
        finally:
            plt.close(fig)


class TestRadportDynamicSpectrum:
    """Tests for dynamic_spectrum() method."""

    def test_dynamic_spectrum_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum() returns xarray DataArray."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert isinstance(dynspec, xr.DataArray)

    def test_dynamic_spectrum_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum() returns 2D array with time and frequency."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert set(dynspec.dims) == {"time", "frequency"}

    def test_dynamic_spectrum_shape(self, valid_ovro_dataset: xr.Dataset) -> None:
        """dynamic_spectrum() has expected shape."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert dynspec.sizes["time"] == 2
        assert dynspec.sizes["frequency"] == 3

    def test_dynamic_spectrum_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """dynamic_spectrum() adds pixel metadata attributes."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert "pixel_l" in dynspec.attrs
        assert "pixel_m" in dynspec.attrs
        assert "l_idx" in dynspec.attrs
        assert "m_idx" in dynspec.attrs

    def test_dynamic_spectrum_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0, var="BEAM")


class TestRadportPlotDynamicSpectrum:
    """Tests for plot_dynamic_spectrum() method."""

    def test_plot_dynamic_spectrum_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_axis_labels(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum() has correct axis labels."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
        try:
            ax = fig.axes[0]
            assert "Time" in ax.get_xlabel()
            assert "Frequency" in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_with_options(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum() accepts customization options."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum(
            l=0.0, m=0.0, cmap="viridis", robust=False, vmin=0.0, vmax=10.0
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


class TestRadportDiff:
    """Tests for diff() method."""

    def test_diff_time_returns_dataarray(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='time' returns xarray DataArray."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1)
        assert isinstance(diff, xr.DataArray)

    def test_diff_frequency_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """diff() with mode='frequency' returns xarray DataArray."""
        diff = valid_ovro_dataset.radport.diff(mode="frequency", freq_idx=1)
        assert isinstance(diff, xr.DataArray)

    def test_diff_has_lm_dims(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() returns 2D array with l and m dimensions."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1)
        assert set(diff.dims) == {"l", "m"}

    def test_diff_time_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='time' adds correct metadata."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1)
        assert diff.attrs["diff_mode"] == "time"
        assert diff.attrs["time_idx_current"] == 1
        assert diff.attrs["time_idx_prev"] == 0

    def test_diff_frequency_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='frequency' adds correct metadata."""
        diff = valid_ovro_dataset.radport.diff(mode="frequency", freq_idx=2)
        assert diff.attrs["diff_mode"] == "frequency"
        assert diff.attrs["freq_idx_current"] == 2
        assert diff.attrs["freq_idx_prev"] == 1

    def test_diff_time_idx_zero_raises(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='time' and time_idx=0 raises ValueError."""
        with pytest.raises(ValueError, match="time_idx must be >= 1"):
            valid_ovro_dataset.radport.diff(mode="time", time_idx=0)

    def test_diff_freq_idx_zero_raises(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='frequency' and freq_idx=0 raises ValueError."""
        with pytest.raises(ValueError, match="freq_idx must be >= 1"):
            valid_ovro_dataset.radport.diff(mode="frequency", freq_idx=0)

    def test_diff_with_freq_mhz(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() accepts freq_mhz parameter."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1, freq_mhz=50.0)
        assert diff.attrs["freq_idx"] == 1


class TestRadportPlotDiff:
    """Tests for plot_diff() method."""

    def test_plot_diff_time_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() with mode='time' returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_diff_frequency_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() with mode='frequency' returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="frequency", freq_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_diff_uses_diverging_cmap(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() uses diverging colormap by default."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            # Default cmap is RdBu_r (diverging)
            ax = fig.axes[0]
            im = ax.images[0]
            assert im.cmap.name == "RdBu_r"
        finally:
            plt.close(fig)

    def test_plot_diff_symmetric_scale(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_diff() uses symmetric color scale by default."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            ax = fig.axes[0]
            im = ax.images[0]
            vmin, vmax = im.get_clim()
            # Symmetric means |vmin| == |vmax|
            assert abs(abs(vmin) - abs(vmax)) < 0.01
        finally:
            plt.close(fig)

    def test_plot_diff_title_contains_info(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() title contains relevant information."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            assert "Diff" in title
        finally:
            plt.close(fig)


# =============================================================================
# Phase C Tests: Data Quality and Grid Plots
# =============================================================================


class TestRadportFindValidFrame:
    """Tests for find_valid_frame() method."""

    def test_find_valid_frame_returns_tuple(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() returns a tuple of two integers."""
        result = valid_ovro_dataset.radport.find_valid_frame()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_find_valid_frame_returns_first_valid(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() returns first (0, 0) for all-valid dataset."""
        ti, fi = valid_ovro_dataset.radport.find_valid_frame()
        assert ti == 0
        assert fi == 0

    def test_find_valid_frame_with_threshold(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() respects min_finite_fraction threshold."""
        # With 100% threshold, should still find frame (all data is finite)
        ti, fi = valid_ovro_dataset.radport.find_valid_frame(min_finite_fraction=1.0)
        assert ti >= 0
        assert fi >= 0

    def test_find_valid_frame_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.find_valid_frame(var="BEAM")


class TestRadportFiniteFraction:
    """Tests for finite_fraction() method."""

    def test_finite_fraction_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() returns xarray DataArray."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert isinstance(frac, xr.DataArray)

    def test_finite_fraction_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() returns 2D array with time and frequency dims."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert set(frac.dims) == {"time", "frequency"}

    def test_finite_fraction_values_in_range(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() values are between 0 and 1."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert float(frac.min()) >= 0.0
        assert float(frac.max()) <= 1.0

    def test_finite_fraction_all_valid_data(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() returns 1.0 for all-valid dataset."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert float(frac.min()) == 1.0

    def test_finite_fraction_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """finite_fraction() adds metadata attributes."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert frac.attrs["variable"] == "SKY"
        assert frac.attrs["pol"] == 0


class TestRadportPlotGrid:
    """Tests for plot_grid() method."""

    def test_plot_grid_returns_figure(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_grid() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_grid()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts time_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(time_indices=[0, 1])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_freq_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts freq_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(freq_indices=[0, 1, 2])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_freq_mhz_list(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts freq_mhz_list parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(freq_mhz_list=[46.0, 50.0])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_custom_ncols(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_grid() accepts ncols parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(ncols=2)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.plot_grid(var="BEAM")

    def test_plot_grid_creates_multiple_axes(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() creates correct number of axes."""
        # 2 times x 3 frequencies = 6 panels
        fig = valid_ovro_dataset.radport.plot_grid()
        try:
            # At least 6 axes (may have colorbar axis)
            assert len(fig.axes) >= 6
        finally:
            plt.close(fig)


class TestRadportPlotFrequencyGrid:
    """Tests for plot_frequency_grid() method."""

    def test_plot_frequency_grid_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_grid() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_frequency_grid()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_grid_single_time(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_grid() plots single time across frequencies."""
        fig = valid_ovro_dataset.radport.plot_frequency_grid(time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_grid_with_freq_list(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_grid() accepts freq_mhz_list parameter."""
        fig = valid_ovro_dataset.radport.plot_frequency_grid(
            freq_mhz_list=[46.0, 54.0]
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


class TestRadportPlotTimeGrid:
    """Tests for plot_time_grid() method."""

    def test_plot_time_grid_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_time_grid()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_grid_with_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() accepts freq_idx parameter."""
        fig = valid_ovro_dataset.radport.plot_time_grid(freq_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_grid_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot_time_grid(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_grid_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() accepts time_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_time_grid(
            freq_mhz=50.0, time_indices=[0, 1]
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =============================================================================
# Phase D: 1D Analysis Methods Tests
# =============================================================================


class TestRadportLightCurve:
    """Tests for light_curve() method."""

    def test_light_curve_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() returns xr.DataArray."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert isinstance(lc, xr.DataArray)

    def test_light_curve_has_time_dimension(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() result has 'time' as only dimension."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert lc.dims == ("time",)

    def test_light_curve_correct_length(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() has correct number of time points."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert len(lc) == valid_ovro_dataset.sizes["time"]

    def test_light_curve_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() accepts freq_mhz parameter."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        assert lc.attrs["freq_mhz"] == 50.0

    def test_light_curve_with_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() accepts freq_idx parameter."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0, freq_idx=1)
        assert lc.attrs["freq_idx"] == 1

    def test_light_curve_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() includes metadata attributes."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert "variable" in lc.attrs
        assert "l" in lc.attrs
        assert "m" in lc.attrs
        assert "freq_mhz" in lc.attrs

    def test_light_curve_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0, var="INVALID")


class TestRadportPlotLightCurve:
    """Tests for plot_light_curve() method."""

    def test_plot_light_curve_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_light_curve() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_light_curve(l=0.0, m=0.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_light_curve_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_light_curve() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot_light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_light_curve_axis_labels(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_light_curve() has correct axis labels."""
        fig = valid_ovro_dataset.radport.plot_light_curve(l=0.0, m=0.0)
        try:
            ax = fig.axes[0]
            assert "Time" in ax.get_xlabel()
            assert "Intensity" in ax.get_ylabel()
        finally:
            plt.close(fig)


class TestRadportSpectrum:
    """Tests for spectrum() method."""

    def test_spectrum_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() returns xr.DataArray."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert isinstance(spec, xr.DataArray)

    def test_spectrum_has_frequency_dimension(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() result has 'frequency' as only dimension."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert spec.dims == ("frequency",)

    def test_spectrum_correct_length(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() has correct number of frequency points."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert len(spec) == valid_ovro_dataset.sizes["frequency"]

    def test_spectrum_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() accepts time_idx parameter."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0, time_idx=1)
        assert spec.attrs["time_idx"] == 1

    def test_spectrum_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() includes metadata attributes."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert "variable" in spec.attrs
        assert "l" in spec.attrs
        assert "m" in spec.attrs
        assert "time_mjd" in spec.attrs

    def test_spectrum_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0, var="INVALID")


class TestRadportPlotSpectrum:
    """Tests for plot_spectrum() method."""

    def test_plot_spectrum_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_spectrum_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() accepts time_idx parameter."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0, time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_spectrum_axis_labels(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() has correct axis labels."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0)
        try:
            ax = fig.axes[0]
            assert "Frequency" in ax.get_xlabel()
            assert "Intensity" in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_plot_spectrum_freq_unit_hz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() accepts freq_unit='Hz'."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0, freq_unit="Hz")
        try:
            ax = fig.axes[0]
            assert "Hz" in ax.get_xlabel()
        finally:
            plt.close(fig)


class TestRadportTimeAverage:
    """Tests for time_average() method."""

    def test_time_average_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() returns xr.DataArray."""
        avg = valid_ovro_dataset.radport.time_average()
        assert isinstance(avg, xr.DataArray)

    def test_time_average_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() result has (frequency, l, m) dimensions."""
        avg = valid_ovro_dataset.radport.time_average()
        assert set(avg.dims) == {"frequency", "l", "m"}

    def test_time_average_removes_time_dim(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() removes time dimension."""
        avg = valid_ovro_dataset.radport.time_average()
        assert "time" not in avg.dims

    def test_time_average_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() accepts time_indices parameter."""
        avg = valid_ovro_dataset.radport.time_average(time_indices=[0, 1])
        assert "time_indices" in avg.attrs

    def test_time_average_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() includes metadata attributes."""
        avg = valid_ovro_dataset.radport.time_average()
        assert avg.attrs["operation"] == "time_average"
        assert "variable" in avg.attrs

    def test_time_average_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.time_average(var="INVALID")


class TestRadportFrequencyAverage:
    """Tests for frequency_average() method."""

    def test_frequency_average_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() returns xr.DataArray."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert isinstance(avg, xr.DataArray)

    def test_frequency_average_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() result has (time, l, m) dimensions."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert set(avg.dims) == {"time", "l", "m"}

    def test_frequency_average_removes_freq_dim(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() removes frequency dimension."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert "frequency" not in avg.dims

    def test_frequency_average_with_freq_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() accepts freq_indices parameter."""
        avg = valid_ovro_dataset.radport.frequency_average(freq_indices=[0, 1])
        assert "freq_indices" in avg.attrs

    def test_frequency_average_with_freq_range(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() accepts freq_min/max_mhz parameters."""
        avg = valid_ovro_dataset.radport.frequency_average(
            freq_min_mhz=46.0, freq_max_mhz=54.0
        )
        assert avg.attrs.get("freq_min_mhz") == 46.0
        assert avg.attrs.get("freq_max_mhz") == 54.0

    def test_frequency_average_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() includes metadata attributes."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert avg.attrs["operation"] == "frequency_average"
        assert "variable" in avg.attrs

    def test_frequency_average_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.frequency_average(var="INVALID")

    def test_frequency_average_invalid_range_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() raises ValueError for invalid frequency range."""
        with pytest.raises(ValueError, match="No frequencies in range"):
            valid_ovro_dataset.radport.frequency_average(
                freq_min_mhz=1000.0, freq_max_mhz=2000.0
            )


class TestRadportPlotTimeAverage:
    """Tests for plot_time_average() method."""

    def test_plot_time_average_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_time_average()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_average_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot_time_average(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_average_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() accepts time_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_time_average(time_indices=[0, 1])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_average_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_time_average(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


class TestRadportPlotFrequencyAverage:
    """Tests for plot_frequency_average() method."""

    def test_plot_frequency_average_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_frequency_average()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_average_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() accepts time_idx parameter."""
        fig = valid_ovro_dataset.radport.plot_frequency_average(time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_average_with_freq_range(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() accepts freq_min/max_mhz parameters."""
        fig = valid_ovro_dataset.radport.plot_frequency_average(
            freq_min_mhz=46.0, freq_max_mhz=54.0
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_average_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_frequency_average(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =============================================================================
# Phase E: WCS & Coordinate Methods Tests
# =============================================================================


class TestRadportHasWcs:
    """Tests for has_wcs property."""

    def test_has_wcs_false_without_wcs(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """has_wcs returns False when no WCS header is present."""
        assert valid_ovro_dataset.radport.has_wcs is False

    def test_has_wcs_true_with_wcs(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """has_wcs returns True when WCS header is present."""
        assert valid_ovro_dataset_with_wcs.radport.has_wcs is True


class TestRadportPixelToCoords:
    """Tests for pixel_to_coords() method."""

    def test_pixel_to_coords_returns_tuple(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords() returns tuple of (ra, dec)."""
        result = valid_ovro_dataset_with_wcs.radport.pixel_to_coords(25, 25)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pixel_to_coords_center_pixel(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords() returns expected coords for center pixel."""
        ra, dec = valid_ovro_dataset_with_wcs.radport.pixel_to_coords(25, 25)
        # WCS is centered at RA=180, Dec=45 with CRPIX at (25, 25)
        # Allow 2 degree tolerance due to SIN projection effects
        assert abs(ra - 180.0) < 2.0
        assert abs(dec - 45.0) < 2.0

    def test_pixel_to_coords_ra_range(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords() returns RA in [0, 360) range."""
        ra, dec = valid_ovro_dataset_with_wcs.radport.pixel_to_coords(0, 0)
        assert 0 <= ra < 360

    def test_pixel_to_coords_out_of_bounds_l_raises(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords() raises for l_idx out of bounds."""
        with pytest.raises(ValueError, match="l_idx=100 out of bounds"):
            valid_ovro_dataset_with_wcs.radport.pixel_to_coords(100, 25)

    def test_pixel_to_coords_out_of_bounds_m_raises(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords() raises for m_idx out of bounds."""
        with pytest.raises(ValueError, match="m_idx=100 out of bounds"):
            valid_ovro_dataset_with_wcs.radport.pixel_to_coords(25, 100)

    def test_pixel_to_coords_no_wcs_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """pixel_to_coords() raises when no WCS is available."""
        with pytest.raises(ValueError, match="No WCS header found"):
            valid_ovro_dataset.radport.pixel_to_coords(25, 25)


class TestRadportCoordsToPixel:
    """Tests for coords_to_pixel() method."""

    def test_coords_to_pixel_returns_tuple(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """coords_to_pixel() returns tuple of (l_idx, m_idx)."""
        result = valid_ovro_dataset_with_wcs.radport.coords_to_pixel(180.0, 45.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_coords_to_pixel_returns_integers(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """coords_to_pixel() returns integer indices."""
        l_idx, m_idx = valid_ovro_dataset_with_wcs.radport.coords_to_pixel(180.0, 45.0)
        assert isinstance(l_idx, int)
        assert isinstance(m_idx, int)

    def test_coords_to_pixel_center_coords(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """coords_to_pixel() returns center pixel for center coords."""
        l_idx, m_idx = valid_ovro_dataset_with_wcs.radport.coords_to_pixel(180.0, 45.0)
        # Should be near the reference pixel (25, 25)
        assert abs(l_idx - 25) <= 1
        assert abs(m_idx - 25) <= 1

    def test_coords_to_pixel_roundtrip(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords and coords_to_pixel are approximate inverses."""
        # Start with pixel
        l_orig, m_orig = 30, 30
        ra, dec = valid_ovro_dataset_with_wcs.radport.pixel_to_coords(l_orig, m_orig)
        l_back, m_back = valid_ovro_dataset_with_wcs.radport.coords_to_pixel(ra, dec)
        # Should round-trip approximately
        assert abs(l_back - l_orig) <= 1
        assert abs(m_back - m_orig) <= 1

    def test_coords_to_pixel_no_wcs_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """coords_to_pixel() raises when no WCS is available."""
        with pytest.raises(ValueError, match="No WCS header found"):
            valid_ovro_dataset.radport.coords_to_pixel(180.0, 45.0)


class TestRadportPlotWcs:
    """Tests for plot_wcs() method."""

    def test_plot_wcs_returns_figure(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() returns matplotlib Figure."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_with_freq_mhz(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_with_mask_radius(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts mask_radius parameter."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_custom_colors(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts color customization parameters."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(
            grid_color="yellow",
            label_color="cyan",
            facecolor="navy",
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_no_colorbar(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts add_colorbar=False."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(add_colorbar=False)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_no_wcs_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_wcs() raises when no WCS is available."""
        with pytest.raises(ValueError, match="No WCS header found"):
            valid_ovro_dataset.radport.plot_wcs()

    def test_plot_wcs_invalid_var_raises(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset_with_wcs.radport.plot_wcs(var="INVALID")


# =============================================================================
# Phase F: Animation & Export Tests
# =============================================================================


class TestRadportAnimateTime:
    """Tests for RadportAccessor.animate_time() method."""

    def test_animate_time_returns_animation(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() returns a FuncAnimation object."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time()
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts freq_mhz parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(freq_mhz=50.0)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_with_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts freq_idx parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(freq_idx=1)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts mask_radius parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(mask_radius=20)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_custom_cmap(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts custom colormap."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(cmap="viridis")
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.animate_time(var="INVALID")


class TestRadportAnimateFrequency:
    """Tests for RadportAccessor.animate_frequency() method."""

    def test_animate_frequency_returns_animation(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() returns a FuncAnimation object."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency()
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() accepts time_idx parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency(time_idx=1)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_with_time_mjd(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() accepts time_mjd parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency(time_mjd=60000.0)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() accepts mask_radius parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency(mask_radius=20)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.animate_frequency(var="INVALID")


class TestRadportExportFrames:
    """Tests for RadportAccessor.export_frames() method."""

    def test_export_frames_returns_list(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() returns a list of file paths."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
        )
        assert isinstance(files, list)
        assert len(files) == 1

    def test_export_frames_creates_files(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() creates actual files on disk."""
        import os

        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
        )
        for f in files:
            assert os.path.exists(f)

    def test_export_frames_all_combinations(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() exports all time/freq combinations when not specified."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(output_dir)
        # Dataset has 2 times x 3 frequencies = 6 frames
        assert len(files) == 6

    def test_export_frames_custom_format(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() accepts custom format parameter."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
            format="jpg",
        )
        assert files[0].endswith(".jpg")

    def test_export_frames_custom_template(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() accepts custom filename template."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
            filename_template="frame_{time_idx}_{freq_idx}.{format}",
        )
        assert "frame_0_0.png" in files[0]

    def test_export_frames_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() accepts mask_radius parameter."""
        import os

        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
            mask_radius=20,
        )
        assert len(files) == 1
        assert os.path.exists(files[0])

    def test_export_frames_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() raises ValueError for invalid variable."""
        output_dir = str(tmp_path / "frames")
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.export_frames(output_dir, var="INVALID")

    def test_export_frames_creates_directory(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() creates output directory if it doesn't exist."""
        import os

        output_dir = str(tmp_path / "new_directory" / "frames")
        assert not os.path.exists(output_dir)
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
        )
        assert os.path.exists(output_dir)
        assert len(files) == 1
