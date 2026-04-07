"""Tests for visualization dependency import guards."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_check_viz_deps_passes_when_installed() -> None:
    """check_viz_deps succeeds when all deps are available."""
    from ovro_lwa_portal.viz._imports import check_viz_deps

    # Reset cached state for fresh check
    import ovro_lwa_portal.viz._imports as mod

    mod._VIZ_DEPS_CHECKED = None
    check_viz_deps()  # Should not raise


def test_check_viz_deps_raises_when_missing() -> None:
    """check_viz_deps raises ImportError with install instructions."""
    import ovro_lwa_portal.viz._imports as mod

    mod._VIZ_DEPS_CHECKED = None

    with patch("builtins.__import__", side_effect=ImportError("no panel")):
        with pytest.raises(ImportError, match="visualization"):
            mod.check_viz_deps()

    # Reset for other tests
    mod._VIZ_DEPS_CHECKED = None


def test_check_viz_deps_caches_result() -> None:
    """check_viz_deps caches the result after first successful check."""
    import ovro_lwa_portal.viz._imports as mod

    mod._VIZ_DEPS_CHECKED = None
    mod.check_viz_deps()
    assert mod._VIZ_DEPS_CHECKED is True

    # Second call should be a no-op (uses cache)
    mod.check_viz_deps()
    assert mod._VIZ_DEPS_CHECKED is True

    # Reset
    mod._VIZ_DEPS_CHECKED = None
