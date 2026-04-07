"""Tests for visualization dependency import guards."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import ovro_lwa_portal.viz._imports as _imports_mod


@pytest.fixture(autouse=True)
def _reset_viz_deps_state():
    """Reset the cached dependency check state before and after each test."""
    _imports_mod._VIZ_DEPS_CHECKED = None
    yield
    _imports_mod._VIZ_DEPS_CHECKED = None


def test_check_viz_deps_passes_when_installed() -> None:
    """check_viz_deps succeeds when all deps are available."""
    _imports_mod.check_viz_deps()  # Should not raise


def test_check_viz_deps_raises_when_missing() -> None:
    """check_viz_deps raises ImportError with install instructions."""
    with patch("builtins.__import__", side_effect=ImportError("no panel")):
        with pytest.raises(ImportError, match="visualization"):
            _imports_mod.check_viz_deps()


def test_check_viz_deps_caches_result() -> None:
    """check_viz_deps caches the result after first successful check."""
    _imports_mod.check_viz_deps()
    assert _imports_mod._VIZ_DEPS_CHECKED is True

    # Second call should be a no-op (uses cache)
    _imports_mod.check_viz_deps()
    assert _imports_mod._VIZ_DEPS_CHECKED is True
