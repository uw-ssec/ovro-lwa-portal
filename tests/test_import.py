"""Test basic imports for the OVRO-LWA Portal package."""

from __future__ import annotations


def test_import_package():
    """Test that the package can be imported."""
    import ovro_lwa_portal

    assert ovro_lwa_portal is not None


def test_version():
    """Test that the version is defined."""
    import ovro_lwa_portal

    assert hasattr(ovro_lwa_portal, "__version__")
    assert isinstance(ovro_lwa_portal.__version__, str)
    assert len(ovro_lwa_portal.__version__) > 0
