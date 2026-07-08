"""Placeholder test proving pytest and the src layout are wired.

Replaced by real tests from Step 2 onward.
"""

from volrisk import __version__


def test_package_is_importable() -> None:
    assert __version__ == "0.1.0"
