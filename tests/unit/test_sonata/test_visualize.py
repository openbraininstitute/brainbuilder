# SPDX-License-Identifier: Apache-2.0
"""Tests for brainbuilder.utils.sonata.visualize."""

from brainbuilder.utils.sonata.visualize import _color_for_type


def test_color_for_type_stability():
    """Verify that _color_for_type returns the same color for the same type across calls."""
    expected = {
        "external": "#e2d5b6",
        "virtual": "#dbc4cf",
        "biophysical": "#d3e1ea",
        "local": "#b9ddeb",
    }
    for type_label, color in expected.items():
        assert _color_for_type(type_label) == color


def test_color_for_type_distinct():
    """Different types should produce different colors."""
    types = ["external", "virtual", "biophysical", "local"]
    colors = [_color_for_type(t) for t in types]
    assert len(set(colors)) == len(types)
