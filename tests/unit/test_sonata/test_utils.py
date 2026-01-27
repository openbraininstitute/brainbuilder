
from brainbuilder.utils import utils

import pytest

import numpy as np

# Mock h5py.Group
class MockGroup(dict):
    def __getitem__(self, key):
        val = super().__getitem__(key)
        if isinstance(val, dict):
            return MockGroup(val)
        return val

@pytest.fixture
def node_group():
    return MockGroup({
        "@library": {
            "prop1": np.array([10, 20, 30, 40, 50])
        }
    })

def test_integer_indices(node_group):
    data = np.array([0, 2, 1, 2])
    expected = np.array([10, 30, 20, 30])
    result = utils.get_property(node_group, data, "prop1")
    np.testing.assert_array_equal(result, expected)

def test_non_integer_data(node_group):
    data = np.array([5.5, 1.1])
    result = utils.get_property(node_group, data, "prop1")
    np.testing.assert_array_equal(result, data)

def test_missing_library():
    node_group = MockGroup({})
    data = np.array([0, 1])
    result = utils.get_property(node_group, data, "prop1")
    np.testing.assert_array_equal(result, data)

def test_empty_data(node_group):
    data = np.array([])
    result = utils.get_property(node_group, data, "prop1")
    np.testing.assert_array_equal(result, data)

def test_repeated_indices(node_group):
    data = np.array([2, 2, 0, 4, 2])
    expected = np.array([30, 30, 10, 50, 30])
    result = utils.get_property(node_group, data, "prop1")
    np.testing.assert_array_equal(result, expected)

def test__gather_layout_from_networks():
    res = utils.gather_layout_from_networks({"nodes": [], "edges": []})
    assert res == ({}, {})

    nodes, edges = utils.gather_layout_from_networks(
        {
            "nodes": [
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"a": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"b": {"type": "biophysical"}, "c": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"A": {"type": "biophysical"}},
                },
            ],
            "edges": [
                {
                    "edges_file": "a/b/a.h5",
                    "populations": {"a_a": {"type": "biophysical"}},
                },
                {
                    "edges_file": "a/b/bc.h5",
                    "populations": {"b_c": {"type": "biophysical"}, "c_b": {"type": "biophysical"}},
                },
                {
                    "edges_file": "a/a/bc.h5",
                    "populations": {"a_c": {"type": "biophysical"}, "a_b": {"type": "biophysical"}},
                },
                {
                    "edges_file": "a/b/a.h5",
                    "populations": {"A_a": {"type": "biophysical"}},
                },
            ],
        }
    )
    assert nodes == {
        "A": "A/a.h5",
        "a": "a/a.h5",
        "b": "b/bc.h5",
        "c": "b/bc.h5",
    }
    assert edges == {
        "A_a": "A_a/a.h5",
        "a_a": "a_a/a.h5",
        "a_b": "a/bc.h5",
        "a_c": "a/bc.h5",
        "b_c": "b/bc.h5",
        "c_b": "b/bc.h5",
    }

    nodes, edges = utils.gather_layout_from_networks(
        {
            "nodes": [
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"b": {"type": "biophysical"}, "c": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"B": {"type": "biophysical"}, "C": {"type": "biophysical"}},
                },
            ],
            "edges": [],
        }
    )
    assert nodes == {"B": "b/bc.h5", "C": "b/bc.h5", "b": "b/bc.h5", "c": "b/bc.h5"}

    nodes, edges = utils.gather_layout_from_networks(
        {
            "nodes": [
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"a": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"b": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/bc.h5",
                    "populations": {"c": {"type": "biophysical"}},
                },
                {
                    "nodes_file": "a/b/a.h5",
                    "populations": {"A": {"type": "biophysical"}},
                },
            ],
            "edges": [],
        }
    )
    assert nodes == {
        "A": "A/a.h5",
        "a": "a/a.h5",
        "b": "b/bc.h5",
        "c": "c/bc.h5",
    }