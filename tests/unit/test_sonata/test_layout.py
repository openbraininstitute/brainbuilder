

from brainbuilder.utils.sonata import layout

def test__gather_layout_from_networks():
    res = layout._gather_layout_from_networks({"nodes": [], "edges": []})
    assert res == ({}, {})

    nodes, edges = layout._gather_layout_from_networks(
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

    nodes, edges = layout._gather_layout_from_networks(
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

    nodes, edges = layout._gather_layout_from_networks(
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
