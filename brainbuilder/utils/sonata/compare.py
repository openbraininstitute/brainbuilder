# SPDX-License-Identifier: Apache-2.0
"""Compare two extracted SONATA circuits for equivalence using original IDs."""

from pathlib import Path

import bluepysnap
import h5py

from brainbuilder.utils import load_json


def assert_circuits_equal(path_a, path_b, strict_order=False):
    """Assert two extracted circuits are equal using original IDs.

    Checks that they have the same populations with the same original nodes,
    and the same edges (translated to original IDs).

    Args:
        path_a: Path to first circuit directory.
        path_b: Path to second circuit directory.
        strict_order: If True, require same node and edge ordering.
            If False (default), allow reordering.

    Raises:
        AssertionError: If the circuits differ.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)

    circ_a = bluepysnap.Circuit(path_a / "circuit_config.json")
    circ_b = bluepysnap.Circuit(path_b / "circuit_config.json")
    mapping_a = load_json(path_a / "id_mapping.json")
    mapping_b = load_json(path_b / "id_mapping.json")

    # Same node populations
    assert set(circ_a.nodes.keys()) == set(circ_b.nodes.keys()), (
        f"Node populations differ: {set(circ_a.nodes.keys())} vs {set(circ_b.nodes.keys())}"
    )

    # Same original IDs per population
    for pop_name in circ_a.nodes.keys():
        orig_a = mapping_a[pop_name]["original_id"]
        orig_b = mapping_b[pop_name]["original_id"]
        if strict_order:
            assert orig_a == orig_b, (
                f"Population '{pop_name}' original_ids differ: {orig_a} vs {orig_b}"
            )
        else:
            assert sorted(orig_a) == sorted(orig_b), (
                f"Population '{pop_name}' original_ids differ: {sorted(orig_a)} vs {sorted(orig_b)}"
            )
        assert mapping_a[pop_name]["original_name"] == mapping_b[pop_name]["original_name"], (
            f"Population '{pop_name}' original_name differs"
        )

    # Same edge populations
    assert set(circ_a.edges.keys()) == set(circ_b.edges.keys()), (
        f"Edge populations differ: {set(circ_a.edges.keys())} vs {set(circ_b.edges.keys())}"
    )

    # Same edges (translated to original IDs)
    for edge_name in circ_a.edges.keys():
        edge_a = circ_a.edges[edge_name]
        edge_b = circ_b.edges[edge_name]
        src_pop = edge_a.source.name
        tgt_pop = edge_a.target.name

        # Build new_id -> original_id lookup for each circuit
        orig_src_a = dict(zip(mapping_a[src_pop]["new_id"], mapping_a[src_pop]["original_id"]))
        orig_tgt_a = dict(zip(mapping_a[tgt_pop]["new_id"], mapping_a[tgt_pop]["original_id"]))
        orig_src_b = dict(zip(mapping_b[src_pop]["new_id"], mapping_b[src_pop]["original_id"]))
        orig_tgt_b = dict(zip(mapping_b[tgt_pop]["new_id"], mapping_b[tgt_pop]["original_id"]))

        with h5py.File(edge_a.h5_filepath, "r") as h5:
            sgids_a = h5[f"edges/{edge_name}/source_node_id"][:]
            tgids_a = h5[f"edges/{edge_name}/target_node_id"][:]
        with h5py.File(edge_b.h5_filepath, "r") as h5:
            sgids_b = h5[f"edges/{edge_name}/source_node_id"][:]
            tgids_b = h5[f"edges/{edge_name}/target_node_id"][:]

        edges_a = [(orig_src_a[int(s)], orig_tgt_a[int(t)]) for s, t in zip(sgids_a, tgids_a)]
        edges_b = [(orig_src_b[int(s)], orig_tgt_b[int(t)]) for s, t in zip(sgids_b, tgids_b)]

        if strict_order:
            assert edges_a == edges_b, (
                f"Edge population '{edge_name}' differs:\n  {edges_a}\n  vs\n  {edges_b}"
            )
        else:
            assert sorted(edges_a) == sorted(edges_b), (
                f"Edge population '{edge_name}' differs:\n  "
                f"{sorted(edges_a)}\n  vs\n  {sorted(edges_b)}"
            )
