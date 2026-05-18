# SPDX-License-Identifier: Apache-2.0
"""Compare two extracted SONATA circuits for equivalence using original IDs."""

from pathlib import Path

import bluepysnap
import numpy as np
from bluepysnap.sonata_constants import Edge

from brainbuilder.utils import load_json


def _build_original_id_lut(mapping, pop_name):
    """Build a numpy lookup table: lut[new_id] = original_id.

    Since new_ids are contiguous 0-based integers, a plain numpy array
    gives O(1) vectorized remapping via fancy indexing.
    """
    new_ids = np.asarray(mapping[pop_name]["new_id"], dtype=np.int64)
    orig_ids = np.asarray(mapping[pop_name]["original_id"], dtype=np.int64)
    lut = np.empty(new_ids.max() + 1, dtype=np.int64)
    lut[new_ids] = orig_ids
    return lut


def _edges_to_sorted_pairs(sgids, tgids):
    """Return a lexicographically sorted (N, 2) array of edge pairs."""
    pairs = np.column_stack([sgids, tgids])
    # lexsort sorts by last key first, so (tgids, sgids) gives sort by sgids then tgids
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def assert_circuits_equal(path_a, path_b, strict_node_order=False, strict_edge_order=False):
    """Assert two extracted circuits are equal using original IDs.

    Checks that they have the same populations with the same original nodes,
    and the same edges (translated to original IDs).

    Args:
        path_a: Path to first circuit directory.
        path_b: Path to second circuit directory.
        strict_node_order: If True, require same node ordering per population.
        strict_edge_order: If True, require same edge ordering per edge population.

    Raises:
        AssertionError: If the circuits differ.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)

    circ_a = bluepysnap.Circuit(path_a / "circuit_config.json")
    circ_b = bluepysnap.Circuit(path_b / "circuit_config.json")
    mapping_a = load_json(path_a / "id_mapping.json")
    mapping_b = load_json(path_b / "id_mapping.json")

    assert set(circ_a.nodes.keys()) == set(circ_b.nodes.keys()), (
        f"Node populations differ: {set(circ_a.nodes.keys())} vs {set(circ_b.nodes.keys())}"
    )

    for pop_name in circ_a.nodes.keys():
        orig_a = np.asarray(mapping_a[pop_name]["original_id"], dtype=np.int64)
        orig_b = np.asarray(mapping_b[pop_name]["original_id"], dtype=np.int64)
        if strict_node_order:
            assert np.array_equal(orig_a, orig_b), (
                f"Population '{pop_name}' original_ids differ"
            )
        else:
            assert np.array_equal(np.sort(orig_a), np.sort(orig_b)), (
                f"Population '{pop_name}' original_ids differ"
            )
        assert mapping_a[pop_name]["original_name"] == mapping_b[pop_name]["original_name"], (
            f"Population '{pop_name}' original_name differs"
        )

    assert set(circ_a.edges.keys()) == set(circ_b.edges.keys()), (
        f"Edge populations differ: {set(circ_a.edges.keys())} vs {set(circ_b.edges.keys())}"
    )

    for edge_name in circ_a.edges.keys():
        edge_a = circ_a.edges[edge_name]
        edge_b = circ_b.edges[edge_name]
        src_pop = edge_a.source.name
        tgt_pop = edge_a.target.name

        # Build lookup tables: lut[new_id] -> original_id
        src_lut_a = _build_original_id_lut(mapping_a, src_pop)
        tgt_lut_a = _build_original_id_lut(mapping_a, tgt_pop)
        src_lut_b = _build_original_id_lut(mapping_b, src_pop)
        tgt_lut_b = _build_original_id_lut(mapping_b, tgt_pop)

        # Load edge source/target IDs
        edges_df_a = edge_a.get(edge_a.ids(), [Edge.SOURCE_NODE_ID, Edge.TARGET_NODE_ID])
        sgids_a = edges_df_a[Edge.SOURCE_NODE_ID].to_numpy()
        tgids_a = edges_df_a[Edge.TARGET_NODE_ID].to_numpy()

        edges_df_b = edge_b.get(edge_b.ids(), [Edge.SOURCE_NODE_ID, Edge.TARGET_NODE_ID])
        sgids_b = edges_df_b[Edge.SOURCE_NODE_ID].to_numpy()
        tgids_b = edges_df_b[Edge.TARGET_NODE_ID].to_numpy()

        # Vectorized remap to original IDs
        orig_sgids_a = src_lut_a[sgids_a]
        orig_tgids_a = tgt_lut_a[tgids_a]
        orig_sgids_b = src_lut_b[sgids_b]
        orig_tgids_b = tgt_lut_b[tgids_b]

        if strict_edge_order:
            assert np.array_equal(orig_sgids_a, orig_sgids_b) and np.array_equal(
                orig_tgids_a, orig_tgids_b
            ), f"Edge population '{edge_name}' differs (strict order)"
        else:
            pairs_a = _edges_to_sorted_pairs(orig_sgids_a, orig_tgids_a)
            pairs_b = _edges_to_sorted_pairs(orig_sgids_b, orig_tgids_b)
            assert np.array_equal(pairs_a, pairs_b), (
                f"Edge population '{edge_name}' differs"
            )
