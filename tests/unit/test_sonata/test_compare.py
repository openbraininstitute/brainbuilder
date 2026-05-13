# SPDX-License-Identifier: Apache-2.0
"""Tests for brainbuilder.utils.sonata.compare."""

import shutil
from pathlib import Path

import pytest

from brainbuilder.utils import dump_json, load_json
from brainbuilder.utils.sonata import split_population
from brainbuilder.utils.sonata.compare import assert_circuits_equal

SPLIT_SUBCIRCUIT_DATA_PATH = (Path(__file__).parent / "../data/sonata/split_subcircuit/").resolve()


def _split_custom_subcircuit(output, circuit_config, node_set_name, node_set_def,
                            do_virtual=False, create_external=False):
    """Run split_subcircuit after injecting a custom node_set into a copy of the circuit."""
    fixture = output.parent / (output.name + "_fixture")
    shutil.copytree(Path(circuit_config).parent, fixture)

    node_sets = load_json(fixture / "node_sets.json")
    node_sets.update(node_set_def)
    dump_json(fixture / "node_sets.json", node_sets)

    split_population.split_subcircuit(
        output, node_set_name, str(fixture / "circuit_config.json"),
        do_virtual=do_virtual, create_external=create_external
    )
    return output


def test_assert_circuits_equal_same_extraction(tmp_path):
    """Two identical extractions from the same source should compare equal."""
    node_set_def = {
        "subset": ["subset_popA", "subset_popB", "subset_popC"],
        "subset_popA": {"population": "A", "node_id": [0, 2, 4]},
        "subset_popB": {"population": "B", "node_id": [0, 1, 2, 3, 4, 5]},
        "subset_popC": {"population": "C", "node_id": [0, 1, 2, 3, 4, 5]},
    }

    circuit_config = str(SPLIT_SUBCIRCUIT_DATA_PATH / "circuit_config.json")

    path_a = _split_custom_subcircuit(
        tmp_path / "a", circuit_config, "subset", node_set_def,
        do_virtual=False, create_external=True
    )
    path_b = _split_custom_subcircuit(
        tmp_path / "b", circuit_config, "subset", node_set_def,
        do_virtual=False, create_external=True
    )

    # Should not raise
    assert_circuits_equal(path_a, path_b)
    assert_circuits_equal(path_a, path_b, strict_order=True)


def test_assert_circuits_equal_different_subsets_fail(tmp_path):
    """Two different extractions should fail comparison."""
    circuit_config = str(SPLIT_SUBCIRCUIT_DATA_PATH / "circuit_config.json")

    node_set_a = {
        "subset_a": ["subset_a_popA", "subset_a_popB", "subset_a_popC"],
        "subset_a_popA": {"population": "A", "node_id": [0, 2, 4]},
        "subset_a_popB": {"population": "B", "node_id": [0, 1, 2, 3, 4, 5]},
        "subset_a_popC": {"population": "C", "node_id": [0, 1, 2, 3, 4, 5]},
    }
    node_set_b = {
        "subset_b": ["subset_b_popA", "subset_b_popB", "subset_b_popC"],
        "subset_b_popA": {"population": "A", "node_id": [1, 3, 5]},
        "subset_b_popB": {"population": "B", "node_id": [0, 1, 2, 3, 4, 5]},
        "subset_b_popC": {"population": "C", "node_id": [0, 1, 2, 3, 4, 5]},
    }

    path_a = _split_custom_subcircuit(
        tmp_path / "a", circuit_config, "subset_a", node_set_a,
        do_virtual=False, create_external=True
    )
    path_b = _split_custom_subcircuit(
        tmp_path / "b", circuit_config, "subset_b", node_set_b,
        do_virtual=False, create_external=True
    )

    with pytest.raises(AssertionError):
        assert_circuits_equal(path_a, path_b)
