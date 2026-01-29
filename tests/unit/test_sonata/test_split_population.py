# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import utils as test_utils
from numpy.testing import assert_array_equal

from brainbuilder.utils.utils import load_json
from brainbuilder.utils.sonata import split_population
import bluepysnap
from brainbuilder.utils import utils

DATA_PATH = (Path(__file__).parent / "../data/sonata/split_population/").resolve()

def make_edge_mapping_df(old_ids):
    """
    Convert a list/array of old IDs into a DataFrame suitable for edge_mappings.
    
    Parameters
    ----------
    old_ids : list or np.ndarray
        Array of old IDs.
    
    Returns
    -------
    pd.DataFrame
        Indexed by old IDs, column 'new_id' = 0..N-1
    """
    old_ids = np.asarray(old_ids)  # ensure NumPy array
    return pd.DataFrame(
        {"new_id": np.arange(len(old_ids), dtype=np.int64)},
        index=old_ids
    )

def dict_to_h5_group(data: dict, tmp_path, root_name="root"):
    """
    Convert a nested dict into a real h5py.Group.

    - dict → group
    - numpy array / list → dataset
    - strings → UTF-8 datasets
    """
    h5file = tmp_path / "test.h5"

    def _write(group, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                sub = group.create_group(str(k))
                _write(sub, v)
            else:
                arr = np.asarray(v)
                if arr.dtype.kind in {"U", "O"}:
                    arr = arr.astype("S")
                group.create_dataset(str(k), data=arr)

    with h5py.File(h5file, "w") as f:
        root = f.create_group(root_name)
        _write(root, data)

    # reopen read-only (realistic usage)
    f = h5py.File(h5file, "r")
    return f[root_name]

def _find_populations_by_path(networks, key, name):
    populations = {
        k: v
        for population in networks[key]
        for k, v in population["populations"].items()
        if population[f"{key}_file"] == name
    }
    return populations

def _check_edge_indices(nodes_file, edges_file):
    def check_index_consistency(node2range, range2edge, ids):
        for id_ in range(node2range.shape[0]):
            range_start, range_end = node2range[id_, :]
            for edge_start, edge_end in range2edge[range_start:range_end, :]:
                assert all(ids[edge_start:edge_end] == id_)

    with h5py.File(edges_file, "r") as h5edges, h5py.File(nodes_file, "r") as h5nodes:
        for pop_name in h5edges["edges"]:
            base_path = "edges/" + pop_name

            src_pop = h5edges[base_path + "/source_node_id"].attrs["node_population"]
            tgt_pop = h5edges[base_path + "/target_node_id"].attrs["node_population"]

            src_node2range = h5edges[base_path + "/indices/source_to_target/node_id_to_ranges"][:]
            tgt_node2range = h5edges[base_path + "/indices/target_to_source/node_id_to_ranges"][:]

            # check index length is equal to population size
            assert src_node2range.shape[0] == h5nodes["nodes"][src_pop]["node_type_id"].shape[0]
            assert tgt_node2range.shape[0] == h5nodes["nodes"][tgt_pop]["node_type_id"].shape[0]

            src_range2edge = h5edges[base_path + "/indices/source_to_target/range_to_edge_id"][:]
            tgt_range2edge = h5edges[base_path + "/indices/target_to_source/range_to_edge_id"][:]

            src_ids = h5edges[base_path + "/source_node_id"][:]
            tgt_ids = h5edges[base_path + "/target_node_id"][:]

            check_index_consistency(src_node2range, src_range2edge, src_ids)
            check_index_consistency(tgt_node2range, tgt_range2edge, tgt_ids)

def _check_biophysical_nodes(path, has_virtual, has_external, from_subcircuit=False):
    mapping = load_json(path / "id_mapping.json")

    def _orig_id_map(ids, pop):
        orig_offset = {"A": 1000, "B": 2000, "C": 3000, "V1": 8000, "V2": 9000}
        if from_subcircuit:
            return [_id + orig_offset[pop] for _id in ids]
        else:
            return ids

    def _orig_name_map(name):
        if from_subcircuit:
            return "All" + name
        else:
            return name

    assert mapping["A"] == {"new_id": [0, 1, 2], "parent_id": [0, 2, 4], "parent_name": "A", "original_id": _orig_id_map([0, 2, 4], "A"), "original_name": _orig_name_map("A")}
    assert mapping["B"] == {"new_id": [0, 1, 2, 3], "parent_id": [0, 2, 4, 5], "parent_name": "B", "original_id": _orig_id_map([0, 2, 4, 5], "B"), "original_name": _orig_name_map("B")}
    assert mapping["C"] == {"new_id": [0, 1, 2, 3], "parent_id": [0, 2, 4, 5], "parent_name": "C", "original_id": _orig_id_map([0, 2, 4, 5], "C"), "original_name": _orig_name_map("C")}

    with h5py.File(path / "nodes" / "nodes.h5", "r") as h5:
        nodes = h5["nodes"]
        for src in ("A", "B", "C"):
            assert src in nodes
            mtypes = utils.get_property(nodes[src]["0"], nodes[src]["0/mtype"][:], "mtype")
            assert np.all(mtypes == b"a")
                

        assert len(nodes["A/node_type_id"]) == 3
        assert len(nodes["B/node_type_id"]) == 4
        assert len(nodes["C/node_type_id"]) == 4

    with h5py.File(path / "edges" / "edges.h5", "r") as h5:
        edges = h5["edges"]

        assert "A__B" in edges
        assert list(edges["A__B"]["source_node_id"]) == [0, 0, 0]
        assert list(edges["A__B"]["target_node_id"]) == [0, 0, 1]  # 2nd is duplicate edge

        assert "B__A" not in edges

        assert "A__C" in edges
        assert list(edges["A__C"]["source_node_id"]) == [2]
        assert list(edges["A__C"]["target_node_id"]) == [2]

        assert "B__C" in edges
        assert list(edges["B__C"]["source_node_id"]) == [1]
        assert list(edges["B__C"]["target_node_id"]) == [1]

        assert "C__A" in edges
        assert list(edges["C__A"]["source_node_id"]) == [2]
        assert list(edges["C__A"]["target_node_id"]) == [2]

        assert "C__B" not in edges

        config = load_json(path / "circuit_config.json")

        assert "manifest" in config
        assert config["manifest"]["$BASE_DIR"] == "./"
        assert "networks" in config
        assert "nodes" in config["networks"]
        node_pops = _find_populations_by_path(
            config["networks"], "nodes", "$BASE_DIR/nodes/nodes.h5"
        )
        assert node_pops == {
            "A": {"type": "biophysical"},
            "B": {"type": "biophysical"},
            "C": {"type": "biophysical"},
        }
        assert "edges" in config["networks"]
        edge_pops = _find_populations_by_path(
            config["networks"], "edges", "$BASE_DIR/edges/edges.h5"
        )
        assert edge_pops == {
            "A__B": {"type": "chemical"},
            "A__C": {"type": "chemical"},
            "B__C": {"type": "chemical"},
            "C__A": {"type": "chemical"},
        }   

        virtual_node_count = sum(
            population["type"] == "virtual"
            for node in config["networks"]["nodes"]
            for population in node["populations"].values()
        )
        if has_virtual or has_external:
            assert virtual_node_count > 0
        else:
            assert virtual_node_count == 0
            assert len(node_pops) == 3
            assert len(edge_pops) == 4

        node_sets = load_json(path / "node_sets.json")
        assert node_sets == {
            "mtype_a": {"mtype": "a"},
            "mtype_b": {"mtype": "b"},
            "someA": {"node_id": [0, 1], "population": "A"},
            "allB": {"node_id": [0, 1, 2, 3], "population": "B"},
            "someB": {"node_id": [1, 2], "population": "B"},
            "noC": {"node_id": [], "population": "C"},
        }

        expected_mapping = {
            "A": {"new_id": [0, 1, 2], "parent_id": [0, 2, 4], "parent_name": "A", "original_id": _orig_id_map([0, 2, 4], "A"), "original_name": _orig_name_map("A")},
            "B": {"new_id": [0, 1, 2, 3], "parent_id": [0, 2, 4, 5], "parent_name": "B", "original_id": _orig_id_map([0, 2, 4, 5], "B"), "original_name": _orig_name_map("B")},
            "C": {"new_id": [0, 1, 2, 3], "parent_id": [0, 2, 4, 5], "parent_name": "C", "original_id": _orig_id_map([0, 2, 4, 5], "C"), "original_name": _orig_name_map("C")},
        }

        if has_virtual:
            expected_mapping["V1"] = {"new_id": [0, 1, 2], "parent_id": [0, 2, 3], "parent_name": "V1", "original_id": _orig_id_map([0, 2, 3], "V1"), "original_name": _orig_name_map("V1")}
            expected_mapping["V2"] = {"new_id": [0], "parent_id": [0], "parent_name": "V2", "original_id": _orig_id_map([0], "V2"), "original_name": _orig_name_map("V2")}

        if has_external:
            expected_mapping["external_A"] = {"new_id": [0, 1], "parent_id": [5, 3], "parent_name": "A", "original_id": _orig_id_map([5, 3], "A"), "original_name": _orig_name_map("A")}

        mapping = load_json(path / "id_mapping.json")
        assert mapping == expected_mapping

# -------------------------------
# Fixtures / Test data
# -------------------------------
@pytest.fixture
def edge_mappings():
    return {
        "A": make_edge_mapping_df([10, 15, 20]),
        "B": make_edge_mapping_df([30, 35, 40])
    }

@pytest.fixture
def orig_group_mock():
    return {
        "synapse_id": np.array([10, 15, 20, 30, 35, 40]),
        "synapse_population": np.array(["A", "A", "A", "B", "B", "B"])
    }

@pytest.fixture
def sl_mask_mock():
    override_map = {}
    return [(slice(0, 6), np.arange(6), override_map)]

# # -------------------------------
# # Tests
# # -------------------------------


def test__get_population_name():
    assert "src__dst__chemical" == split_population._get_population_name(src="src", dst="dst")
    assert "src" == split_population._get_population_name(src="src", dst="src")


def test__get_unique_population():
    nodes = DATA_PATH / "split_subcircuit" / "networks" / "nodes" / "nodes.h5"
    with h5py.File(nodes, "r") as h5:
        with pytest.raises(ValueError):
            split_population._get_unique_population(h5["nodes"])

    nodes = DATA_PATH / "nodes.h5"
    with h5py.File(nodes, "r") as h5:
        assert split_population._get_unique_population(h5["nodes"]) == "default"


def test__get_unique_group(tmp_path):
    nodes = DATA_PATH / "nodes.h5"
    with h5py.File(nodes, "r") as h5:
        parent = h5["nodes/default"]
        assert split_population._get_unique_group(parent)

    with h5py.File(tmp_path / "nodes.h5", "w") as h5:
        parent = h5.create_group("/edges/")
        parent.create_group("/pop_name/0")
        parent.create_group("/pop_name/1")
        with pytest.raises(ValueError):
            split_population._get_unique_group(parent)


def test__write_nodes(tmp_path):
    split_nodes = {
        "A": pd.DataFrame({"fake_prop": range(10)}, index=np.arange(10)),
        "B": pd.DataFrame({"fake_prop": range(5)}, index=np.arange(10, 15)),
    }
    split_population._write_nodes(tmp_path, split_nodes)
    assert (tmp_path / "nodes_A.h5").exists()
    assert (tmp_path / "nodes_B.h5").exists()

    with h5py.File(tmp_path / "nodes_A.h5", "r") as h5:
        assert_array_equal(h5["/nodes/A/0/fake_prop"], np.arange(10))
        assert_array_equal(h5["/nodes/A/node_type_id"], np.full(10, -1))
    with h5py.File(tmp_path / "nodes_B.h5", "r") as h5:
        assert_array_equal(h5["/nodes/B/0/fake_prop"], np.arange(5))
        assert_array_equal(h5["/nodes/B/node_type_id"], np.full(5, -1))

def test_add_synapse_id_override_basic(edge_mappings, orig_group_mock, sl_mask_mock):
    split_population._add_synapse_id_override(sl_mask_mock, edge_mappings, orig_group_mock)
    override_map = sl_mask_mock[0][2]
    expected_new_ids = np.array([0, 1, 2, 0, 1, 2], dtype=int)
    np.testing.assert_array_equal(override_map["synapse_id"], expected_new_ids)


def test_collect_sl_and_masks_basic(tmp_path):
    orig_edges = {
        "source_node_id": np.array([1, 2, 3, 4]),
        "target_node_id": np.array([10, 20, 30, 40]),
        "0": {"synapse_id": np.array([100, 101, 102, 103]),"synapse_population": np.array(["A", "A", "B", "B"]),}  # placeholder for utils.get_property
    }
    orig_edges = dict_to_h5_group(orig_edges, tmp_path)
    sgids_new = np.array([1, 3])
    tgids_new = np.array([10, 30])

    # temporarily replace _compute_syn_mask in split_population
    old_func = split_population._compute_syn_mask
    split_population._compute_syn_mask = lambda syn_ids, syn_pops, edge_mappings: np.ones_like(syn_ids, dtype=bool)

    ans = split_population._collect_sl_and_masks(
        orig_edges,
        h5_read_chunk_size=2,
        sgids_new=sgids_new,
        tgids_new=tgids_new,
        edge_mappings={}
    )

    # restore original function
    split_population._compute_syn_mask = old_func

    # expect slices and indices where both sgid and tgid are in the new sets
    expected_indices = [
        (slice(0, 2), np.array([0])),
        (slice(2, 4), np.array([0]))
    ]

    # flatten ans to slices + indices
    flat_ans = [(sl, idxs) for sl, idxs, _ in ans]

    # just check that returned indices match expected
    for (sl, idxs), (exp_sl, exp_idxs) in zip(flat_ans, expected_indices):
        assert sl.start == exp_sl.start and sl.stop == exp_sl.stop
        np.testing.assert_array_equal(idxs, exp_idxs)

def test_collect_lib_id_mapping_basic_h5(tmp_path):
    file = h5py.File(tmp_path / "test.h5", "w")
    dset = file.create_dataset("pop", data=[10, 20, 30])
    group0 = {
        "@library": {"pop": None},
        "pop": dset
    }
    sl_mask = [(slice(0, 3), np.array([0, 1, 2]), {})]

    lib_map = split_population._collect_lib_id_mapping(sl_mask, group0)
    df = lib_map["pop"]
    np.testing.assert_array_equal(df.index.to_numpy(), [10, 20, 30])
    np.testing.assert_array_equal(df["new_id"].to_numpy(), [0, 1, 2])
    

def test_basic_mapping():
    syn_ids = np.array([10, 20, 30, 40])
    syn_pops = np.array(["A", "A", "B", "B"])
    edge_mappings = {
        "A": make_edge_mapping_df([10, 15, 20]),
        "B": make_edge_mapping_df([30, 35, 40]),
    }

    mask = split_population._compute_syn_mask(syn_ids, syn_pops, edge_mappings)
    
    assert np.all(mask == np.array([True, True, True, True]))
    

def test_out_of_bounds_ids():
    syn_ids = np.array([5, 25, 50])
    syn_pops = np.array(["A", "A", "B"])
    edge_mappings = {
        "A": make_edge_mapping_df([10, 20]),
        "B": make_edge_mapping_df([30, 40]),
    }

    mask = split_population._compute_syn_mask(syn_ids, syn_pops, edge_mappings)
    
    assert np.all(mask == np.array([False, False, False]))

def test_partial_matches():
    syn_ids = np.array([10, 25, 30])
    syn_pops = np.array(["A", "A", "B"])
    edge_mappings = {
        "A": make_edge_mapping_df([10, 20]),
        "B": make_edge_mapping_df([30, 40]),
    }

    mask = split_population._compute_syn_mask(syn_ids, syn_pops, edge_mappings)
    
    assert np.all(mask == np.array([True, False, True]))

def test_empty_input():
    syn_ids = np.array([], dtype=int)
    syn_pops = np.array([], dtype=str)
    edge_mappings = {"A": make_edge_mapping_df([10, 20])}

    mask = split_population._compute_syn_mask(syn_ids, syn_pops, edge_mappings)
    
    assert mask.size == 0

def test__get_node_id_mapping():
    split_nodes = {
        "A": pd.DataFrame(index=np.arange(0, 10)),
        "B": pd.DataFrame(index=np.arange(10, 15)),
    }
    ret = split_population._get_node_id_mapping(split_nodes)
    assert len(ret) == 2
    assert ret["A"].new_id.to_list() == list(range(10))
    assert ret["B"].new_id.to_list() == list(range(5))


def test__split_population_by_attribute():
    # nodes.h5 contains 3 nodes with mtypes "L2_X", "L6_Y", "L6_Y"
    nodes_path = DATA_PATH / "nodes.h5"
    ret = split_population._split_population_by_attribute(nodes_path, "mtype")
    assert len(ret) == 2
    assert isinstance(ret["L2_X"], pd.DataFrame)

    assert len(ret["L2_X"]) == 1
    assert ret["L2_X"].mtype.unique()[0] == "L2_X"
    assert_array_equal(ret["L2_X"].index, [0])

    assert len(ret["L6_Y"]) == 2
    assert ret["L6_Y"].mtype.unique()[0] == "L6_Y"
    assert_array_equal(ret["L6_Y"].index, [1, 2])


def test__write_circuit_config(tmp_path):
    split_nodes = {
        "A": pd.DataFrame(index=np.arange(0, 10)),
        "B": pd.DataFrame(index=np.arange(10, 15)),
    }
    split_population._write_circuit_config(tmp_path, split_nodes)
    ret = load_json(tmp_path / "circuit_config.json")
    assert "manifest" in ret
    assert "networks" in ret
    assert "nodes" in ret["networks"]
    assert "edges" in ret["networks"]
    assert len(ret["networks"]["edges"]) == 0  # no edge files

    open(tmp_path / "edges_A.h5", "w").close()
    open(tmp_path / "edges_B.h5", "w").close()
    open(tmp_path / "edges_A__B__chemical.h5", "w").close()
    split_population._write_circuit_config(tmp_path, split_nodes)
    ret = load_json(tmp_path / "circuit_config.json")
    assert len(ret["networks"]["edges"]) == 3


@pytest.mark.parametrize(
    "id_mapping, h5_read_chunk_size, expected_dir",
    [
        (
            {
                # edges: A -> B (2), B -> A, B -> B
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[5, 4, 3, 0]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[1, 2]),
            },
            10,
            DATA_PATH / "01",
        ),
        (
            {
                # edges: A -> A (4)
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[3, 2, 1, 0]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[5, 4]),
            },
            10,
            DATA_PATH / "02",
        ),
        (
            {
                # edges: B -> B (4), reduced chunk size
                "A": pd.DataFrame({"new_id": np.arange(3)}, index=[5, 4, 3]),
                "B": pd.DataFrame({"new_id": np.arange(3)}, index=[2, 1, 0]),
            },
            3,
            DATA_PATH / "03",
        ),
        (
            {
                # edges: A -> A, A -> B (3)
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[2, 0, 4, 5]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[1, 3]),
            },
            10,
            DATA_PATH / "04",
        ),
        (
            {
                # edges: B -> B, B -> A (3)
                "A": pd.DataFrame({"new_id": np.arange(4)}, index=[1, 3, 4, 5]),
                "B": pd.DataFrame({"new_id": np.arange(2)}, index=[2, 0]),
            },
            10,
            DATA_PATH / "05",
        ),
    ],
)
def test__write_edges(tmp_path, id_mapping, h5_read_chunk_size, expected_dir):
    # edges.h5 contains the following edges:
    # '/edges/default/source_node_id': [2, 0, 0, 2]
    # '/edges/default/target_node_id': [0, 1, 1, 1]
    edges_path = DATA_PATH / "edges.h5"
    # iterate over different id_mappings to split the edges in different ways
    split_population._write_edges(
        tmp_path,
        edges_path,
        id_mapping,
        expect_to_use_all_edges=True,
        h5_read_chunk_size=h5_read_chunk_size,
    )
    test_utils.assert_h5_dirs_equal(tmp_path, expected_dir, pattern="edges_*.h5")


def test_split_population(tmp_path):
    attribute = "mtype"
    nodes_path = DATA_PATH / "nodes.h5"
    edges_path = DATA_PATH / "edges.h5"
    expected_dir = DATA_PATH / "00"

    split_population.split_population(tmp_path, attribute, nodes_path, edges_path)
    test_utils.assert_h5_dirs_equal(tmp_path, expected_dir)
    test_utils.assert_json_files_equal(
        tmp_path / "circuit_config.json", expected_dir / "circuit_config.json"
    )
    _check_edge_indices(nodes_path, edges_path)


def test__split_population_by_node_set():
    nodes_path = DATA_PATH / "nodes.h5"
    node_set_name = "L2_X"
    node_set_path = DATA_PATH / "node_sets.json"

    ret = split_population._split_population_by_node_set(nodes_path, node_set_name, node_set_path)

    assert len(ret) == 1
    assert isinstance(ret["L2_X"], pd.DataFrame)

    assert len(ret["L2_X"]) == 1
    assert ret["L2_X"].mtype.unique()[0] == "L2_X"
    assert_array_equal(ret["L2_X"].index, [0])


def test_simple_split_subcircuit(tmp_path):
    nodes_path = DATA_PATH / "nodes.h5"
    edges_path = DATA_PATH / "edges.h5"
    node_set_name = "L6_Y"
    node_set_path = DATA_PATH / "node_sets.json"

    split_population.simple_split_subcircuit(
        tmp_path, node_set_name, node_set_path, nodes_path, edges_path
    )

    assert (tmp_path / "nodes_L6_Y.h5").exists()
    with h5py.File(tmp_path / "nodes_L6_Y.h5", "r") as h5:
        population = h5["nodes/L6_Y/"]
        assert list(population["node_type_id"]) == [-1, -1]
        assert len(population["0/layer"]) == 2

    assert (tmp_path / "edges_L6_Y.h5").exists()
    with h5py.File(tmp_path / "edges_L6_Y.h5", "r") as h5:
        group = h5["edges/L6_Y/"]
        assert list(group["source_node_id"]) == [1]
        assert list(group["target_node_id"]) == [0]

def test__update_node_sets():
    ret = split_population._update_node_sets(node_sets={}, id_mapping={})
    assert ret == {}

    node_sets = {
        "CopiedNoNodeIds": ["All"],
        "MissingPopluationNotCopied": {"node_id": [15, 280, 397, 509, 555, 624, 651, 789]},
        "HasPopulationCopied": {
            "population": "A",
            "node_id": [
                # exist in the mapping
                3,
                4,
                5,
                # not in the mapping
                1003,
                1004,
                1005,
            ],
            "mtype": "foo",
        },
    }
    id_mapping = {
        "A": pd.DataFrame({"new_id": np.arange(4)}, index=[0, 5, 4, 3]),
    }
    ret = split_population._update_node_sets(node_sets, id_mapping)

    expected = {
        "CopiedNoNodeIds": ["All"],
        "HasPopulationCopied": {
            "node_id": [1, 2, 3],
            "population": "A",
            "mtype": "foo",
        },
    }
    assert ret == expected


def test_get_subcircuit_external_ids(monkeypatch):
    all_sgids = np.array([10, 10, 11, 11, 12, 12, 10, 10, 11, 11, 12, 12, 10, 10, 11, 11, 12, 12])
    all_tgids = np.array([10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12])

    def get_ids(wanted_src_ids, wanted_dst_ids):
        monkeypatch.setenv("H5_READ_CHUNKSIZE", "3")
        return split_population._get_subcircuit_external_ids(
            all_sgids, all_tgids, wanted_src_ids, wanted_dst_ids
        )

    wanted_src_ids = [10, 12]
    wanted_dst_ids = [10]
    expected = pd.DataFrame({"new_id": np.array([0, 1], np.uint)}, index=[10, 12])
    pd.testing.assert_frame_equal(expected, get_ids(wanted_src_ids, wanted_dst_ids))

    wanted_src_ids = [10]
    wanted_dst_ids = [10, 12, 11]
    expected = pd.DataFrame({"new_id": np.array([0], np.uint)}, index=[10])
    pd.testing.assert_frame_equal(expected, get_ids(wanted_src_ids, wanted_dst_ids))

    wanted_src_ids = [10, 12, 11]
    wanted_dst_ids = [10, 12]
    expected = pd.DataFrame({"new_id": np.array([0, 1, 2], np.uint)}, index=[10, 11, 12])
    pd.testing.assert_frame_equal(expected, get_ids(wanted_src_ids, wanted_dst_ids))




@pytest.mark.parametrize(
    "circuit,from_subcircuit",
    [
        (DATA_PATH / "split_subcircuit" / "circuit_config.json", False),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config.json"), False),
        (DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json", True),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json"), True),
    ],
)
def test_split_subcircuit_with_no_externals(tmp_path, circuit, from_subcircuit):
    node_set_name = "mtype_a"

    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit, do_virtual=False, create_external=False
    )

    _check_biophysical_nodes(path=tmp_path, has_virtual=False, has_external=False, from_subcircuit=from_subcircuit)


@pytest.mark.parametrize(
    "circuit,from_subcircuit",
    [
        (DATA_PATH / "split_subcircuit" / "circuit_config.json", False),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config.json"), False),
        (DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json", True),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json"), True),
    ],
)
def test_split_subcircuit_with_externals(tmp_path, circuit, from_subcircuit):
    node_set_name = "mtype_a"

    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit, do_virtual=False, create_external=True
    )

    _check_biophysical_nodes(path=tmp_path, has_virtual=False, has_external=True, from_subcircuit=from_subcircuit)

    mapping = load_json(tmp_path / "id_mapping.json")
    if from_subcircuit:
        assert mapping["external_A"] == {"new_id": [0, 1], "parent_id": [5, 3], "parent_name": "A", "original_id": [1005, 1003], "original_name": "AllA"}
    else:
        assert mapping["external_A"] == {"new_id": [0, 1], "parent_id": [5, 3], "parent_name": "A", "original_id": [5, 3], "original_name": "A"}
    assert "external_B" not in mapping
    assert "external_C" not in mapping

    with h5py.File(tmp_path / "external_A/nodes.h5", "r") as h5:
        assert len(h5["nodes/external_A/0/model_type"]) == 2

    with h5py.File(tmp_path / "external_A__B.h5", "r") as h5:
        assert h5["edges/external_A__B/source_node_id"].attrs["node_population"] == "external_A"
        assert h5["edges/external_A__B/target_node_id"].attrs["node_population"] == "B"
        assert len(h5["edges/external_A__B/0/delay"]) == 1
        assert h5["edges/external_A__B/0/delay"][0] == 0.5
        assert list(h5["edges/external_A__B/source_node_id"]) == [0]
        assert list(h5["edges/external_A__B/target_node_id"]) == [3]

    with h5py.File(tmp_path / "external_A__C.h5", "r") as h5:
        assert h5["edges/external_A__C/source_node_id"].attrs["node_population"] == "external_A"
        assert h5["edges/external_A__C/target_node_id"].attrs["node_population"] == "C"
        assert len(h5["edges/external_A__C/0/delay"]) == 2
        assert h5["edges/external_A__C/0/delay"][0] == 0.5
        assert h5["edges/external_A__C/0/delay"][1] == 0.5
        assert list(h5["edges/external_A__C/source_node_id"]) == [0, 1]
        assert list(h5["edges/external_A__C/target_node_id"]) == [3, 1]

    networks = load_json(tmp_path / "circuit_config.json")["networks"]
    assert len(networks["nodes"]) == 4
    assert len(networks["edges"]) == 6


@pytest.mark.parametrize(
    "circuit,from_subcircuit",
    [
        (DATA_PATH / "split_subcircuit" / "circuit_config.json", False),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config.json"), False),
        (DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json", True),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json"), True),
    ],
)
def test_split_subcircuit_with_virtual(tmp_path, circuit, from_subcircuit):
    node_set_name = "mtype_a"
    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit, do_virtual=True, create_external=False
    )

    _check_biophysical_nodes(path=tmp_path, has_virtual=True, has_external=False, from_subcircuit=from_subcircuit)

    with h5py.File(tmp_path / "V1" / "nodes.h5", "r") as h5:
        assert len(h5["nodes/V1/0/model_type"]) == 3

    with h5py.File(tmp_path / "V2" / "nodes.h5", "r") as h5:
        assert len(h5["nodes/V2/0/model_type"]) == 1

    with h5py.File(tmp_path / "edges" / "virtual_edges_V1.h5", "r") as h5:
        assert len(h5["edges/V1__A/0/delay"]) == 2
        assert list(h5["edges/V1__A/source_node_id"]) == [0, 2]
        assert list(h5["edges/V1__A/target_node_id"]) == [0, 0]

        assert len(h5["edges/V1__B/0/delay"]) == 1
        assert list(h5["edges/V1__B/source_node_id"]) == [1]
        assert list(h5["edges/V1__B/target_node_id"]) == [0]

    with h5py.File(tmp_path / "V2__C" / "virtual_edges_V2.h5", "r") as h5:
        assert len(h5["edges/V2__C/0/delay"]) == 1

        assert list(h5["edges/V2__C/source_node_id"]) == [0]
        assert list(h5["edges/V2__C/target_node_id"]) == [1]

    networks = load_json(tmp_path / "circuit_config.json")["networks"]

    # nodes
    for pop in (1, 2):
        virtual_pop = _find_populations_by_path(networks, "nodes", f"$BASE_DIR/V{pop}/nodes.h5")
        assert len(virtual_pop) == 1
        assert virtual_pop[f"V{pop}"] == {"type": "virtual"}

    # edges
    virtual_pop = _find_populations_by_path(
        networks, "edges", "$BASE_DIR/edges/virtual_edges_V1.h5"
    )
    assert virtual_pop == {"V1__A": {"type": "chemical"}, "V1__B": {"type": "chemical"}}

    virtual_pop = _find_populations_by_path(
        networks, "edges", "$BASE_DIR/V2__C/virtual_edges_V2.h5"
    )
    assert virtual_pop == {"V2__C": {"type": "chemical"}}


@pytest.mark.parametrize(
    "circuit,from_subcircuit",
    [
        (DATA_PATH / "split_subcircuit" / "circuit_config.json", False),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config.json"), False),
        (DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json", True),
        (bluepysnap.Circuit(DATA_PATH / "split_subcircuit" / "circuit_config_subcircuit.json"), True),
    ],
)
def test_split_subcircuit_with_empty_virtual(tmp_path, circuit, from_subcircuit):
    node_set_name = "mtype_b"
    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit, do_virtual=True, create_external=False
    )

    mapping = load_json(tmp_path / "id_mapping.json")

    def _orig_id_map(ids, pop):
        orig_offset = {"A": 1000, "B": 2000, "C": 3000, "V1": 8000, "V2": 9000}
        if from_subcircuit:
            return [_id + orig_offset[pop] for _id in ids]
        else:
            return ids

    def _orig_name_map(name):
        if from_subcircuit:
            return "All" + name
        else:
            return name

    assert mapping["A"] == {"new_id": [0, 1, 2], "parent_id": [1, 3, 5], "parent_name": "A", "original_id": _orig_id_map([1, 3, 5], "A"), "original_name": _orig_name_map("A")}
    assert mapping["B"] == {"new_id": [0, 1], "parent_id": [1, 3], "parent_name": "B", "original_id": _orig_id_map([1, 3], "B"), "original_name": _orig_name_map("B")}
    assert mapping["C"] == {"new_id": [0, 1], "parent_id": [1, 3], "parent_name": "C", "original_id": _orig_id_map([1, 3], "C"), "original_name": _orig_name_map("C")}
    assert mapping["V1"] == {"new_id": [0], "parent_id": [1], "parent_name": "V1", "original_id": _orig_id_map([1], "V1"), "original_name": _orig_name_map("V1")}
    assert "V2" not in mapping

    with h5py.File(tmp_path / "nodes" / "nodes.h5", "r") as h5:
        nodes = h5["nodes"]
        for src in ("A", ):  # Cagegorical m-types, i.e., @library created by Voxcell (#unique < 0.5 * #total)
            assert src in nodes
            mtypes = utils.get_property(nodes[src]["0"], nodes[src]["0/mtype"][:], "mtype")
            assert np.all(mtypes == b"b")
        for src in ("B", "C"):  # Non-cagegorical m-types, i.e., @library not created by Voxcell (not #unique < 0.5 * #total)
            assert src in nodes
            assert "@library" not in nodes[src]["0"].keys()
            assert np.all(nodes[src]["0/mtype"][:] == b"b")

        assert len(nodes["A/node_type_id"]) == 3
        assert len(nodes["B/node_type_id"]) == 2
        assert len(nodes["C/node_type_id"]) == 2

    with h5py.File(tmp_path / "V1" / "nodes.h5", "r") as h5:
        assert len(h5["nodes/V1/0/model_type"]) == 1

    assert not (tmp_path / "V2" / "nodes.h5").exists()

    with h5py.File(tmp_path / "edges" / "edges.h5", "r") as h5:
        edges = h5["edges"]

        assert "A__B" not in edges

        assert "B__A" in edges
        assert list(edges["B__A"]["source_node_id"]) == [0, 0]
        assert list(edges["B__A"]["target_node_id"]) == [0, 0]  # 2nd is duplicate edge

        assert "A__C" not in edges

        assert "B__C" not in edges

        assert "C__A" not in edges

        assert "C__B" in edges
        assert list(edges["C__B"]["source_node_id"]) == [1]
        assert list(edges["C__B"]["target_node_id"]) == [1]

    with h5py.File(tmp_path / "edges" / "virtual_edges_V1.h5", "r") as h5:
        assert list(h5["edges"].keys()) == ["V1__B"]
        assert len(h5["edges/V1__B/0/delay"]) == 1
        assert list(h5["edges/V1__B/source_node_id"]) == [0]
        assert list(h5["edges/V1__B/target_node_id"]) == [0]

    assert not (tmp_path / "edges" / "virtual_edges_V2.h5").exists()
    
    config = load_json(tmp_path / "circuit_config.json")

    assert "manifest" in config
    assert config["manifest"]["$BASE_DIR"] == "./"
    assert "networks" in config
    assert "nodes" in config["networks"]
    node_pops = _find_populations_by_path(
        config["networks"], "nodes", "$BASE_DIR/nodes/nodes.h5"
    )
    assert node_pops == {
        "A": {"type": "biophysical"},
        "B": {"type": "biophysical"},
        "C": {"type": "biophysical"},
    }
    assert "edges" in config["networks"]
    edge_pops = _find_populations_by_path(
        config["networks"], "edges", "$BASE_DIR/edges/edges.h5"
    )
    assert edge_pops == {
        "B__A": {"type": "chemical"},
        "C__B": {"type": "chemical"},
    }

    virtual_node_count = sum(
        population["type"] == "virtual"
        for node in config["networks"]["nodes"]
        for population in node["populations"].values()
    )
    assert virtual_node_count == 1

    networks = config["networks"]
    virtual_pop = _find_populations_by_path(networks, "nodes", f"$BASE_DIR/V1/nodes.h5")
    assert len(virtual_pop) == 1
    assert virtual_pop[f"V1"] == {"type": "virtual"}

    virtual_pop = _find_populations_by_path(
        networks, "edges", "$BASE_DIR/edges/virtual_edges_V1.h5"
    )
    assert virtual_pop == {"V1__B": {"type": "chemical"}}

    node_sets = load_json(tmp_path / "node_sets.json")
    assert node_sets == {
        "mtype_a": {"mtype": "a"},
        "mtype_b": {"mtype": "b"},
        "someA": {"node_id": [0], "population": "A"},
        "allB": {"node_id": [0, 1], "population": "B"},
        "someB": {"node_id": [1], "population": "B"},
        "noC": {"node_id": [], "population": "C"},
    }


def test_split_subcircuit_edge_indices(tmp_path):
    node_set_name = "mtype_a"
    circuit_config_path = str(DATA_PATH / "split_subcircuit" / "circuit_config.json")

    split_population.split_subcircuit(
        tmp_path, node_set_name, circuit_config_path, do_virtual=False, create_external=False
    )

    _check_biophysical_nodes(path=tmp_path, has_virtual=False, has_external=False)

    nodes_path = tmp_path / "nodes" / "nodes.h5"
    edges_path = tmp_path / "edges" / "edges.h5"
    _check_edge_indices(nodes_path, edges_path)


def test_copy_edge_attributes_advanced(tmp_path):
    """
    Test the _copy_edge_attributes function with a non-trivial edge population,
    including edge groups, edge attributes, and a library dataset.

    This test verifies that:
    1. Edges are correctly filtered according to the provided source/target node mappings.
    2. Source and target node IDs are remapped to new IDs as specified in the mapping DataFrame.
    3. Edge group datasets (e.g., 'weight', 'lib_var') are correctly copied and sliced.
    4. Library datasets in '@library' are correctly filtered and preserved, including string datasets.
    5. Chunked reading (h5_read_chunk_size=3) does not affect correctness.

    Steps performed:
    - Creates a minimal SONATA-style HDF5 input with 10 edges, a weight dataset, an integer library 
      dataset, and a string library dataset.
    - Defines an identity mapping for a subset of nodes to simulate filtering.
    - Calls _copy_edge_attributes to copy and filter edges to a new HDF5 output file.
    - Verifies that:
        * 'keep' indexes match the filtered edges.
        * source_node_id and target_node_id are correctly remapped.
        * edge group attributes and library datasets are copied and filtered as expected.

    The test ensures that both numeric and string library datasets are correctly handled
    when edges are filtered and remapped, making it a comprehensive verification of
    the edge-copying logic.
    """
    infile = tmp_path / "in.h5"
    outfile = tmp_path / "out.h5"

    # ----------------------
    # Build minimal SONATA input
    # ----------------------
    with h5py.File(infile, "w") as f:
        edges = f.create_group("edges/pop_name")
        edges.create_dataset(
            "source_node_id",
            data=np.arange(10, dtype=np.uint64),
            maxshape=(None,),
        )
        edges.create_dataset(
            "target_node_id",
            data=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.uint64),
            maxshape=(None,),
        )

        # mandatory edge group
        egrp = edges.create_group('0')
        egrp.create_dataset(
            "weight",
            data=np.arange(10, dtype=np.uint64),
            maxshape=(None,),
        )
        egrp.create_dataset(
            "lib_var",
            data=np.array([0, 0, 2, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint64),
            maxshape=(None,),
        )
        lib_group = egrp.create_group('@library')
        lib_group.create_dataset(
            "lib_var",
            data=np.array(["0", "01", "2"], dtype=h5py.string_dtype(encoding="utf-8")),
            maxshape=(None,),
        )

        egrp.create_dataset(
            "synapse_id",
            data=np.arange(100, 110, dtype=np.uint64),
            maxshape=(None,),
        )
        egrp.create_dataset(
            "synapse_population",
            data=np.array([0, 0, 0, 2, 1, 1, 1, 1, 1, 1], dtype=np.uint64),
            maxshape=(None,),
        )
        lib_group.create_dataset(
            "synapse_population",
            data=np.array(["parent_edge_pop", "parent_edge_pop2", "parent_edge_pop3"], dtype=h5py.string_dtype(encoding="utf-8")),
            maxshape=(None,),
        )

    # ----------------------
    # Identity mappings
    # ----------------------
    mapping = pd.DataFrame(
        {"new_id": np.arange(5, dtype=np.uint64)},
        index=np.array([2, 3, 4, 7, 8], dtype=np.uint64),
    )

    edge_mappings = {b"parent_edge_pop": make_edge_mapping_df([100, 101, 102]),
                     b"parent_edge_pop2": make_edge_mapping_df([104, 105, 106, 107, 108, 109]),
                     b"parent_edge_pop3": make_edge_mapping_df([])}


    # ----------------------
    # Run
    # ----------------------
    with h5py.File(infile, "r") as h5in, h5py.File(outfile, "w") as h5out:
        edge_write_config = split_population.EdgeWriteConfig(
            src_node_name="src",
            dst_node_name="dst",
            src_edge_name="pop_name",
            dst_edge_name="pop_name_var",
            src_mapping=mapping,
            dst_mapping=mapping,
            h5_read_chunk_size=3,  # FORCE chunking
        )
        keep = split_population._copy_edge_attributes(
            h5in=h5in,
            h5out=h5out,
edge_write_config=edge_write_config,
edge_mappings=edge_mappings
        )

    # ----------------------
    # Verification
    # ----------------------
    with h5py.File(outfile, "r") as f:
        out_edges = f["edges/pop_name_var"]
        src_ids = out_edges["source_node_id"][:]
        tgt_ids = out_edges["target_node_id"][:]

        # keep should match the indices in the original dataset that were kept
        orig_src = np.arange(10, dtype=np.uint64)
        orig_tgt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.uint64)
        # 3 is erased by the synapse_id. the others by the nodes
        expected_keep = np.array([2, 7])  # indices of edges whose source nodes are in mapping.index
        np.testing.assert_array_equal(keep.index.to_numpy(), expected_keep)

        # source_node_id remapped correctly
        expected_src = mapping.loc[orig_src[expected_keep]]["new_id"].to_numpy()
        np.testing.assert_array_equal(src_ids, expected_src)

        # target_node_id remapped correctly
        expected_tgt = mapping.loc[orig_tgt[expected_keep]]["new_id"].to_numpy()
        np.testing.assert_array_equal(tgt_ids, expected_tgt)

        # --- Library datasets ---
        lib_grp = out_edges['0']["@library"]
        lib_data = lib_grp["lib_var"][:]
        np.testing.assert_array_equal(lib_data, [b'01', b'2'])
        lib_var_ds = out_edges['0']["lib_var"][:]
        np.testing.assert_array_equal(lib_var_ds, [1, 0])
        lib_data = lib_grp["synapse_population"][:]
        np.testing.assert_array_equal(lib_data, [b"parent_edge_pop", b"parent_edge_pop2"])
        lib_var_ds = out_edges['0']["synapse_population"][:]
        np.testing.assert_array_equal(lib_var_ds, [0, 1])

        lib_var_ds = out_edges['0']["synapse_id"][:]
        # 102 is the 3rd element of "parent_edge_pop"
        # 107 is the 4th element of "parent_edge_pop2"
        np.testing.assert_array_equal(lib_var_ds, [2, 3])


def test_copy_edge_attributes_empty_edge_mapping(tmp_path):
    """
    Test _copy_edge_attributes with edge_mappings containing empty DataFrames.
    Verifies that edges with no mapping are correctly ignored and no errors occur.
    """
    infile = tmp_path / "in_empty.h5"
    outfile = tmp_path / "out_empty.h5"

    # ----------------------
    # Build minimal SONATA input
    # ----------------------
    with h5py.File(infile, "w") as f:
        edges = f.create_group("edges/pop_name")
        edges.create_dataset(
            "source_node_id",
            data=np.arange(10, dtype=np.uint64),
            maxshape=(None,),
        )
        edges.create_dataset(
            "target_node_id",
            data=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.uint64),
            maxshape=(None,),
        )

        egrp = edges.create_group('0')
        egrp.create_dataset(
            "weight",
            data=np.arange(10, dtype=np.uint64),
            maxshape=(None,),
        )
        egrp.create_dataset(
            "lib_var",
            data=np.array([0, 0, 2, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint64),
            maxshape=(None,),
        )
        lib_group = egrp.create_group('@library')
        lib_group.create_dataset(
            "lib_var",
            data=np.array(["0", "01", "2"], dtype=h5py.string_dtype(encoding="utf-8")),
            maxshape=(None,),
        )

        egrp.create_dataset(
            "synapse_id",
            data=np.arange(100, 110, dtype=np.uint64),
            maxshape=(None,),
        )
        egrp.create_dataset(
            "synapse_population",
            data=np.array([0, 0, 0, 2, 1, 1, 1, 1, 1, 1], dtype=np.uint64),
            maxshape=(None,),
        )
        lib_group.create_dataset(
            "synapse_population",
            data=np.array(["parent_edge_pop", "parent_edge_pop2", "parent_edge_pop3"], dtype=h5py.string_dtype(encoding="utf-8")),
            maxshape=(None,),
        )

    # ----------------------
    # Identity mapping
    # ----------------------
    mapping = pd.DataFrame(
        {"new_id": np.arange(5, dtype=np.uint64)},
        index=np.array([2, 3, 4, 7, 8], dtype=np.uint64),
    )

    # Empty edge mappings
    edge_mappings = {
        b"parent_edge_pop": make_edge_mapping_df([]),
        b"parent_edge_pop2": make_edge_mapping_df([]),
        b"parent_edge_pop3": make_edge_mapping_df([])
    }

    # ----------------------
    # Run
    # ----------------------
    with h5py.File(infile, "r") as h5in, h5py.File(outfile, "w") as h5out:
        edge_write_config = split_population.EdgeWriteConfig(
            src_node_name="src",
            dst_node_name="dst",
            src_edge_name="pop_name",
            dst_edge_name="pop_name_var",
            src_mapping=mapping,
            dst_mapping=mapping,
            h5_read_chunk_size=3,
            
        )
        keep = split_population._copy_edge_attributes(
            h5in=h5in,
            h5out=h5out,
edge_write_config=edge_write_config,
edge_mappings=edge_mappings
        )

    # ----------------------
    # Verification
    # ----------------------
    with h5py.File(outfile, "r") as f:
        out_edges = f["edges/pop_name_var"]

        # All edge library mappings are empty, so no edges should be kept
        assert len(keep) == 0
        assert out_edges["source_node_id"][:].size == 0
        assert out_edges["target_node_id"][:].size == 0

        # Edge group datasets should also be empty
        for ds_name in ["weight", "lib_var", "synapse_id", "synapse_population"]:
            assert out_edges['0'][ds_name][:].size == 0
        for ds_name in ["lib_var", "synapse_population"]:
            assert out_edges['0']["@library"][ds_name][:].size == 0
