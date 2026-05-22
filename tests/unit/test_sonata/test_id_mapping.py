# SPDX-License-Identifier: Apache-2.0
"""Unit tests for brainbuilder.utils.sonata.id_mapping.IdMapping"""

import numpy as np
import pandas as pd

from brainbuilder.utils import load_json, dump_json
from brainbuilder.utils.sonata.id_mapping import IdMapping, NEW_IDS


def test_add_source_single():
    m = IdMapping()
    m.add_source("A", "A", [10, 20, 30])
    df = m.data["A"]["A"]
    assert list(df.index) == [10, 20, 30]
    assert list(df[NEW_IDS]) == [0, 1, 2]


def test_add_source_multi_source_shift():
    """Second source for same dest gets shifted IDs."""
    m = IdMapping()
    m.add_source("ext_A", "A", [10, 20])
    m.add_source("ext_A", "B", [5, 15])
    df_a = m.data["ext_A"]["A"]
    df_b = m.data["ext_A"]["B"]
    assert list(df_a[NEW_IDS]) == [0, 1]
    assert list(df_b[NEW_IDS]) == [2, 3]


def test_add_source_three_sources_shift():
    """Three sources accumulate shifts correctly."""
    m = IdMapping()
    m.add_source("dest", "s1", [0, 1])
    m.add_source("dest", "s2", [10, 11, 12])
    m.add_source("dest", "s3", [20])
    assert list(m.data["dest"]["s1"][NEW_IDS]) == [0, 1]
    assert list(m.data["dest"]["s2"][NEW_IDS]) == [2, 3, 4]
    assert list(m.data["dest"]["s3"][NEW_IDS]) == [5]


def test_add_source_deduplication_same_source():
    """Calling add_source with same (dest, source) skips existing IDs."""
    m = IdMapping()
    m.add_source("ext_A", "A", [10, 20, 30])
    m.add_source("ext_A", "A", [20, 30, 40])
    df = m.data["ext_A"]["A"]
    assert list(df.index) == [10, 20, 30, 40]
    assert list(df[NEW_IDS]) == [0, 1, 2, 3]


def test_add_source_deduplication_all_existing():
    """If all IDs already exist, nothing changes."""
    m = IdMapping()
    m.add_source("ext_A", "A", [10, 20])
    result = m.add_source("ext_A", "A", [10, 20])
    assert list(result.index) == [10, 20]
    assert list(result[NEW_IDS]) == [0, 1]


def test_add_source_deduplication_with_other_sources():
    """Dedup respects shift from other sources under same dest."""
    m = IdMapping()
    m.add_source("ext_A", "B", [100, 200])  # IDs 0, 1
    m.add_source("ext_A", "A", [10, 20])  # IDs 2, 3
    m.add_source("ext_A", "A", [20, 30])  # 20 exists, 30 is new -> ID 4
    df = m.data["ext_A"]["A"]
    assert list(df.index) == [10, 20, 30]
    assert list(df[NEW_IDS]) == [2, 3, 4]


def test_add_source_empty_ids():
    """Adding empty IDs creates an empty DataFrame."""
    m = IdMapping()
    m.add_source("A", "A", [])
    assert len(m.data["A"]["A"]) == 0


def test_add_source_numpy_array():
    """Works with numpy arrays."""
    m = IdMapping()
    m.add_source("A", "A", np.array([5, 10, 15]))
    assert list(m.data["A"]["A"].index) == [5, 10, 15]


def test_add_source_contiguous_ids():
    """All new_ids across sources form a contiguous 0..N-1 sequence."""
    m = IdMapping()
    m.add_source("dest", "s1", [0, 1, 2])
    m.add_source("dest", "s2", [10, 11])
    m.add_source("dest", "s3", [20, 21, 22, 23])
    all_ids = []
    for df in m.data["dest"].values():
        all_ids.extend(df[NEW_IDS].tolist())
    assert all_ids == list(range(9))


def test_node_count():
    m = IdMapping()
    m.add_source("A", "A", [10, 20, 30])
    assert m.node_count("A") == 3
    m.add_source("A", "B", [5, 15])
    assert m.node_count("A") == 5


def test_resolve_original_ids_no_parent():
    """Without parent mapping, original_id == parent_id."""
    result = IdMapping._resolve_original_ids(pd.Index([0, 1, 2]), "A", None)
    assert result == [0, 1, 2]


def test_resolve_original_ids_source_not_in_parent():
    """If source_pop not in parent_mapping, original_id == parent_id."""
    parent_mapping = {"B": {"original_id": [100, 200, 300]}}
    result = IdMapping._resolve_original_ids(pd.Index([0, 1, 2]), "A", parent_mapping)
    assert result == [0, 1, 2]


def test_resolve_original_ids_chains_through_parent():
    """Resolves original IDs by indexing into parent's original_id array."""
    parent_mapping = {"A": {"original_id": [1000, 1001, 1002, 1003, 1004]}}
    result = IdMapping._resolve_original_ids(pd.Index([0, 2, 4]), "A", parent_mapping)
    assert result == [1000, 1002, 1004]


def test_write_single_source(tmp_path):
    """First-level extraction: original_id == parent_id."""
    m = IdMapping()
    m.add_source("A", "A", [10, 20, 30])

    fn = m.write(tmp_path)
    assert fn == "id_mapping.json"

    result = load_json(tmp_path / "id_mapping.json")
    assert result == {
        "A": {
            "parent_id": [10, 20, 30],
            "new_id": [0, 1, 2],
            "parent_name": "A",
            "original_id": [10, 20, 30],
            "original_name": "A",
        }
    }


def test_write_multi_source(tmp_path):
    """Multiple sources for same dest get separate parentN fields."""
    m = IdMapping()
    m.add_source("ext_A", "A", [10, 20])
    m.add_source("ext_A", "B", [5, 15])

    m.write(tmp_path)
    result = load_json(tmp_path / "id_mapping.json")

    assert result["ext_A"]["parent_id"] == [10, 20]
    assert result["ext_A"]["parent_name"] == "A"
    assert result["ext_A"]["parent2_id"] == [5, 15]
    assert result["ext_A"]["parent2_name"] == "B"
    assert result["ext_A"]["new_id"] == [0, 1, 2, 3]
    assert result["ext_A"]["original_id"] == [10, 20, 5, 15]
    assert result["ext_A"]["original_name"] == "A"


def test_write_nested_extraction(tmp_path):
    """Nested extraction chains through parent provenance."""
    m = IdMapping()
    m.add_source("A", "A", [0, 2])

    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    dump_json(parent_dir / "id_mapping.json", {
        "A": {
            "parent_id": [10, 20, 30],
            "new_id": [0, 1, 2],
            "parent_name": "A",
            "original_id": [1000, 2000, 3000],
            "original_name": "OrigA",
        }
    })

    output = tmp_path / "output"
    output.mkdir()
    m.write(output, parent_mapping_path=parent_dir / "id_mapping.json")
    result = load_json(output / "id_mapping.json")

    assert result["A"]["parent_id"] == [0, 2]
    assert result["A"]["new_id"] == [0, 1]
    assert result["A"]["original_id"] == [1000, 3000]
    assert result["A"]["original_name"] == "OrigA"
    assert result["A"]["parent_name"] == "A"
