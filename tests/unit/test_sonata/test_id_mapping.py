# SPDX-License-Identifier: Apache-2.0
"""Unit tests for brainbuilder.utils.sonata.id_mapping.IdMapping"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from brainbuilder.utils import load_json
from brainbuilder.utils.sonata.id_mapping import IdMapping, NEW_IDS


class TestAddSource:
    """Tests for IdMapping.add_source"""

    def test_single_source(self):
        m = IdMapping()
        m.add_source("A", "A", [10, 20, 30])
        df = m.data["A"]["A"]
        assert list(df.index) == [10, 20, 30]
        assert list(df[NEW_IDS]) == [0, 1, 2]

    def test_multi_source_shift(self):
        """Second source for same dest gets shifted IDs."""
        m = IdMapping()
        m.add_source("ext_A", "A", [10, 20])
        m.add_source("ext_A", "B", [5, 15])
        df_a = m.data["ext_A"]["A"]
        df_b = m.data["ext_A"]["B"]
        assert list(df_a[NEW_IDS]) == [0, 1]
        assert list(df_b[NEW_IDS]) == [2, 3]

    def test_three_sources_shift(self):
        """Three sources accumulate shifts correctly."""
        m = IdMapping()
        m.add_source("dest", "s1", [0, 1])
        m.add_source("dest", "s2", [10, 11, 12])
        m.add_source("dest", "s3", [20])
        assert list(m.data["dest"]["s1"][NEW_IDS]) == [0, 1]
        assert list(m.data["dest"]["s2"][NEW_IDS]) == [2, 3, 4]
        assert list(m.data["dest"]["s3"][NEW_IDS]) == [5]

    def test_deduplication_same_source(self):
        """Calling add_source with same (dest, source) skips existing IDs."""
        m = IdMapping()
        m.add_source("ext_A", "A", [10, 20, 30])
        # Add again with overlap — only 40 is new
        m.add_source("ext_A", "A", [20, 30, 40])
        df = m.data["ext_A"]["A"]
        assert list(df.index) == [10, 20, 30, 40]
        assert list(df[NEW_IDS]) == [0, 1, 2, 3]

    def test_deduplication_all_existing(self):
        """If all IDs already exist, nothing changes."""
        m = IdMapping()
        m.add_source("ext_A", "A", [10, 20])
        result = m.add_source("ext_A", "A", [10, 20])
        assert list(result.index) == [10, 20]
        assert list(result[NEW_IDS]) == [0, 1]

    def test_deduplication_with_other_sources(self):
        """Dedup respects shift from other sources under same dest."""
        m = IdMapping()
        m.add_source("ext_A", "B", [100, 200])  # IDs 0, 1
        m.add_source("ext_A", "A", [10, 20])  # IDs 2, 3
        m.add_source("ext_A", "A", [20, 30])  # 20 exists, 30 is new -> ID 4
        df = m.data["ext_A"]["A"]
        assert list(df.index) == [10, 20, 30]
        assert list(df[NEW_IDS]) == [2, 3, 4]

    def test_empty_ids(self):
        """Adding empty IDs creates an empty DataFrame."""
        m = IdMapping()
        m.add_source("A", "A", [])
        assert len(m.data["A"]["A"]) == 0

    def test_numpy_array_input(self):
        """Works with numpy arrays."""
        m = IdMapping()
        m.add_source("A", "A", np.array([5, 10, 15]))
        assert list(m.data["A"]["A"].index) == [5, 10, 15]

    def test_contiguous_ids_across_all_sources(self):
        """All new_ids across sources form a contiguous 0..N-1 sequence."""
        m = IdMapping()
        m.add_source("dest", "s1", [0, 1, 2])
        m.add_source("dest", "s2", [10, 11])
        m.add_source("dest", "s3", [20, 21, 22, 23])
        all_ids = []
        for df in m.data["dest"].values():
            all_ids.extend(df[NEW_IDS].tolist())
        assert all_ids == list(range(9))


class TestResolveOriginalIds:
    """Tests for IdMapping._resolve_original_ids"""

    def test_no_parent_mapping(self):
        """Without parent mapping, original_id == parent_id."""
        result = IdMapping._resolve_original_ids(
            pd.Index([0, 1, 2]), "A", None
        )
        assert result == [0, 1, 2]

    def test_source_not_in_parent(self):
        """If source_pop not in parent_mapping, original_id == parent_id."""
        parent_mapping = {"B": {"original_id": [100, 200, 300]}}
        result = IdMapping._resolve_original_ids(
            pd.Index([0, 1, 2]), "A", parent_mapping
        )
        assert result == [0, 1, 2]

    def test_chains_through_parent(self):
        """Resolves original IDs by indexing into parent's original_id array."""
        parent_mapping = {
            "A": {"original_id": [1000, 1001, 1002, 1003, 1004]}
        }
        result = IdMapping._resolve_original_ids(
            pd.Index([0, 2, 4]), "A", parent_mapping
        )
        assert result == [1000, 1002, 1004]


class TestWrite:
    """Tests for IdMapping.write"""

    def test_single_source_first_level(self, tmp_path):
        """First-level extraction: original_id == parent_id."""
        m = IdMapping()
        m.add_source("A", "A", [10, 20, 30])

        parent_circ = MagicMock()
        parent_circ.config = {}

        fn = m.write(tmp_path, parent_circ)
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

    def test_multi_source_merged_in_output(self, tmp_path):
        """Multiple sources for same dest are merged in the JSON output."""
        m = IdMapping()
        m.add_source("ext_A", "A", [10, 20])
        m.add_source("ext_A", "B", [5, 15])

        parent_circ = MagicMock()
        parent_circ.config = {}

        m.write(tmp_path, parent_circ)
        result = load_json(tmp_path / "id_mapping.json")

        assert result["ext_A"]["parent_id"] == [10, 20, 5, 15]
        assert result["ext_A"]["new_id"] == [0, 1, 2, 3]
        assert result["ext_A"]["original_id"] == [10, 20, 5, 15]
        assert result["ext_A"]["parent_name"] == "A"
        assert result["ext_A"]["original_name"] == "A"

    def test_nested_extraction_resolves_original(self, tmp_path):
        """Nested extraction chains through parent provenance."""
        m = IdMapping()
        m.add_source("A", "A", [0, 2])

        # Write a parent id_mapping.json
        parent_mapping = {
            "A": {
                "parent_id": [10, 20, 30],
                "new_id": [0, 1, 2],
                "parent_name": "A",
                "original_id": [1000, 2000, 3000],
                "original_name": "OrigA",
            }
        }
        from brainbuilder.utils import dump_json
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        dump_json(parent_dir / "id_mapping.json", parent_mapping)

        parent_circ = MagicMock()
        parent_circ.config = {
            "components": {"provenance": {"id_mapping": "id_mapping.json"}}
        }
        parent_circ._circuit_config_path = str(parent_dir / "circuit_config.json")

        output = tmp_path / "output"
        output.mkdir()
        m.write(output, parent_circ)
        result = load_json(output / "id_mapping.json")

        assert result["A"]["parent_id"] == [0, 2]
        assert result["A"]["new_id"] == [0, 1]
        assert result["A"]["original_id"] == [1000, 3000]
        assert result["A"]["original_name"] == "OrigA"
        assert result["A"]["parent_name"] == "A"
