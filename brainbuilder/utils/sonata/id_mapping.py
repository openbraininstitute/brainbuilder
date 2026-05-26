# SPDX-License-Identifier: Apache-2.0
"""IdMapping: nested dict structure for node ID remapping during subcircuit extraction."""

import numpy as np
import pandas as pd

# name of field with ids that are valid in extracted circuit
NEW_IDS = "new_id"
# name of field with ids that are valid in parent circuit
PARENT_IDS = "parent_id"
# name of field with ids that are valid in original circuit
ORIG_IDS = "original_id"
# name of field with node population name in parent circuit
PARENT_NAME = "parent_name"
# name of field with node population name in original circuit
ORIG_NAME = "original_name"


class IdMapping:
    """Nested dict mapping destination_pop -> source_pop -> DataFrame(index=old_ids, columns=[new_id]).

    Encapsulates the id remapping logic for subcircuit extraction:
    - Adding sources with contiguous new_id assignment (each new source's IDs
      are offset by the count of already-assigned IDs so the full range is 0..N-1)
    - Serialization to id_mapping.json
    """

    def __init__(self):
        self.data: dict[str, dict[str, pd.DataFrame]] = {}

    @staticmethod
    def _resolve_original_ids(parent_ids, source_pop, parent_mapping):
        """Resolve original IDs by chaining through parent provenance.

        Args:
            parent_ids: Index of parent IDs to resolve.
            source_pop: The source population name in the parent circuit.
            parent_mapping: The parent's id_mapping.json (or None for first-level extraction).

        Returns:
            list: Original IDs corresponding to the parent IDs.
        """
        if parent_mapping is not None and source_pop in parent_mapping:
            parent_orig_ids = np.array(parent_mapping[source_pop][ORIG_IDS])
            return parent_orig_ids[parent_ids.astype(int)].tolist()
        return parent_ids.tolist()

    def node_count(self, dest_pop: str) -> int:
        """Return the total number of nodes for a destination population (max new_id + 1)."""
        return int(max(df[NEW_IDS].max() for df in self.data[dest_pop].values())) + 1

    def add_source(self, dest_pop: str, source_pop: str, old_ids) -> pd.DataFrame:
        """Add a source entry with shifted new_ids. Returns the created/updated DataFrame.

        If (dest_pop, source_pop) already exists, only IDs not already present are added.
        """
        if dest_pop not in self.data:
            self.data[dest_pop] = {}

        if source_pop in self.data[dest_pop]:
            existing_df = self.data[dest_pop][source_pop]
            old_ids = pd.Index(old_ids).difference(existing_df.index)
            if len(old_ids) == 0:
                return existing_df
            shift = sum(len(df) for df in self.data[dest_pop].values())
            new_df = pd.DataFrame(
                {NEW_IDS: shift + np.arange(len(old_ids), dtype=np.int64)},
                index=old_ids,
            )
            self.data[dest_pop][source_pop] = pd.concat([existing_df, new_df])
        else:
            shift = sum(len(df) for df in self.data[dest_pop].values())
            new_df = pd.DataFrame(
                {NEW_IDS: shift + np.arange(len(old_ids), dtype=np.int64)},
                index=old_ids,
            )
            self.data[dest_pop][source_pop] = new_df

        return self.data[dest_pop][source_pop]

    def to_dict(self, parent_mapping: dict | None = None) -> dict:
        """Build the id_mapping.json content as a dict.

        For single-source populations, the format is unchanged (backward compatible).
        For multi-source populations, additional parentN_name/parentN_id fields are added
        for the 2nd, 3rd, etc. sources.

        Args:
            parent_mapping: The parent's id_mapping.json content (or None for first-level
                extractions).

        Returns:
            Dict ready to be serialized to id_mapping.json.
        """
        mapping = {}
        for dest_pop, sources in self.data.items():
            entry = {}
            entry[NEW_IDS] = all_new_ids = []
            entry[ORIG_IDS] = all_orig_ids = []

            for i, (source_pop, df) in enumerate(sources.items(), start=1):
                id_key = PARENT_IDS if i == 1 else f"parent{i}_id"
                name_key = PARENT_NAME if i == 1 else f"parent{i}_name"
                entry[id_key] = df.index.tolist()
                entry[name_key] = source_pop
                all_new_ids.extend(df[NEW_IDS].tolist())
                all_orig_ids.extend(
                    self._resolve_original_ids(df.index, source_pop, parent_mapping)
                )

            first_source = entry[PARENT_NAME]
            if parent_mapping is not None and first_source in parent_mapping:
                entry[ORIG_NAME] = parent_mapping[first_source][ORIG_NAME]
            else:
                entry[ORIG_NAME] = first_source

            mapping[dest_pop] = entry

        return mapping
