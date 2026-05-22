# SPDX-License-Identifier: Apache-2.0
"""IdMapping: nested dict structure for node ID remapping during subcircuit extraction."""

from pathlib import Path

import numpy as np
import pandas as pd

from brainbuilder import utils

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
    - Adding sources with automatic shift computation
    - Serialization to id_mapping.json with lazy original_id resolution
    """

    def __init__(self):
        self.data: dict[str, dict[str, pd.DataFrame]] = {}

    @classmethod
    def load(cls, path: Path) -> "IdMapping":
        """Reconstruct an IdMapping from an id_mapping.json file.

        Handles both single-source (parent_id/parent_name) and multi-source
        (parent2_id/parent2_name, etc.) entries.
        """
        raw = utils.load_json(path)
        obj = cls()
        for dest_pop, entry in raw.items():
            obj.data[dest_pop] = {}
            # First source
            parent_ids = entry[PARENT_IDS]
            source_name = entry[PARENT_NAME]
            n_first = len(parent_ids)
            obj.data[dest_pop][source_name] = pd.DataFrame(
                {NEW_IDS: entry[NEW_IDS][:n_first]},
                index=parent_ids,
            )
            # Additional sources (parent2_id, parent3_id, ...)
            i = 2
            offset = n_first
            while f"parent{i}_id" in entry:
                p_ids = entry[f"parent{i}_id"]
                p_name = entry[f"parent{i}_name"]
                n = len(p_ids)
                obj.data[dest_pop][p_name] = pd.DataFrame(
                    {NEW_IDS: entry[NEW_IDS][offset : offset + n]},
                    index=p_ids,
                )
                offset += n
                i += 1
        return obj

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

    def write(self, output: Path, parent_circ) -> str:
        """Write id_mapping.json, resolving original_id on the fly from parent provenance.

        For single-source populations, the format is unchanged (backward compatible).
        For multi-source populations, additional parentN_name/parentN_id fields are added
        for the 2nd, 3rd, etc. sources.

        Returns:
            The filename of the written mapping (relative to output).
        """
        provenance = parent_circ.config.get("components", {}).get("provenance", {})
        parent_mapping = None
        if "id_mapping" in provenance:
            parent_root = Path(parent_circ._circuit_config_path).parent
            parent_mapping = utils.load_json(parent_root / provenance["id_mapping"])

        mapping = {}
        for dest_pop, sources in self.data.items():
            all_new_ids = []
            all_orig_ids = []

            source_items = list(sources.items())
            first_source = source_items[0][0]

            # First source goes into parent_id / parent_name
            first_df = source_items[0][1]
            entry = {
                PARENT_IDS: first_df.index.tolist(),
                PARENT_NAME: first_source,
            }
            all_new_ids.extend(first_df[NEW_IDS].tolist())
            all_orig_ids.extend(
                self._resolve_original_ids(first_df.index, first_source, parent_mapping)
            )

            # Additional sources get parentN_id / parentN_name (N=2,3,...)
            for i, (source_pop, df) in enumerate(source_items[1:], start=2):
                entry[f"parent{i}_id"] = df.index.tolist()
                entry[f"parent{i}_name"] = source_pop
                all_new_ids.extend(df[NEW_IDS].tolist())
                all_orig_ids.extend(
                    self._resolve_original_ids(df.index, source_pop, parent_mapping)
                )

            entry[NEW_IDS] = all_new_ids
            entry[ORIG_IDS] = all_orig_ids

            if parent_mapping is not None and first_source in parent_mapping:
                entry[ORIG_NAME] = parent_mapping[first_source][ORIG_NAME]
            else:
                entry[ORIG_NAME] = first_source

            mapping[dest_pop] = entry

        mapping_fn = "id_mapping.json"
        utils.dump_json(output / mapping_fn, mapping)
        return mapping_fn
