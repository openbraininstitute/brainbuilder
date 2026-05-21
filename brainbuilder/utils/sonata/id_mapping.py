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
            all_parent_ids = []
            all_new_ids = []
            all_orig_ids = []

            for source_pop, df in sources.items():
                all_parent_ids.extend(df.index.tolist())
                all_new_ids.extend(df[NEW_IDS].tolist())
                all_orig_ids.extend(
                    self._resolve_original_ids(df.index, source_pop, parent_mapping)
                )

            # Derive parent_name and orig_name from the first source
            first_source = next(iter(sources))
            if parent_mapping is not None and first_source in parent_mapping:
                orig_name = parent_mapping[first_source][ORIG_NAME]
            else:
                orig_name = first_source

            mapping[dest_pop] = {
                PARENT_IDS: all_parent_ids,
                NEW_IDS: all_new_ids,
                PARENT_NAME: first_source,
                ORIG_IDS: all_orig_ids,
                ORIG_NAME: orig_name,
            }

        mapping_fn = "id_mapping.json"
        utils.dump_json(output / mapping_fn, mapping)
        return mapping_fn
