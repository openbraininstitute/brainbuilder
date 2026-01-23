from pathlib import Path


def gather_layout_from_networks(networks):
    """find the layout of the nodes and edges files, return a dict of the name -> relative path"""

    # Note: we are 'prioritizing' the layout of the config over the layout of the files on disk:
    # 1) the `nodes`/`edges` network keys will still have the same number of elements
    #    after writing the new config (unless populations aren't used)
    # 2) The layout of the files may be slightly different; if the config has a single population
    #    in the dict, the output population will be writen to $population_name/$original_filename.h5
    #    if it has multiple elements, it will be written to
    #    $original_parent_dir/$original_filename.h5
    #
    # See tests for more clarity
    node_populations_to_paths, edge_populations_to_paths = {}, {}

    def _extract_population_paths(key):
        """extract populations from `network_base`; return dictionary with their file path"""
        key_name = f"{key}_file"
        ret = {}
        for stanza in networks[key]:
            filename = Path(stanza[key_name]).name
            if len(stanza["populations"]) == 1:
                population = next(iter(stanza["populations"]))
                ret[population] = str(Path(population) / filename)
            else:
                # multiple populations; need to group them into the same file
                base_path = Path(stanza[key_name]).parent.name
                for population in stanza["populations"]:
                    ret[population] = str(Path(base_path) / filename)
        return ret

    node_populations_to_paths = _extract_population_paths("nodes")
    edge_populations_to_paths = _extract_population_paths("edges")

    return node_populations_to_paths, edge_populations_to_paths
import numpy as np


def get_property(node_group, data, prop_name):
    """
    Resolve one level of indirection for a node property using @library,
    extracting only unique entries and broadcasting back to the original shape.

    Args:
        node_group: h5py.Group for the node (e.g., nodes[src]["0"])
        data: np.ndarray of integer indices or array of values
        prop_name: str, property name

    Returns:
        np.ndarray: resolved values, shape matches `data`
    """
    if data.size == 0:
        return data

    if "@library" not in node_group or not np.issubdtype(data.dtype, np.integer):
        return data

    # --- 1. compute unique sorted indices used ---
    unique_idx, inverse_idx = np.unique(data, return_inverse=True)

    # --- 2. extract only the needed entries from library ---
    lib_dataset = node_group[f"@library/{prop_name}"][:]
    selected_values = lib_dataset[unique_idx]

    # --- 3. broadcast back using inverse indices ---
    return selected_values[inverse_idx]
