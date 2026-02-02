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
