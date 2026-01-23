import numpy as np


def get_property(node_group, data, prop_name):
    """
    Resolve one level of indirection for a node property using @library.

    Args:
        node_group: h5py.Group for the node (e.g., nodes[src]["0"])
        data: np.ndarray, the column values (may be integers)
        prop_name: str, property name (e.g., "mtype")

    Returns:
        np.ndarray: resolved values (from @library if integers, else original data)
    """
    if "@library" in node_group and np.issubdtype(data.dtype, np.integer):
        return node_group[f"@library/{prop_name}"][data]

    return data
