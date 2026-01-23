import numpy as np


def get_property(node_group, data, prop_name):
    """
    Resolve one level of indirection for a node property using @library.

    ASSUMPTION:
    - data is an integer array
    - indices are increasing

    Args:
        node_group: h5py.Group for the node (e.g., nodes[src]["0"])
        data: np.ndarray of integer indices or array of values
        prop_name: str, property name

    Returns:
        np.ndarray: resolved values
    """
    if data.size == 0:
        return data

    if "@library" in node_group and np.issubdtype(data.dtype, np.integer):
        return node_group[f"@library/{prop_name}"][data]

    return data
