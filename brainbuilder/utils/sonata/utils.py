import numpy as np


def get_property(node_group, data, prop_name):
    """
    Resolve one level of indirection for a node property using @library
    without loading the full dataset into memory.

    Optimized: uses direct indexing if `data` is already sorted.

    Args:
        node_group: h5py.Group for the node (e.g., nodes[src]["0"])
        data: np.ndarray of integer indices
        prop_name: str, property name

    Returns:
        np.ndarray: resolved values
    """
    if "@library" in node_group and np.issubdtype(data.dtype, np.integer):
        library_ds = node_group[f"@library/{prop_name}"]

        if len(data) == 0:
            return np.empty_like(data)

        if np.all(data[:-1] <= data[1:]):  # already sorted
            return library_ds[data]

        # sort indices to satisfy h5py
        order = np.argsort(data)
        sorted_indices = data[order]
        fetched = library_ds[sorted_indices]
        return fetched[np.argsort(order)]

    return data
