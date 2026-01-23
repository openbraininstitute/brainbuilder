import numpy as np


def get_property(node_group, data, prop_name):
    """
    Resolve one level of indirection for a node property using @library
    without loading the full dataset into memory.

    Works safely with unordered indices and duplicates.

    Args:
        node_group: h5py.Group for the node (e.g., nodes[src]["0"])
        data: np.ndarray of integer indices or array of values
        prop_name: str, property name

    Returns:
        np.ndarray: resolved values
    """
    # Only resolve via @library if it exists and data is integer indices
    if "@library" in node_group and np.issubdtype(data.dtype, np.integer):
        library_ds = node_group[f"@library/{prop_name}"]

        if len(data) == 0:
            return np.empty_like(data)

        # Sort indices to satisfy h5py
        order = np.argsort(data, kind="stable")
        sorted_indices = data[order]

        # Fetch values from HDF5 in sorted order
        fetched = library_ds[sorted_indices]

        # Reorder to match original `data`
        return fetched[np.argsort(order, kind="stable")]

    # Otherwise, just return data as-is
    return data
