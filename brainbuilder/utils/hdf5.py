from pathlib import PurePosixPath

import h5py


def create_appendable_dataset(h5_root, name, dtype, chunksize=1000):
    """create an h5 appendable dataset at `h5_root` w/ `name`"""
    dset = h5_root.create_dataset(
        name,
        dtype=dtype,
        chunks=(chunksize,),
        shape=(0,),
        maxshape=(None,),
    )
    return dset


def append_to_dataset(dset, values):
    """append `values` to `dset`, which should be an appendable dataset"""
    dset.resize(dset.shape[0] + len(values), axis=0)
    dset[-len(values) :] = values


def copy_h5_filtered(src, dst, exclude_paths=None):
    """
    Recursively copy HDF5 content from src to dst, skipping paths in exclude_paths.
    Paths are relative, POSIX-style (no leading slash).

    Args:
        src (h5py.File or h5py.Group)
        dst (h5py.File or h5py.Group)
        exclude_paths (set[str] | None): Paths to skip
    """
    exclude_paths = set(exclude_paths or [])

    def ensure_group(root, path: PurePosixPath):
        g = root
        for part in path.parts:
            g = g.require_group(part)
        return g

    def walk(src_group, src_rel=PurePosixPath(), dst_rel=PurePosixPath()):
        for name, obj in src_group.items():
            src_path = src_rel / name
            src_path_str = str(src_path)

            if src_path_str in exclude_paths:
                continue

            dst_path = dst_rel / name

            if isinstance(obj, h5py.Group):
                ensure_group(dst, dst_path)
                walk(obj, src_path, dst_path)
            else:
                dst_parent = ensure_group(dst, dst_path.parent)
                src_group.copy(name, dst_parent, name=dst_path.name)

    walk(src)
