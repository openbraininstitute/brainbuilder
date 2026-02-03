def create_appendable_dataset(h5_root, name, dtype, chunksize=1000):
    """create an h5 appendable dataset at `h5_root` w/ `name`"""
    h5_root.create_dataset(
        name,
        dtype=dtype,
        chunks=(chunksize,),
        shape=(0,),
        maxshape=(None,),
    )


def append_to_dataset(dset, values):
    """append `values` to `dset`, which should be an appendable dataset"""
    dset.resize(dset.shape[0] + len(values), axis=0)
    dset[-len(values) :] = values
