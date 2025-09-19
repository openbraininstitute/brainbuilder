import os
import h5py
import pandas
import numpy

OR_NODE_GRP = "original_node_group"

def properties_group_to_df(grp):
    assert "@library" not in grp.keys(), "Groups with categorical properties not supported at the moment!"

    keys = list(grp.keys())
    df = pandas.DataFrame({
        k: grp[k][:]
        for k in keys
    })
    grpname = os.path.split(grp.name)[-1]
    mi = pandas.MultiIndex.from_frame(
        pandas.DataFrame({
            OR_NODE_GRP: [int(grpname)] * len(df),
            "index": range(len(df))
        })
    )
    df.index = mi
    return df

def group_lookup_df(grp):
    return pandas.MultiIndex.from_frame(
        pandas.DataFrame({
            OR_NODE_GRP: grp["node_group_id"][:],
            "index": grp["node_group_index"][:]
        })
    )

def concatenated_and_ordered_properties(grp):
    group_names = [_x for _x in grp.keys() if not _x.startswith("node_")]
    props_concat = pandas.concat(
        [
            properties_group_to_df(grp[k])
            for k in group_names
        ], axis=0
    ).fillna(-1)
    lo = group_lookup_df(grp)

    props_concat = props_concat.loc[lo]
    props_concat = props_concat.reset_index(0, drop=False)
    return props_concat

def write_nodeprops(grp, df_props):
    for _col in df_props.columns:
        grp.create_dataset(_col, data=df_props[_col].to_numpy())

def multi_group_to_single_group(node_file_in, nodepop, node_file_out):
    with h5py.File(node_file_out, "w") as h5out:
        with h5py.File(node_file_in, "r") as h5in:
            grpname = f"nodes/{nodepop}"
            grpout = h5out.create_group(grpname)
            grpin = h5in[grpname]

            nodeprops = concatenated_and_ordered_properties(grpin)
            write_nodeprops(grpout.create_group("0"), nodeprops)

            grpout.create_dataset("node_group_id",
                                  data=numpy.zeros(len(nodeprops),
                                                   dtype=grpin["node_group_id"].dtype))
            grpout.create_dataset("node_group_index",
                                  data=numpy.arange(len(nodeprops),
                                                   dtype=grpin["node_group_index"].dtype))
            for to_copy in ["node_id", "node_type_id"]:
                grpin.copy(to_copy, grpout)

if __name__ == "__main__":
    import sys
    multi_group_to_single_group(*sys.argv[1:])

