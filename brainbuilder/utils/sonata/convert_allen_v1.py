import h5py
import pandas as pd
import numpy as np
from itertools import chain


def sonata_to_dataframe(sonata_file, file_type="nodes"):
    cells_df = pd.DataFrame()
    with h5py.File(sonata_file, "r") as h5f:
        population_names = list(h5f[f"/{file_type}"].keys())
        assert len(population_names) == 1, "Single population is supported only"
        population_name = population_names[0]

        population = h5f[f"/{file_type}/{population_name}"]
        assert "0" in population, 'Single group "0" is supported only'
        group = population["0"]

        for key in group.keys():
            cells_df[key] = group[key][()]
        type_id_key = "node_type_id" if file_type == "nodes" else "edge_type_id"
        cells_df[type_id_key] = population[type_id_key][()]
        res_pop = population_name
        if file_type == "edges":
            src_group = population["source_node_id"]
            tgt_group = population["target_node_id"]
            cells_df["source_node_id"] = src_group[()]
            cells_df["target_node_id"] = tgt_group[()]
            src_pop = src_group.attrs["node_population"]
            tgt_pop = tgt_group.attrs["node_population"]
            res_pop = (src_pop, tgt_pop)
    return cells_df, res_pop


def load_allen_nodes(nodes_file, node_types_file):
    node_types_df = pd.read_csv(node_types_file, sep=r"\s+")
    cells_df, node_population = sonata_to_dataframe(nodes_file, file_type="nodes")
    cells_df = cells_df.merge(
        node_types_df[
            ["node_type_id", "model_template", "morphology", "rotation_angle_zaxis", "model_type"]
        ],
        on="node_type_id",
        how="left",
    )
    # print(cells_df.loc[cells_df["node_type_id"] >= 100000121, "rotation_angle_yaxis"])
    cells_df["model_template"] = cells_df["model_template"].str.replace(
        r"^.*:([A-Za-z0-9_]+)(?:\.hoc)?$", r"hoc:\1", regex=True
    )
    cells_df["morphology"] = cells_df["morphology"].str.replace(r"\.[^.]+$", "", regex=True)
    cells_df["rotation_angle_zaxis"] = cells_df["rotation_angle_zaxis"].fillna(0)
    cells_df["morphology"] = cells_df["morphology"].fillna("None")

    for dummy_field in ["mtype", "etype"]:
        if dummy_field not in cells_df.columns:
            cells_df[dummy_field]="None"

    return cells_df, node_population


def load_allen_edges(edges_file, edge_types_file):
    edge_types_df = pd.read_csv(edge_types_file, sep=r"\s+")
    edges_df, pop = sonata_to_dataframe(edges_file, file_type="edges")
    assert len(pop) == 2, "Should return source and target population names for edges"
    edges_df = edges_df.merge(
        edge_types_df[["edge_type_id", "syn_weight", "weight_function", "weight_sigma", "delay"]], on="edge_type_id", how="left"
    )
    # edges_df.rename(columns={"syn_weight": "conductance"}, inplace=True)
    return edges_df, pop[0], pop[1]

def prepare_synapses(edges_df, nodes_df):
    adjust_synapse_weights(edges_df, nodes_df)
    edges_df_expanded = edges_df.loc[edges_df.index.repeat(edges_df["nsyns"])].reset_index(drop=True)
    return edges_df_expanded

def adjust_synapse_weights(edges_df, nodes_df):
    src_df = nodes_df.loc[edges_df["source_node_id"], ["tuning_angle", "x","z"]].reset_index(drop=True)
    tgt_df = nodes_df.loc[edges_df["target_node_id"], ["tuning_angle", "x","z"]].reset_index(drop=True)
    edges_df.loc[edges_df["weight_function"]=="DirectionRule_others", "conductance"] = DirectionRule_others(edges_df, src_df, tgt_df)
    edges_df.loc[edges_df["weight_function"]=="DirectionRule_EE", "conductance"] = DirectionRule_EE(edges_df, src_df, tgt_df)


def write_edges_from_dataframe(data_df, src_pop, tgt_pop, outfile):
    edge_population = f"{src_pop}__{tgt_pop}__chemical"
    group = outfile.create_group(f"/edges/{edge_population}")
    group_pop = group.create_group("0")
    dset = group.create_dataset("source_node_id", data=data_df["source_node_id"].to_numpy())
    dset.attrs["node_population"] = src_pop
    dset = group.create_dataset("target_node_id", data=data_df["target_node_id"].to_numpy())
    dset.attrs["node_population"] = tgt_pop
    group.create_dataset("edge_type_id", data=data_df["edge_type_id"].to_numpy())
    for attribute_name in set(data_df.columns) - set(
        ["source_node_id", "target_node_id", "edge_type_id"]
    ):
        group_pop.create_dataset(attribute_name, data=data_df[attribute_name].to_numpy())
    group_indices_src = group.create_group("indices/source_to_target")
    group_indices_tgt = group.create_group("indices/target_to_source")
    write_index_group(group_indices_src, data_df.groupby("source_node_id"))
    write_index_group(group_indices_tgt, data_df.groupby("target_node_id"))


def write_index_group(group, grouped_df):
    node_to_edge_ids = {
        key: indices_to_ranges(list(value)) for key, value in grouped_df.groups.items()
    }
    range_ids = ranges_per_node(node_to_edge_ids)
    edge_ids = list(chain.from_iterable(node_to_edge_ids.values()))
    group.create_dataset("node_id_to_ranges", data=range_ids)
    group.create_dataset("range_to_edge_id", data=edge_ids)


def indices_to_ranges(indices):
    if not indices:
        return []
    arr = np.sort(np.array(indices))
    # find where consecutive list breaks
    breaks = np.where(np.diff(arr) != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(arr)]))
    # equivalent to [[arr[s], arr[e-1]+1] for s, e in zip(starts, ends)]
    return np.stack((arr[starts], arr[ends - 1] + 1), axis=1)


def ranges_per_node(node_to_edge_ids):
    res = []
    start = 0
    for ranges in node_to_edge_ids.values():
        n_ranges = len(ranges)
        end = start + n_ranges
        res.append([start, end])
        start = end
    return res


def DirectionRule_others(edge_props, src_node, trg_node):
    """Adjust the synapse weight, copied from bmtk 
    """
    sigma = edge_props['weight_sigma']
    src_tuning = src_node['tuning_angle']
    tar_tuning = trg_node['tuning_angle']

    delta_tuning_180 = abs(abs((abs(tar_tuning - src_tuning) % 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180 / sigma) ** 2)
    return w_multiplier_180 * edge_props['syn_weight']


def DirectionRule_EE(edge_props, src_node, trg_node):
    """Adjust the synapse weight, copied from bmtk 
    """
    sigma = edge_props['weight_sigma']

    src_tuning = src_node['tuning_angle']
    x_src = src_node['x']
    z_src = src_node['z']

    tar_tuning = trg_node['tuning_angle']
    x_tar = trg_node['x']
    z_tar = trg_node['z']

    delta_tuning_180 = abs(abs((abs(tar_tuning - src_tuning) % 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180 / sigma) ** 2)

    delta_x = (x_tar - x_src) * 0.07
    delta_z = (z_tar - z_src) * 0.04

    theta_pref = tar_tuning * (np.pi / 180.)
    xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
    sigma_phase = 1.0
    phase_scale_ratio = np.exp(- (xz ** 2 / (2 * sigma_phase ** 2)))

    # To account for the 0.07 vs 0.04 dimensions. This ensures
    # the horizontal neurons are scaled by 5.5/4 (from the midpoint
    # of 4 & 7). Also, ensures the vertical is scaled by 5.5/7. This
    # was a basic linear estimate to get the numbers (y = ax + b).
    theta_tar_scale = abs(abs(abs(180.0 - abs(tar_tuning) % 360.0) - 90.0) - 90.0)
    phase_scale_ratio = phase_scale_ratio * (5.5 / 4 - 11. / 1680 * theta_tar_scale)

    return w_multiplier_180 * phase_scale_ratio * edge_props['syn_weight']

