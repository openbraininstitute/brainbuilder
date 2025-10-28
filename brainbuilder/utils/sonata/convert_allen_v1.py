import h5py
import pandas as pd
import numpy as np
from itertools import chain
from brainbuilder import utils
from pathlib import Path


def sonata_to_dataframe(sonata_file, file_type="nodes"):
    cells_df = pd.DataFrame()
    with h5py.File(sonata_file, "r") as h5f:
        population_names = list(h5f[f"/{file_type}"].keys())
        assert len(population_names) == 1, "Single population is supported only"
        population_name = population_names[0]

        population = h5f[f"{file_type}/{population_name}"]
        assert "0" in population, "group '0' doesn't exst"
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
            ["node_type_id", "dynamics_params", "morphology", "rotation_angle_zaxis", "model_type"]
        ],
        on="node_type_id",
        how="left",
    )
    cells_df.rename(columns={"dynamics_params": "model_template"}, inplace=True)
    # hoc template name can not be started with number, prefix with BP_ where necessary
    cells_df["model_template"] = cells_df["model_template"].str.replace(
        r"^([0-9][A-Za-z0-9_]*)(?:\.json)?$|^([A-Za-z][A-Za-z0-9_]*)(?:\.json)?$",
        lambda m: f"hoc:BP_{m.group(1)}" if m.group(1) else f"hoc:{m.group(2)}",
        regex=True,
    )
    cells_df["morphology"] = cells_df["morphology"].str.replace(r"\.[^.]+$", "", regex=True)
    cells_df["rotation_angle_zaxis"] = cells_df["rotation_angle_zaxis"].fillna(0)
    cells_df["morphology"] = cells_df["morphology"].fillna("None")

    # add dummy attributes
    add_dummy_values(cells_df, ["mtype", "etype"], "None")

    return cells_df, node_population


def load_allen_edges(edges_file, edge_types_file):
    edge_types_df = pd.read_csv(edge_types_file, sep=r"\s+")
    edges_df, pop = sonata_to_dataframe(edges_file, file_type="edges")
    assert len(pop) == 2, "Should return source and target population names for edges"
    edges_df = edges_df.merge(
        edge_types_df[
            [
                "edge_type_id",
                "syn_weight",
                "weight_function",
                "weight_sigma",
                "delay",
                "dynamics_params",
            ]
        ],
        on="edge_type_id",
        how="left",
    )
    return edges_df, pop[0], pop[1]


def prepare_synapses(edges_df, nodes_df, precomputed_edges_file, syn_parameter_dir):
    adjust_synapse_weights(edges_df, nodes_df)
    edges_df = add_synapse_parameters(edges_df, syn_parameter_dir)
    add_dummy_values(edges_df, ["depression_time", "n_rrp_vesicles", "syn_type_id"], -1)
    edges_df_expanded = add_precomputed_synapse_locations(
        edges_df, nodes_df, precomputed_edges_file
    )
    return edges_df_expanded


def add_dummy_values(df, attribute_names, default_value):
    for attribute_name in attribute_names:
        if attribute_name not in df.columns:
            df[attribute_name] = default_value


def add_precomputed_synapse_locations(edges_df, nodes_df, precomputed_edges_file):
    # Read synapse location and weights from precomputed edges file
    syn_biophysical_df, syn_point_df = load_precomputed_edges_file(precomputed_edges_file)

    biophysical_gids = nodes_df.index[nodes_df["model_type"] == "biophysical"]
    point_gids = nodes_df.index[nodes_df["model_type"] == "point_process"]

    # For edges targeting point cells, multiple syn_weight by nsys
    mask_point = edges_df["target_node_id"].isin(point_gids)
    edges_df.loc[mask_point, "conductance"] *= edges_df.loc[mask_point, "nsyns"]
    # cross check with precompuated file to make sure the weights are correct
    assert np.allclose(edges_df.loc[mask_point, "conductance"], abs(syn_point_df["syn_weight"])), (
        "point syn weight is not consistent with the precomputed file"
    )

    # For edges targeting biophysical cells, expand synapses, apply precomputed sec_id and seg_x
    repeat_counts = edges_df["nsyns"].where(edges_df["target_node_id"].isin(biophysical_gids), 1)
    edges_df_expanded = edges_df.loc[edges_df.index.repeat(repeat_counts)].reset_index(drop=True)
    mask_biophysical = edges_df_expanded["target_node_id"].isin(biophysical_gids)
    assert np.allclose(
        edges_df_expanded.loc[mask_biophysical, "conductance"], syn_biophysical_df["syn_weight"]
    ), "biophysical syn weight is not consistent with the precomputed file"
    edges_df_expanded["afferent_section_id"] = -1
    edges_df_expanded["afferent_section_pos"] = -1.
    edges_df_expanded.loc[mask_biophysical, "afferent_section_id"] = syn_biophysical_df[
        "sec_id"
    ].to_numpy()  # row-to-row, not by index
    edges_df_expanded.loc[mask_biophysical, "afferent_section_pos"] = syn_biophysical_df[
        "sec_x"
    ].to_numpy()

    return edges_df_expanded


def add_synapse_parameters(edges_df, sym_parameter_dir):
    # We rename tau1, tau2 and erev with the BBP synapse parameter names in the output file, so that we can run with our simulator directly
    syn_params_map = {"dynamics_params": [], "facilitation_time": [], "decay_time": [], "u_syn": []}
    for json_file in edges_df["dynamics_params"].unique():
        params = utils.load_json(Path(sym_parameter_dir) / json_file)
        if params["level_of_detail"] == "exp2syn":
            syn_params_map["dynamics_params"].append(json_file)
            syn_params_map["facilitation_time"].append(params["tau1"])
            syn_params_map["decay_time"].append(params["tau2"])
            syn_params_map["u_syn"].append(params["erev"])
    # create a dataframe from syn_params_map
    syn_params_df = pd.DataFrame(syn_params_map)
    return edges_df.merge(syn_params_df, on="dynamics_params", how="left")


def load_precomputed_edges_file(precomputed_edges_file):
    res = []
    with h5py.File(precomputed_edges_file, "r") as h5f:
        population_names = list(h5f["/edges"].keys())
        assert len(population_names) == 1, "Single population is supported only"
        population_name = population_names[0]

        population = h5f[f"/edges/{population_name}"]
        for group_name in ["0", "1"]:
            assert group_name in population, f"group {group_name} doesn't exst"
            group = population[group_name]
            syn_weight = group["syn_weight"][()]
            sec_id = group["sec_id"][()] if "sec_id" in group else np.empty(len(syn_weight))
            sec_x = group["sec_x"][()] if "sec_x" in group else np.empty(len(syn_weight))
            res.append(pd.DataFrame({"syn_weight": syn_weight, "sec_id": sec_id, "sec_x": sec_x}))
    return res


def adjust_synapse_weights(edges_df, nodes_df):
    src_df = nodes_df.loc[edges_df["source_node_id"], ["tuning_angle", "x", "z"]].reset_index(
        drop=True
    )
    tgt_df = nodes_df.loc[edges_df["target_node_id"], ["tuning_angle", "x", "z"]].reset_index(
        drop=True
    )
    edges_df.loc[:, "conductance"] = edges_df["syn_weight"]  # default cond
    edges_df.loc[edges_df["weight_function"] == "DirectionRule_others", "conductance"] = (
        DirectionRule_others(edges_df, src_df, tgt_df)
    )
    edges_df.loc[edges_df["weight_function"] == "DirectionRule_EE", "conductance"] = (
        DirectionRule_EE(edges_df, src_df, tgt_df)
    )


def write_edges_from_dataframe(data_df, src_pop, tgt_pop, n_source_nodes, n_target_nodes, outfile):
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
    write_index_group(group_indices_src, data_df.groupby("source_node_id"), n_source_nodes)
    write_index_group(group_indices_tgt, data_df.groupby("target_node_id"), n_target_nodes)


def write_index_group(group, grouped_df, n_nodes):
    """Write the index group for nodes ids: [0, n_nodes-1]
    grouped_df.groups = {"node_id": [list of edge indices]}
    """
    node_to_edge_ids = dict.fromkeys(list(range(n_nodes)), [])
    for key, value in grouped_df.groups.items():
        node_to_edge_ids[key] = indices_to_ranges(list(value))
    range_ids = ranges_per_node(node_to_edge_ids)
    edge_ids = list(chain.from_iterable(node_to_edge_ids.values()))
    group.create_dataset("node_id_to_ranges", data=range_ids)
    group.create_dataset("range_to_edge_id", data=edge_ids)


def indices_to_ranges(indices):
    """Given a list of [indices], return list of [start, end) ranges .
    e.g. [0,1,2,7,8,10,11,12] -> [[0,3], [7,9], [10,13]]"""
    if not indices:
        return [0, 0]
    arr = np.sort(np.array(indices))
    # find where consecutive list breaks
    breaks = np.where(np.diff(arr) != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(arr)]))
    # equivalent to [[arr[s], arr[e-1]+1] for s, e in zip(starts, ends)]
    return np.stack((arr[starts], arr[ends - 1] + 1), axis=1)


def ranges_per_node(node_to_edge_ids):
    """Given list of [edge_ids], return list of [start, end) ranges.
    e.g. [[[0,3], [3,5], [5,8]], [[9,10]]] -> [[0,3],[3,4]]
    Range 0 -> ids[0,3), Range 1 -> ids[3,4), etc.]
    """
    res = []
    start = 0
    for ranges in node_to_edge_ids.values():
        n_ranges = len(ranges)
        end = start + n_ranges
        if n_ranges == 0:
            res.append([0, 0])
        else:
            res.append([start, end])
        start = end
    return res


def DirectionRule_others(edge_props, src_node, trg_node):
    """Adjust the synapse weight, copied from bmtk"""
    sigma = edge_props["weight_sigma"]
    src_tuning = src_node["tuning_angle"]
    tar_tuning = trg_node["tuning_angle"]

    delta_tuning_180 = abs(abs((abs(tar_tuning - src_tuning) % 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-((delta_tuning_180 / sigma) ** 2))
    return w_multiplier_180 * edge_props["syn_weight"]


def DirectionRule_EE(edge_props, src_node, trg_node):
    """Adjust the synapse weight, copied from bmtk"""
    sigma = edge_props["weight_sigma"]

    src_tuning = src_node["tuning_angle"]
    x_src = src_node["x"]
    z_src = src_node["z"]

    tar_tuning = trg_node["tuning_angle"]
    x_tar = trg_node["x"]
    z_tar = trg_node["z"]

    delta_tuning_180 = abs(abs((abs(tar_tuning - src_tuning) % 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-((delta_tuning_180 / sigma) ** 2))

    delta_x = (x_tar - x_src) * 0.07
    delta_z = (z_tar - z_src) * 0.04

    theta_pref = tar_tuning * (np.pi / 180.0)
    xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
    sigma_phase = 1.0
    phase_scale_ratio = np.exp(-(xz**2 / (2 * sigma_phase**2)))

    # To account for the 0.07 vs 0.04 dimensions. This ensures
    # the horizontal neurons are scaled by 5.5/4 (from the midpoint
    # of 4 & 7). Also, ensures the vertical is scaled by 5.5/7. This
    # was a basic linear estimate to get the numbers (y = ax + b).
    theta_tar_scale = abs(abs(abs(180.0 - abs(tar_tuning) % 360.0) - 90.0) - 90.0)
    phase_scale_ratio = phase_scale_ratio * (5.5 / 4 - 11.0 / 1680 * theta_tar_scale)

    return w_multiplier_180 * phase_scale_ratio * edge_props["syn_weight"]
