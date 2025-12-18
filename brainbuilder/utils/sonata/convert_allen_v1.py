import h5py
import pandas as pd
import numpy as np
import re
from itertools import chain
from brainbuilder import utils
from pathlib import Path
from collections import defaultdict


def sonata_to_dataframe(sonata_file, file_type="nodes"):
    with h5py.File(sonata_file, "r") as h5f:
        population_names = list(h5f[f"/{file_type}"].keys())
        assert len(population_names) == 1, "Single population is supported only"
        population_name = population_names[0]

        population = h5f[f"{file_type}/{population_name}"]
        assert "0" in population, "group '0' doesn't exst"

        data = defaultdict(list)
        for group_name in population.keys():
            # loop through groups /0, /1 ... in allen's sonata files
            if not group_name.isdigit():
                continue
            group = population[group_name]
            for key in group.keys():
                data[key].extend(group[key][()])

        cells_df = pd.DataFrame(data)
        # # Create DataFrame with NaN for missing values, but very slow
        # cells_df = pd.DataFrame.from_dict(data, orient='index').transpose()
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
            [
                "node_type_id",
                "dynamics_params",
                "morphology",
                "rotation_angle_zaxis",
                "model_type",
                "ei",
                "location",
            ]
        ],
        on="node_type_id",
        how="left",
    )
    cells_df.rename(
        columns={"dynamics_params": "model_template", "ei": "synapse_class", "location": "layer"},
        inplace=True,
    )
    # hoc template name can not be started with number, prefix with BP_ where necessary
    cells_df["model_template"] = cells_df["model_template"].str.replace(
        r"^([0-9][A-Za-z0-9_]*)(?:\.json)?$|^([A-Za-z][A-Za-z0-9_]*)(?:\.json)?$",
        lambda m: f"hoc:BP_{m.group(1)}" if m.group(1) else f"hoc:{m.group(2)}",
        regex=True,
    )
    cells_df["morphology"] = cells_df["morphology"].str.replace(r"\.[^.]+$", "", regex=True)
    cells_df["rotation_angle_zaxis"] = cells_df["rotation_angle_zaxis"].fillna(0)
    cells_df["morphology"] = cells_df["morphology"].fillna("None")
    cells_df["synapse_class"] = cells_df["synapse_class"].map({"e": "EXC", "i": "INH"})

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
                "distance_range",
                "target_sections",
            ]
        ],
        on="edge_type_id",
        how="left",
    )
    return edges_df, pop[0], pop[1]


def prepare_synapses(edges_df, nodes_df, precomputed_edges_file, syn_parameter_dir):
    edges_df = add_synapse_parameters(edges_df, syn_parameter_dir)
    add_dummy_values(edges_df, ["depression_time", "n_rrp_vesicles", "syn_type_id"], -1)
    if "weight_function" in edges_df.columns and "weight_sigma" in edges_df.columns:
        adjust_synapse_weights(edges_df, nodes_df)
    if precomputed_edges_file:
        edges_df = add_precomputed_synapse_locations(edges_df, nodes_df, precomputed_edges_file)
    edges_df.rename(columns={"syn_weight": "conductance"}, inplace=True)
    return edges_df


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
    edges_df.loc[mask_point, "syn_weight"] *= edges_df.loc[mask_point, "nsyns"]
    # cross check with precompuated file to make sure the weights are correct
    assert np.allclose(edges_df.loc[mask_point, "syn_weight"], abs(syn_point_df["syn_weight"])), (
        "point syn weight is not consistent with the precomputed file"
    )

    # For edges targeting biophysical cells, expand synapses, apply precomputed sec_id and seg_x
    repeat_counts = edges_df["nsyns"].where(edges_df["target_node_id"].isin(biophysical_gids), 1)
    edges_df_expanded = edges_df.loc[edges_df.index.repeat(repeat_counts)].reset_index(drop=True)
    mask_biophysical = edges_df_expanded["target_node_id"].isin(biophysical_gids)
    assert np.allclose(
        edges_df_expanded.loc[mask_biophysical, "syn_weight"], syn_biophysical_df["syn_weight"]
    ), "biophysical syn weight is not consistent with the precomputed file"
    edges_df_expanded["afferent_section_id"] = -1
    edges_df_expanded["afferent_section_pos"] = -1.0
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
    edges_df.loc[edges_df["weight_function"] == "DirectionRule_others", "syn_weight"] = (
        DirectionRule_others(edges_df, src_df, tgt_df)
    )
    edges_df.loc[edges_df["weight_function"] == "DirectionRule_EE", "syn_weight"] = (
        DirectionRule_EE(edges_df, src_df, tgt_df)
    )
    edges_df.loc[edges_df["weight_function"] == "gaussianLL", "syn_weight"] = gaussianLL(
        edges_df, src_df, tgt_df
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


def gaussianLL(edge_props, src_props, trg_props):
    src_tuning = src_props["tuning_angle"]
    tar_tuning = trg_props["tuning_angle"]

    mask = src_tuning.isna()
    src_tuning.loc[mask] = np.random.uniform(0.0, 360.0)
    mask = tar_tuning.isna()
    tar_tuning.loc[mask] = np.random.uniform(0.0, 360.0)

    w0 = edge_props["syn_weight"]
    sigma = edge_props["weight_sigma"]

    delta_tuning = abs(abs(abs(180.0 - abs(tar_tuning - src_tuning) % 360.0) - 90.0) - 90.0)
    weight = w0 * np.exp(-((delta_tuning / sigma) ** 2))

    return weight


def compute_synapse_locations(
    nodes_file, node_types_file, edges_file, edge_types_file, output_dir, morphology_dir
):
    nodes_df, _node_pop = load_allen_nodes(nodes_file, node_types_file)
    edges_df, src_pop, tgt_pop = load_allen_edges(edges_file, edge_types_file)
    adjust_synapse_weights(edges_df, nodes_df)
    biophysical_gids = nodes_df.index[nodes_df["model_type"] == "biophysical"]
    biophysical_edges = edges_df[(edges_df["target_node_id"].isin(biophysical_gids))]
    point_gids = nodes_df.index[nodes_df["model_type"] == "point_process"]
    point_edges = edges_df[(edges_df["target_node_id"].isin(point_gids))]
    point_edges.loc[:, "syn_weight"] *= point_edges["nsyns"]
    sec_ids, seg_xs = find_section_locations(biophysical_edges, nodes_df, morphology_dir)
    repeat_counts = biophysical_edges["nsyns"]
    biophysical_edges = biophysical_edges.loc[
        biophysical_edges.index.repeat(repeat_counts)
    ].reset_index(drop=True)
    biophysical_edges["sec_id"] = sec_ids
    biophysical_edges["sec_x"] = seg_xs

    if not Path(output_dir).exists():
        Path(output_dir).mkdir()
    output_prefix = f"{src_pop}_{tgt_pop}"
    biophysical_edges.to_csv(
        Path(output_dir) / f"{output_prefix}_biophysical_edges_with_syn_locations.csv",
        index=False,
        columns=[
            "source_node_id",
            "target_node_id",
            "edge_type_id",
            "nsyns",
            "sec_id",
            "sec_x",
            "syn_weight",
        ],
    )
    output_file = Path(output_dir) / f"{output_prefix}_syn_locations.h5"
    print(f"write output file {output_file}")
    with h5py.File(output_file, "w") as h5f:
        edge_population = f"{src_pop}__{tgt_pop}__chemical"
        group = h5f.create_group(f"/edges/{edge_population}")
        group_pop = group.create_group("0")
        group_pop.create_dataset("sec_id", data=biophysical_edges["sec_id"].to_numpy())
        group_pop.create_dataset("sec_x", data=biophysical_edges["sec_x"].to_numpy())
        group_pop.create_dataset("syn_weight", data=biophysical_edges["syn_weight"].to_numpy())
        group_pop = group.create_group("1")
        group_pop.create_dataset("syn_weight", data=point_edges["syn_weight"].to_numpy())


def find_section_locations(edges_df, nodes_df, morph_dir):
    from tqdm import tqdm

    all_sec_ids = []
    all_seg_xs = []
    for edge_row in tqdm(edges_df.itertuples(index=True, name="Edge"), total=len(edges_df)):
        gid = edge_row.target_node_id
        morpho_file = Path(morph_dir) / (nodes_df.iloc[gid]["morphology"] + ".swc")
        assert morpho_file.exists(), f"Morphology file {morpho_file} does not exist"
        # if morpho_file != check_file:
        #     continue
        distance_range = edge_row.distance_range
        nsyns = edge_row.nsyns
        target_sections = edge_row.target_sections
        if isinstance(distance_range, str):
            distance_range = distance_range.strip("[]")
            distance_range = [float(x) for x in distance_range.split(",")]
        if isinstance(target_sections, str):
            target_sections = target_sections.strip("[]")
            target_sections = [re.sub(r"[\"'\s]", "", x) for x in target_sections.split(",")]
        sec_ids, seg_xs = choose_synapse_locations(
            nsyns, distance_range, target_sections, str(morpho_file), rng_seed=gid
        )
        all_sec_ids.append(sec_ids)
        all_seg_xs.append(seg_xs)
    return np.concatenate(all_sec_ids), np.concatenate(all_seg_xs)


morphology_cache = {} # cache for the same morphology file and target range
prng_cache = {} # one rng per gid with the seed = gid, as bmtk does


def choose_synapse_locations(nsyns, distance_range, target_sections, morph_file, rng_seed=None):
    from bmtk.builder.bionet.swc_reader import SWCReader

    cache_key = (morph_file, tuple(target_sections), tuple(distance_range))
    if cache_key in morphology_cache:
        tar_seg_ix, tar_seg_prob, morph_reader = morphology_cache[cache_key]
    else:
        morph_reader = SWCReader(morph_file, rng_seed)
        morph_reader.set_segment_dl(20)
        # morph_reader.fix_axon() // NO replace axons to preserve the original indices, align with OBI
        tar_seg_ix, tar_seg_prob = morph_reader.find_sections(
            section_names=target_sections, distance_range=distance_range
        )
        morphology_cache[cache_key] = (tar_seg_ix, tar_seg_prob, morph_reader)

    # print(f"tar_seg_ix={tar_seg_ix} tar_seg_prob={tar_seg_prob}")
    if rng_seed in prng_cache:
        prng = prng_cache[rng_seed]
    else:
        prng = np.random.RandomState(rng_seed)
        prng_cache[rng_seed] = prng
  
    secs_ix = prng.choice(tar_seg_ix, nsyns, p=tar_seg_prob)
    sec_ids = morph_reader.seg_props.sec_id[secs_ix]
    seg_xs = morph_reader.seg_props.x[secs_ix]
    assert max(sec_ids) < morph_reader.n_sections, (
        f"sec_id {max(sec_ids)} exceeds number of sections {SWCReader.n_sections} in morphology {morph_file}"
    )
    # sec_ids, seg_xs = morph_reader.choose_sections(
    #          section_names=target_sections,
    #          distance_range=distance_range,
    #          n_sections=nsyns
    # )

    return sec_ids, seg_xs
