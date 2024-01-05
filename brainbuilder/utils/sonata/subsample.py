"""Subsample the circuit nodes and edges."""
import logging
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import bluepysnap
import h5py
import libsonata
import numpy as np
import pandas as pd
from voxcell import CellCollection

from brainbuilder.utils import dump_json

L = logging.getLogger(__name__)

# properties that start with it are dynamic, and handled appropriately, see `dynamics_params` in
# https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-nodes
SONATA_DYNAMIC_PROPERTY = "@dynamics:"


def is_relative_to(first, second):
    """Return True if the path is relative to another path or False."""
    # TODO: for compatibility with Python<3.9, it can be replaced with first.is_relative_to(second)
    try:
        first.relative_to(second)
        return True
    except ValueError:
        return False


def _check_output_dir(output, circuit_config):
    output = Path(output).resolve()
    if output == Path(circuit_config).parent.resolve():
        L.error("The output directory cannot contain the original circuit config")
        sys.exit(1)
    if is_relative_to(Path().resolve(), output):
        L.error("The output directory cannot be the working directory or a parent")
        sys.exit(1)
    if output.is_dir():
        L.info("Removing output directory %s", output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def _is_string_enum(series):
    """Return True if ``series`` contains enum of strings, False otherwise."""
    is_cat_str = (
        isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.categories.dtype == object
    )
    return series.dtype == object or is_cat_str


def _convert_ids(ids, ids_map):
    """Return the given ids converted using ids_map.

    Args:
        ids: array or list of ids to be converted.
        ids_map: pd.Series having the old ids as index, and the new ids as values.

    Returns:
        numpy array of converted ids.
    """
    return ids_map.loc[ids].to_numpy()


def _get_pop_sampling_count(
    pop_sampling_count, pop_sampling_ratio, sampling_count, sampling_ratio, pop_size
):
    """Return the number of ids to sample.

    The input parameters are considered in the following order, until one of them is not 0:

    - pop_sampling_count
    - pop_sampling_ratio
    - sampling_count
    - sampling_ratio

    Args:
        pop_sampling_count (int): population sampling count, or 0 to ignore.
        pop_sampling_ratio (float): population sampling ratio, or 0 to ignore.
        sampling_count (int): global sampling count, or 0 to ignore.
        sampling_ratio (float): global sampling ratio, or 0 to ignore.
        pop_size (int): population size.
    """
    if not pop_sampling_count:
        pop_sampling_count = int(pop_size * pop_sampling_ratio)
    if not pop_sampling_count:
        pop_sampling_count = sampling_count
    if not pop_sampling_count:
        pop_sampling_count = int(pop_size * sampling_ratio)
    return min(pop_sampling_count, pop_size)


def _subsample_nodes(circuit, node_populations, sampling_ratio, sampling_count):
    node_populations = node_populations or defaultdict(dict)
    for node_population_name in circuit.nodes.population_names:
        L.info("Processing node population %r", node_population_name)
        pop = circuit.nodes[node_population_name]
        pop_sampling_count = _get_pop_sampling_count(
            pop_sampling_count=node_populations[node_population_name].get("sampling_count", 0),
            pop_sampling_ratio=node_populations[node_population_name].get("sampling_ratio", 0),
            sampling_count=sampling_count,
            sampling_ratio=sampling_ratio,
            pop_size=pop.size,
        )
        node_ids = pop.ids(sample=pop_sampling_count)
        nodes_df = pop.get(node_ids)
        if len(nodes_df) > 0:
            L.info("Selected %s/%s nodes", pop_sampling_count, pop.size)
            yield node_population_name, nodes_df
        else:
            L.info("Ignored because empty")


def _subsample_edges(circuit, sampled_node_ids):
    for edge_population_name in circuit.edges.population_names:
        L.info("Processing edge population %r", edge_population_name)
        pop = circuit.edges[edge_population_name]
        if pop.source.name not in sampled_node_ids:
            L.info("Ignored b/c the sampled source node population %r is empty", pop.source.name)
            continue
        if pop.target.name not in sampled_node_ids:
            L.info("Ignored b/c the sampled target node population %r is empty", pop.target.name)
            continue
        edges_df = pop.pathway_edges(
            source=sampled_node_ids[pop.source.name].index.to_numpy(),
            target=sampled_node_ids[pop.target.name].index.to_numpy(),
            properties=sorted(pop.property_names),
        )
        edges_df["@source_node"] = _convert_ids(
            edges_df["@source_node"], ids_map=sampled_node_ids[pop.source.name]
        )
        edges_df["@target_node"] = _convert_ids(
            edges_df["@target_node"], ids_map=sampled_node_ids[pop.target.name]
        )
        if len(edges_df) > 0:
            L.info("Selected %s/%s edges", len(edges_df), pop.size)
            yield edge_population_name, edges_df
        else:
            L.info("Ignored because empty")


def _save_node_population(
    nodes_file, nodes_df, node_population_name, forced_library=None, mode="w"
):
    nodes_file.parent.mkdir(exist_ok=True, parents=True)
    # reset the node ids
    nodes_df = nodes_df.copy()
    nodes_df.index = 1 + np.arange(len(nodes_df))
    cc = CellCollection.from_dataframe(nodes_df)
    cc.population_name = node_population_name
    cc.save_sonata(filename=str(nodes_file), forced_library=forced_library, mode=mode)


def _save_edge_population(
    edges_file,
    edges_df,
    edge_population_name,
    source_node_population_name,
    target_node_population_name,
    forced_library=None,
    mode="w",
):
    """Save an edges DataFrame to file.

    Based on https://github.com/BlueBrain/voxcell/blob/7aae6cf5/voxcell/cell_collection.py#L349
    but modified to save edges instead of nodes.
    """
    # pylint: disable=too-many-locals
    forced_library = set() if forced_library is None else set(forced_library)
    str_dt = h5py.special_dtype(vlen=str)
    edges_file.parent.mkdir(exist_ok=True, parents=True)
    ignored_properties = {"@source_node", "@target_node"}
    with h5py.File(str(edges_file), mode=mode) as h5f:
        population = h5f.create_group(f"/edges/{edge_population_name}")
        source_dataset = population.create_dataset("source_node_id", data=edges_df["@source_node"])
        source_dataset.attrs["node_population"] = source_node_population_name
        target_dataset = population.create_dataset("target_node_id", data=edges_df["@target_node"])
        target_dataset.attrs["node_population"] = target_node_population_name
        group = population.create_group("0")
        for name, series in edges_df.items():
            if name in ignored_properties:
                continue
            if name.startswith(SONATA_DYNAMIC_PROPERTY):
                name = name.split(SONATA_DYNAMIC_PROPERTY)[1]
                dt = str_dt if series.dtype == object else series.dtype
                group.create_dataset(
                    f"dynamics_params/{name}",
                    data=series.to_numpy(),
                    dtype=dt,
                )
            elif _is_string_enum(series) or (series.dtype == object and name in forced_library):
                indices, unique_values = series.factorize()
                if name in forced_library or len(unique_values) < 0.5 * len(indices):
                    group.create_dataset(name, data=indices.astype(np.uint32))
                    group.create_dataset(f"@library/{name}", data=unique_values, dtype=str_dt)
                else:
                    group.create_dataset(name, data=series.to_numpy(), dtype=str_dt)
            else:
                group.create_dataset(name, data=series.to_numpy())
        edge_count = len(edges_df)
        population["edge_type_id"] = np.full(edge_count, fill_value=-1)
        population["edge_group_id"] = np.full(edge_count, fill_value=0)
        population["edge_group_index"] = np.arange(edge_count, dtype=np.uint64)

    libsonata.EdgePopulation.write_indices(
        str(edges_file),
        edge_population_name,
        # add 1 because IDs are 0-based
        source_node_count=int(np.max(edges_df["@source_node"]) + 1),
        target_node_count=int(np.max(edges_df["@target_node"]) + 1),
    )


def _write_circuit_config(output_file, networks):
    """Write a simple circuit-config.json for all the node/edge populations created."""

    def _make_relative(path):
        relpath = path.relative_to(output_file.parent)
        return f"$BASE_DIR/{relpath}"

    networks = deepcopy(networks)
    for data in networks["nodes"]:
        data["nodes_file"] = _make_relative(data["nodes_file"])
    for data in networks["edges"]:
        data["edges_file"] = _make_relative(data["edges_file"])
    config_dict = {
        "version": "2",
        "manifest": {"$BASE_DIR": "."},
        "networks": networks,
    }
    dump_json(output_file, config_dict)
    L.debug("Written circuit config %s", output_file)


def _write_mapping(output_file, id_mapping):
    """Write the id mappings between the old and new populations for future analysis."""
    mapping = {}
    for population, series in id_mapping.items():
        mapping[population] = {
            "old_id": series.index.to_list(),
            "new_id": series.to_list(),
        }
    dump_json(output_file, mapping)
    L.debug("Written id mapping %s", output_file)


def subsample_circuit(
    output, circuit_config, sampling_ratio, sampling_count=None, node_populations=None, seed=0
):
    """Apply subsampling to the given circuit.

    Args:
        output (str|Path): path to the output directory.
        circuit_config (str|Path): path to the input circuit config file.
        sampling_ratio (float): sampling ratio for nodes (from 0.0 to 1.0).
        sampling_count (int|None): number of nodes (if specified, sampling_ratio is ignored)
        node_populations (dict|None): optional dict of node populations. Each dict can specify the
            desired sampling_ratio or sampling_count for that population. If the dict is empty,
            then the global values are used.
        seed (int): RNG seed.
    """
    np.random.seed(seed)
    output = _check_output_dir(output, circuit_config)
    # map node_population_name -> pd.Series with index=sampled_node_ids and data=remapped_node_ids
    sampled_node_ids = {}
    networks = {"nodes": [], "edges": []}
    circuit = bluepysnap.Circuit(str(circuit_config))
    for node_population_name, nodes_df in _subsample_nodes(
        circuit, node_populations, sampling_ratio, sampling_count
    ):
        nodes_file = output / "networks" / "nodes" / node_population_name / "nodes.h5"
        _save_node_population(nodes_file, nodes_df, node_population_name)
        networks["nodes"].append(
            {"nodes_file": nodes_file, "populations": {node_population_name: {}}}
        )
        sampled_node_ids[node_population_name] = pd.Series(
            np.arange(len(nodes_df)), index=nodes_df.index
        )

    for edge_population_name, edges_df in _subsample_edges(circuit, sampled_node_ids):
        edges_file = output / "networks" / "edges" / edge_population_name / "edges.h5"
        _save_edge_population(
            edges_file,
            edges_df,
            edge_population_name,
            source_node_population_name=circuit.edges[edge_population_name].source.name,
            target_node_population_name=circuit.edges[edge_population_name].target.name,
        )
        networks["edges"].append(
            {"edges_file": edges_file, "populations": {edge_population_name: {}}}
        )

    _write_circuit_config(output / "circuit_config.json", networks)
    _write_mapping(output / "id_mapping.json", sampled_node_ids)
