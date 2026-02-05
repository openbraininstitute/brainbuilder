# SPDX-License-Identifier: Apache-2.0
"""Split a SONATA node/edge population into sub-populations"""

import collections
import copy
import itertools as it
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import bluepysnap
import h5py
import libsonata
import numpy as np
import pandas as pd
import voxcell
from joblib import Parallel, delayed

from brainbuilder import utils
from brainbuilder.utils import hdf5

L = logging.getLogger(__name__)

# So as not to exhaust memory, the edges files are loaded/written in chunks of this size
H5_READ_CHUNKSIZE = 500_000_000
# Name of the unique expected group in sonata nodes and edges files
GROUP_NAME = "0"
# Sentinel to mark an edge file being empty
DELETED_EMPTY_EDGES_FILE = "DELETED_EMPTY_EDGES_FILE"
# Sentinel to mark an edge population being empty
DELETED_EMPTY_EDGES_POPULATION = "DELETED_EMPTY_EDGES_POPULATION"

# name of field with ids that are valid in extracted circuit
NEW_IDS = "new_id"
# name of field with ids that are valid in parent circuit
PARENT_IDS = "parent_id"
# name of field with ids that are valid in original circuit
ORIG_IDS = "original_id"
# name of field with node population name in parent circuit
PARENT_NAME = "parent_name"
# name of field with node population name in original circuit
ORIG_NAME = "original_name"


@dataclass
class WriteEdgeConfig:
    src_node_name: str
    dst_node_name: str
    src_edge_name: str
    dst_edge_name: str
    src_mapping: pd.DataFrame
    dst_mapping: pd.DataFrame
    h5_read_chunk_size: int | None = None
    input_path: str | Path | None = None
    output_path: str | Path | None = None
    edge_type: type[bytes] | None = None

    def __post_init__(self):
        self.input_path = (
            Path(self.input_path) if isinstance(self.input_path, str) else self.input_path
        )
        self.output_path = (
            Path(self.output_path) if isinstance(self.output_path, str) else self.output_path
        )


def _create_chunked_slices(length, chunk_size):
    """return `slices` each of size `chunk_size`, that cover `length`"""
    return (slice(start, start + chunk_size) for start in range(0, length, chunk_size))


def _isin_worker(elements, test_elements, sl, invert):
    """worker for parallelized version of nump.isin"""
    return np.isin(elements[sl], test_elements, invert=invert)


def _isin(elements, test_elements, invert=False):
    """parallelized version of nump.isin"""
    h5_chunksize = _h5_get_read_chunk_size()

    if len(elements) < h5_chunksize:
        return np.isin(elements, test_elements, invert=invert)

    parallel = Parallel(
        backend="loky",
        n_jobs=-2,
        # verbose=51,
    )

    # arbitrary chunk_size; 1e6 with the default H5_READ_CHUNKSIZE seems about right
    chunk_size = max(500, int(h5_chunksize / 500))
    ret = parallel(
        delayed(_isin_worker)(elements, test_elements, sl, invert)
        for sl in _create_chunked_slices(len(elements), chunk_size)
    )

    ret = np.concatenate(ret)

    return ret


def _get_population_name(src, dst, synapse_type="chemical"):
    """Return the population name based off `src` and `dst` node population names."""
    return src if src == dst else f"{src}__{dst}__{synapse_type}"


def _get_edge_file_name(new_pop_name):
    """Return the name of the edge file split by population."""
    return f"edges_{new_pop_name}.h5"


def _get_node_file_name(new_pop_name):
    """Return the name of the node file split by population."""
    return f"nodes_{new_pop_name}.h5"


def _get_unique_population(parent):
    """Return the h5 unique population, raise an exception if not unique."""
    population_names = list(parent)
    if len(population_names) != 1:
        raise ValueError(f"Single population is supported only, found {population_names}")
    return population_names[0]


def _get_unique_group(parent):
    """Return the h5 group 0, raise an exception if non present."""
    if GROUP_NAME not in parent:
        raise ValueError(f"Single group {GROUP_NAME!r} is required")
    return parent[GROUP_NAME]


def _load_sonata_nodes(nodes_path):
    """Load nodes from a sonata file and return it as dataframe (0-based IDs).

    Note: the returned dataframe contains the orientation matrices, but it does not contain
    the information about the original orientation format (quaternions or eulers).
    """
    df = voxcell.CellCollection.load_sonata(nodes_path).as_dataframe()
    # CellCollection returns 1-based IDs but we need 0-based IDs
    df.index -= 1
    return df


def _save_sonata_nodes(nodes_path, df, population_name):
    """Save a dataframe of nodes (0-based IDs) to sonata file.

    Note: using voxcell >= 2.7.1 to load the dataframe and save the result to sonata,
    CellCollection will save the orientation using the default format (quaternions).
    """
    # CellCollection expects 1-based IDs
    df.index += 1
    cell_collection = voxcell.CellCollection.from_dataframe(df)
    cell_collection.population_name = population_name
    cell_collection.save_sonata(str(nodes_path), mode="a")
    # restore the original index
    df.index -= 1
    return nodes_path


def _init_edge_group(orig_group: h5py.Group, new_group: h5py.Group, additional_attrs):
    """Initialize an edge group by recreating appendable datasets and attrs.

    Copies the dataset layout from orig_group into new_group using appendable
    datasets. Optionally sets extra dataset attributes from additional_attrs.
    Recreates the "dynamics_params" subgroup with matching datasets.

    Args:
        orig_group (h5py.Group): Source edge group (e.g. /edges/default/0).
        new_group (h5py.Group): Destination edge group to initialize.
        additional_attrs (dict[str, dict[str, Any]]): Optional mapping
            dataset_name -> {attr_name: value} for attributes to set on
            created datasets.
    """
    for name, attr in orig_group.items():
        if isinstance(attr, h5py.Dataset):
            hdf5.create_appendable_dataset(new_group, name, attr.dtype)
            if name in additional_attrs:
                for attr_name, val in additional_attrs[name].items():
                    new_group[name].attrs[attr_name] = val
        elif isinstance(attr, h5py.Group) and name == "dynamics_params":
            new_group.create_group(name)
            for k, values in attr.items():
                assert isinstance(values, h5py.Dataset), f"dynamics_params has an h5 subgroup: {k}"
                hdf5.create_appendable_dataset(new_group[name], k, values.dtype)
        else:
            raise ValueError('Only "dynamics_params" group is expected')


def _populate_edge_group(orig_group, new_group, sl, mask, overrides):
    """Append filtered data from orig_group datasets into new_group.

    Copies data chunk-by-chunk using the provided slice and boolean mask.
    Dataset values can be replaced via overrides instead of being read from
    the source. Also handles the "dynamics_params" subgroup recursively.

    Args:
        orig_group (h5py.Group): Source edge group to read from.
        new_group (h5py.Group): Destination edge group to append into.
        sl (slice): Slice selecting the chunk range in each dataset.
        mask (np.ndarray): Boolean mask applied to the sliced data.
        overrides (dict[str, np.ndarray]): Optional mapping of dataset
            name to precomputed values to append instead of ds[sl][mask].
    """
    for name, ds in orig_group.items():
        if isinstance(ds, h5py.Dataset):
            if name in overrides:
                hdf5.append_to_dataset(new_group[name], overrides[name])
            else:
                hdf5.append_to_dataset(new_group[name], ds[sl][mask])
        elif isinstance(ds, h5py.Group) and name == "dynamics_params":
            for k, values in ds.items():
                if isinstance(values, h5py.Dataset):
                    hdf5.append_to_dataset(new_group[name][k], values[sl][mask])
        elif isinstance(ds, h5py.Group) and name == "@library":
            raise NotImplementedError(
                "@library group is defined by the spec but not currently supported"
            )
        else:
            raise ValueError('Only "dynamics_params" group is expected')


def _finalize_edges(new_edges):
    """add datasets for `new_edges` so they fulfil SONATA spec"""
    edge_count = len(new_edges["source_node_id"])
    new_edges["edge_type_id"] = np.full(edge_count, -1)
    new_edges["edge_group_id"] = np.full(edge_count, 0)
    new_edges["edge_group_index"] = np.arange(edge_count, dtype=np.uint64)


def _h5_get_read_chunk_size():
    """get the desired H5 read size, either from default of from env var"""
    return int(os.environ.get("H5_READ_CHUNKSIZE", H5_READ_CHUNKSIZE))


def _copy_filtered_edges(
    h5in: h5py.File,
    h5out: h5py.File,
    write_edge_config: WriteEdgeConfig,
    edge_mappings: dict[str, tuple[pd.DataFrame, str]] = None,
):
    """
    Copy and filter edge datasets from an input HDF5 file to an output HDF5 file.

    This function:
        - Reads the source edge population in chunks.
        - Filters edges based on source and target node mappings.
        - Copies source/target node IDs and all associated datasets.
        - Initializes and populates new edge groups, preserving attributes.
        - Raises errors if invalid IDs or unsupported groups (e.g., @library) are encountered.

    Args:
        h5in (h5py.File): Input HDF5 file containing original edges.
        h5out (h5py.File): Output HDF5 file to store filtered edges.
        write_edge_config (WriteEdgeConfig): Configuration specifying
            source/target populations, edge names, mappings, and read chunk size.

    Notes:
        - Only the "dynamics_params" group is currently supported in edge groups.
        - Supports appendable datasets; existing datasets will raise an error if re-created.
        - Processing is chunked to handle large edge populations efficiently.
    """
    # extract values
    h5_read_chunk_size = (
        write_edge_config.h5_read_chunk_size
        if write_edge_config.h5_read_chunk_size is not None
        else _h5_get_read_chunk_size()
    )
    src_node_name = write_edge_config.src_node_name
    dst_node_name = write_edge_config.dst_node_name
    src_edge_name = write_edge_config.src_edge_name
    dst_edge_name = write_edge_config.dst_edge_name
    src_mapping = write_edge_config.src_mapping
    dst_mapping = write_edge_config.dst_mapping
    is_neuroglial = write_edge_config.edge_type == "synapse_astrocyte"

    # get groups
    orig_edges = h5in["edges"][src_edge_name]
    orig_group = _get_unique_group(orig_edges)
    new_edges = h5out.create_group("edges/" + dst_edge_name)
    new_group = new_edges.create_group(GROUP_NAME)

    hdf5.create_appendable_dataset(new_edges, "source_node_id", np.uint64)
    hdf5.create_appendable_dataset(new_edges, "target_node_id", np.uint64)

    # since create_appendable_dataset already fails if we append to an existing
    # dataset, the following attribute assignments are safe
    new_edges["source_node_id"].attrs["node_population"] = src_node_name
    new_edges["target_node_id"].attrs["node_population"] = dst_node_name

    additional_attrs = {}
    if is_neuroglial:
        # find new name of synapse_edge_pop
        src_syn_edge_pop = orig_edges[GROUP_NAME]["synapse_id"].attrs["edge_population"]
        _, dst_syn_edge_pop = edge_mappings[src_syn_edge_pop]
        # create dataset and add correct attr
        additional_attrs["synapse_id"] = {"edge_population": dst_syn_edge_pop}

    _init_edge_group(orig_group, new_group, additional_attrs)

    sgids_new = src_mapping.index.to_numpy()
    tgids_new = dst_mapping.index.to_numpy()
    assert (sgids_new >= 0).all(), "Source population ids must be positive."
    assert (tgids_new >= 0).all(), "Target population ids must be positive."

    total_chunks = math.ceil(len(orig_edges["source_node_id"]) / h5_read_chunk_size)
    L.debug(
        "Processing %s edges in %s chunks of size %s [src_edge_name=%s]",
        len(orig_edges["source_node_id"]),
        total_chunks,
        h5_read_chunk_size,
        src_edge_name,
    )

    sl_and_masks = _compute_chunks_and_masks(
        orig_edges=orig_edges,
        sgids_new=sgids_new,
        tgids_new=tgids_new,
        h5_read_chunk_size=h5_read_chunk_size,
        edge_mappings=edge_mappings,
        is_neuroglial=is_neuroglial,
    )

    if edge_mappings is not None:
        offset = new_edges["source_node_id"].shape[0]
        if offset != 0:
            raise RuntimeError(
                f"It may be unsafe to append edges with a nonzero offset ({offset}). "
                "Edge IDs may become inconsistent."
            )
        edge_mappings[write_edge_config.src_edge_name] = (
            _compute_edge_mapping(sl_and_masks=sl_and_masks, offset=offset),
            write_edge_config.dst_edge_name,
        )

    _write_masked_edges(
        sl_and_masks=sl_and_masks,
        new_edges=new_edges,
        orig_edges=orig_edges,
        src_mapping=src_mapping,
        dst_mapping=dst_mapping,
        edge_mappings=edge_mappings,
        is_neuroglial=is_neuroglial,
    )

    L.debug("Finalize edges")
    _finalize_edges(new_edges)


def _compute_edge_mapping(sl_and_masks, offset=0):
    """
    Build a pandas DataFrame mapping absolute indices to NEW_IDS.

    Parameters
    ----------
    sl_and_masks : list of (slice, relative_indices)
        Output from `_compute_chunks_and_masks`.
    offset : int
        Starting value for NEW_IDS. Useful for actual appends

    Returns
    -------
    pd.DataFrame
        Index = absolute indices in the original dataset
        Column = NEW_IDS (sequential IDs starting from offset)
    """
    if not sl_and_masks:
        return pd.DataFrame(columns=[NEW_IDS], dtype=np.int64)

    # compute absolute indices
    chunk_indices = [sl.start + rel_idxs for sl, rel_idxs in sl_and_masks]
    flat_idxs = np.hstack(chunk_indices).astype(np.int64)

    # build DataFrame
    edge_mapping = pd.DataFrame(
        {NEW_IDS: np.arange(len(flat_idxs), dtype=np.int64) + offset}, index=flat_idxs
    )
    return edge_mapping


def _compute_chunks_and_masks(
    orig_edges, sgids_new, tgids_new, h5_read_chunk_size, edge_mappings, is_neuroglial=False
):
    """Compute relative indices of edges to keep for each chunk."""
    sl_and_masks = []
    for sl in _create_chunked_slices(len(orig_edges["source_node_id"]), h5_read_chunk_size):
        sgids = orig_edges["source_node_id"][sl]
        tgids = orig_edges["target_node_id"][sl]
        sgid_mask = _isin(sgids, sgids_new)
        tgid_mask = _isin(tgids, tgids_new)

        mask = sgid_mask & tgid_mask

        if is_neuroglial:
            src_syn_edge_pop = orig_edges[GROUP_NAME]["synapse_id"].attrs["edge_population"]
            # TODO remove dst_syn_edge_pop
            edge_mask, dst_syn_edge_pop = edge_mappings[src_syn_edge_pop]
            syn_ids = orig_edges[GROUP_NAME]["synapse_id"][sl]
            syn_mask = _isin(syn_ids, edge_mask.index.to_numpy())
            mask &= syn_mask

        rel_idxs = np.flatnonzero(mask)
        if rel_idxs.size > 0:
            sl_and_masks.append((sl, rel_idxs.astype(np.int64)))

    return sl_and_masks


def _write_masked_edges(
    sl_and_masks, new_edges, orig_edges, src_mapping, dst_mapping, edge_mappings, is_neuroglial
):
    """Apply masks per chunk to write edges to HDF5 and populate edge groups."""
    for sl, mask in sl_and_masks:
        L.debug("Processing chunk %s", sl)
        sgids = orig_edges["source_node_id"][sl]
        tgids = orig_edges["target_node_id"][sl]
        hdf5.append_to_dataset(
            new_edges["source_node_id"], src_mapping.loc[sgids[mask]][NEW_IDS].to_numpy()
        )
        hdf5.append_to_dataset(
            new_edges["target_node_id"], dst_mapping.loc[tgids[mask]][NEW_IDS].to_numpy()
        )
        overrides = {}
        if is_neuroglial:
            src_syn_edge_pop = orig_edges[GROUP_NAME]["synapse_id"].attrs["edge_population"]
            edge_mapping, _ = edge_mappings[src_syn_edge_pop]
            syn_ids = orig_edges[GROUP_NAME]["synapse_id"][sl]
            overrides["synapse_id"] = edge_mapping.loc[syn_ids[mask]][NEW_IDS].to_numpy()

        _populate_edge_group(orig_edges[GROUP_NAME], new_edges[GROUP_NAME], sl, mask, overrides)


def _get_node_counts(
    h5out: h5py.File, new_edge_pop_name: str, src_mapping: pd.DataFrame, dst_mapping: pd.DataFrame
):
    """for `h5out`, return the `new_edge_pop_name`, `source_node_count`, and `target_node_count`"""

    source_node_count = int(np.max(src_mapping)) + 1
    target_node_count = int(np.max(dst_mapping)) + 1

    new_edges = h5out["edges"][new_edge_pop_name]
    edge_count = len(new_edges["source_node_id"])

    if edge_count > 0:
        assert source_node_count >= int(np.max(new_edges["source_node_id"]))
        assert target_node_count >= int(np.max(new_edges["target_node_id"]))

    return edge_count, source_node_count, target_node_count


def _write_indexes(
    edge_file_name: str | Path, new_pop_name: str, source_node_count: int, target_node_count: int
):
    """ibid"""
    libsonata.EdgePopulation.write_indices(
        str(edge_file_name), new_pop_name, source_node_count, target_node_count
    )


def _check_all_edges_used(h5in, written_edges):
    """Verify that the number of written edges matches the number of initial edges."""
    orig_edges = h5in["edges"][_get_unique_population(h5in["edges"])]
    expected_edges = len(orig_edges["source_node_id"])
    if expected_edges != written_edges:
        raise RuntimeError(
            f"Written edges mismatch: expected={expected_edges}, actual={written_edges}"
        )


def _write_edges(
    output,
    edges_path,
    id_mapping,
    h5_read_chunk_size=None,
    expect_to_use_all_edges=True,
):
    """create all new edge populations in separate files"""
    with h5py.File(edges_path, "r") as h5in:
        written_edges = 0
        for src_node_pop, dst_node_pop in it.product(id_mapping, id_mapping):
            edge_pop_name = _get_population_name(src_node_pop, dst_node_pop)
            edge_file_name = Path(output) / _get_edge_file_name(edge_pop_name)

            L.debug("Writing to  %s", edge_file_name)
            with h5py.File(edge_file_name, "w") as h5out:
                write_edge_config = WriteEdgeConfig(
                    src_node_name=src_node_pop,
                    dst_node_name=dst_node_pop,
                    src_edge_name=_get_unique_population(h5in["edges"]),
                    dst_edge_name=edge_pop_name,
                    src_mapping=id_mapping[src_node_pop],
                    dst_mapping=id_mapping[dst_node_pop],
                    h5_read_chunk_size=h5_read_chunk_size,
                )

                _copy_filtered_edges(h5in=h5in, h5out=h5out, write_edge_config=write_edge_config)
                edge_count, sgid_count, tgid_count = _get_node_counts(
                    h5out, edge_pop_name, id_mapping[src_node_pop], id_mapping[dst_node_pop]
                )

            # after the h5 file is closed, it's indexed if valid, or it's removed if empty
            if edge_count > 0:
                _write_indexes(edge_file_name, edge_pop_name, sgid_count, tgid_count)
                L.debug("Wrote %s edges to %s", edge_count, edge_file_name)
                written_edges += edge_count
            else:
                Path(edge_file_name).unlink(missing_ok=True)

        if expect_to_use_all_edges:
            _check_all_edges_used(h5in, written_edges)


def _write_nodes(output, split_nodes, population_to_path=None):
    """create all new node populations in separate files

    Args:
        output(str): base directory to write node files
        split_nodes(dict): new_population_name -> df
        population_to_path(dict): population_name -> output path
    """
    if population_to_path is None:
        population_to_path = {}

    ret = {}
    for new_population, df in split_nodes.items():
        df = df.reset_index(drop=True)
        nodes_path = population_to_path.get(new_population, _get_node_file_name(new_population))
        nodes_path = Path(output) / nodes_path
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        ret[new_population] = _save_sonata_nodes(nodes_path, df, population_name=new_population)
        L.debug("Wrote %s nodes to %s", len(df), nodes_path)

    return ret


def _get_node_id_mapping(split_nodes):
    """return a dict split_nodes.keys() -> DataFrame with index old_ids, and colunm new_id"""
    return {
        new_population: pd.DataFrame({NEW_IDS: np.arange(len(df), dtype=np.int64)}, index=df.index)
        for new_population, df in split_nodes.items()
    }


def _split_population_by_attribute(nodes_path, attribute):
    """return a dictionary keyed on attribute values with each of the new populations

    Each of the unique attribute values becomes a new_population post split

    Args:
        nodes_path: path to SONATA nodes file
        attribute(str): attribute to split on

    Returns:
        dict: new_population -> df containing attributes for that new population
    """
    nodes = _load_sonata_nodes(nodes_path)
    L.debug("Splitting population on %s -> %s", attribute, nodes[attribute].unique())
    split_nodes = dict(tuple(nodes.groupby(attribute)))
    return split_nodes


def _write_circuit_config(output, split_nodes):
    """Write a simple circuit-config.json for all the node/edge populations created"""
    tmpl = {
        "manifest": {
            "$BASE_DIR": ".",
        },
        "networks": {
            "nodes": [],
            "edges": [],
        },
    }

    for src, dst in it.product(split_nodes, split_nodes):
        new_pop_name = _get_population_name(src, dst)
        if src == dst:
            tmpl["networks"]["nodes"].append(
                {
                    "nodes_file": str(Path("$BASE_DIR") / _get_node_file_name(new_pop_name)),
                    "node_types_file": None,
                }
            )

        edge_path = Path(output) / _get_edge_file_name(new_pop_name)
        if edge_path.exists():
            tmpl["networks"]["edges"].append(
                {
                    "edges_file": str(Path("$BASE_DIR") / _get_edge_file_name(new_pop_name)),
                    "edge_types_file": None,
                }
            )

    filepath = Path(output) / "circuit_config.json"
    utils.dump_json(filepath, tmpl)
    L.debug("Written circuit config %s", filepath)


def split_population(output, attribute, nodes_path, edges_path):
    """split a single node SONATA dataset into separate populations based on attribute

    Creates a new nodeset, and the corresponding edges between nodesets for each
    value of the attribute.  For instance, if the attribute chosen is 'region', a nodeset
    will be created for all regions

    The edge file is also split, as required

    Args:
        output(str): path where files will be written
        attribute(str): attribute on which to break up into sub-populations
        nodes_path(str): path to nodes sonata file
        edges_path(str): path to edges sonata file

    """
    split_populations = _split_population_by_attribute(nodes_path, attribute)
    _write_nodes(output, split_populations)

    id_mapping = _get_node_id_mapping(split_populations)
    _write_edges(output, edges_path, id_mapping, expect_to_use_all_edges=True)

    _write_circuit_config(output, split_populations)


def _split_population_by_node_set(nodes_path, node_set_name, node_set_path):
    node_storage = libsonata.NodeStorage(nodes_path)
    node_population = node_storage.open_population(next(iter(node_storage.population_names)))

    node_sets = libsonata.NodeSets.from_file(node_set_path)
    ids = node_sets.materialize(node_set_name, node_population).flatten()

    split_nodes = {node_set_name: _load_sonata_nodes(nodes_path).loc[ids]}
    return split_nodes


def simple_split_subcircuit(output, node_set_name, node_set_path, nodes_path, edges_path):
    """Split a single subcircuit out of a set of nodes and edges, based on nodeset

    Args:
        output(str): path where files will be written
        node_set_name(str): name of nodeset to extract
        node_set_path(str): path to node_sets.json file
        nodes_path(str): path to nodes sonata file
        edges_path(str): path to edges sonata file
    """
    split_populations = _split_population_by_node_set(nodes_path, node_set_name, node_set_path)

    _write_nodes(output, split_populations)

    id_mapping = _get_node_id_mapping(split_populations)
    _write_edges(output, edges_path, id_mapping, expect_to_use_all_edges=False)


def _write_subcircuit_edges(
    write_edge_config: WriteEdgeConfig, edge_mappings: dict[str, tuple[pd.DataFrame, str]]
):
    """copy a population to an edge file

    If DELETED_EMPTY_EDGES_FILE is returned, the file was removed since no
    populations existed in it any more
    If DELETED_EMPTY_EDGES_POPULATION is returned, the population was removed
    """
    output_path = write_edge_config.output_path

    L.debug(
        "Writing edges %s for %s -> %s [%s]",
        write_edge_config.src_edge_name,
        write_edge_config.src_edge_name,
        write_edge_config.dst_edge_name,
        str(output_path),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(write_edge_config.input_path, "r") as h5in:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        is_file_empty = False

        with h5py.File(output_path, "a") as h5out:
            _copy_filtered_edges(
                h5in=h5in,
                h5out=h5out,
                write_edge_config=write_edge_config,
                edge_mappings=edge_mappings,
            )

            edge_count, sgid_count, tgid_count = _get_node_counts(
                h5out=h5out,
                new_edge_pop_name=write_edge_config.dst_edge_name,
                src_mapping=write_edge_config.src_mapping,
                dst_mapping=write_edge_config.dst_mapping,
            )

            if edge_count == 0:
                del h5out[f"/edges/{write_edge_config.dst_edge_name}"]
                is_file_empty = len(h5out["/edges"]) == 0

        # after the h5 file is closed, it's indexed if valid, or it's removed if empty
        if edge_count > 0:
            _write_indexes(
                edge_file_name=write_edge_config.output_path,
                new_pop_name=write_edge_config.dst_edge_name,
                source_node_count=sgid_count,
                target_node_count=tgid_count,
            )
            L.debug("Wrote %s edges to %s", edge_count, output_path)
        elif is_file_empty:
            Path(output_path).unlink(missing_ok=True)
            output_path = DELETED_EMPTY_EDGES_FILE
        else:  # population empty, but not file
            output_path = DELETED_EMPTY_EDGES_POPULATION

        return output_path


def _gather_layout_from_networks(networks):
    """find the layout of the nodes and edges files, return a dict of the name -> relative path"""

    # Note: we are 'prioritizing' the layout of the config over the layout of the files on disk:
    # 1) the `nodes`/`edges` network keys will still have the same number of elements
    #    after writing the new config (unless populations aren't used)
    # 2) The layout of the files may be slightly different; if the config has a single population
    #    in the dict, the output population will be writen to $population_name/$original_filename.h5
    #    if it has multiple elements, it will be written to
    #    $original_parent_dir/$original_filename.h5
    #
    # See tests for more clarity
    node_populations_to_paths, edge_populations_to_paths = {}, {}

    def _extract_population_paths(key):
        """extract populations from `network_base`; return dictionary with their file path"""
        key_name = f"{key}_file"
        ret = {}
        for stanza in networks[key]:
            filename = Path(stanza[key_name]).name
            if len(stanza["populations"]) == 1:
                population = next(iter(stanza["populations"]))
                ret[population] = str(Path(population) / filename)
            else:
                # multiple populations; need to group them into the same file
                base_path = Path(stanza[key_name]).parent.name
                for population in stanza["populations"]:
                    ret[population] = str(Path(base_path) / filename)
        return ret

    node_populations_to_paths = _extract_population_paths("nodes")
    edge_populations_to_paths = _extract_population_paths("edges")

    return node_populations_to_paths, edge_populations_to_paths


def _get_storage_path(edge):
    """Return the storage path."""
    return edge.h5_filepath


def _write_subcircuit_biological(
    output,
    circuit,
    node_pop_to_paths,
    edge_pop_to_paths,
    split_populations,
    id_mapping,
):
    """write node and edge population that belong in a subcircuit

    Args:
        output: path to output
        circuit: bluepysnap circuit
        node_pop_to_paths(dict): node name -> new relative path
        edge_pop_to_paths(dict): node name -> new relative path
        split_populations(dict): population -> node dataframe
        id_mapping(dict): population name -> df with index old_ids, and colunm new_id

    returns `new_node_files`, `new_edges_files`: the paths to node & edges files that were created
    """
    new_node_files = _write_nodes(output, split_populations, node_pop_to_paths)

    write_edge_configs = []
    for edge_pop_name, edge in circuit.edges.items():
        if edge.source.name in id_mapping and edge.target.name in id_mapping:
            write_edge_config = WriteEdgeConfig(
                output_path=output / edge_pop_to_paths[edge_pop_name],
                input_path=_get_storage_path(edge),
                src_node_name=edge.source.name,
                dst_node_name=edge.target.name,
                src_edge_name=edge_pop_name,
                dst_edge_name=edge_pop_name,
                src_mapping=id_mapping[edge.source.name],
                dst_mapping=id_mapping[edge.target.name],
                edge_type=edge.type,
            )
            write_edge_configs.append(write_edge_config)

    new_edges_files = _orchestrate_write_subcircuit_edges(write_edge_configs=write_edge_configs)

    return new_node_files, new_edges_files


def _orchestrate_write_subcircuit_edges(write_edge_configs: list[WriteEdgeConfig]):
    assert all(config.edge_type is not None for config in write_edge_configs)
    new_edges_files = {}
    edge_mappings = {}
    # "synapse_astrocyte" must be processed after the other edges
    write_edge_configs_sorted = sorted(
        write_edge_configs, key=lambda cfg: cfg.edge_type == "synapse_astrocyte"
    )

    for write_edge_config in write_edge_configs_sorted:
        new_edges_files[write_edge_config.dst_edge_name] = _write_subcircuit_edges(
            write_edge_config=write_edge_config, edge_mappings=edge_mappings
        )

    return new_edges_files


def _get_subcircuit_external_ids(all_sgids, all_tgids, wanted_src_ids, wanted_dst_ids):
    """get the `external` ids

    return `id_mapping` style DataFrame for connections between `all_sgids` and
    `all_tgids` where sgids are in wanted_src_ids and tgids are in `wanted_dst_ids`

    These are the 'external' ids that become 'virtual' in the extracted subcircuit
    """
    h5_read_chunk_size = _h5_get_read_chunk_size()
    ret = None
    for sl in _create_chunked_slices(len(all_sgids), h5_read_chunk_size):
        sgids = all_sgids[sl]
        tgids = all_tgids[sl]

        mask = _isin(sgids, wanted_src_ids) & _isin(tgids, wanted_dst_ids)

        if mask.any():
            needed = np.unique(sgids[mask])
            if ret is None:
                ret = pd.DataFrame({NEW_IDS: np.arange(len(needed), dtype=np.uint)}, index=needed)
            else:
                mm = _isin(needed, ret.index.to_numpy(), invert=True)
                if mm.any():
                    needed = needed[mm]
                    start_id = int(ret[NEW_IDS].max()) + 1
                    new = pd.DataFrame(
                        {NEW_IDS: start_id + np.arange(len(needed), dtype=np.uint)}, index=needed
                    )
                    ret = pd.concat((ret, new))

    if ret is None:
        ret = pd.DataFrame({NEW_IDS: np.array([], dtype=np.uint)}, index=[])

    return ret.sort_index()


def _write_subcircuit_external(
    output,
    circuit,
    id_mapping,
    node_pop_name_mapping,
    existing_node_pop_names,
    existing_edge_pop_names,
):
    """Write external connectivity.

    returns: (new_node_files, new_edges_files); with, respectively,
    dictionaries with node and edge population_name -> path

    Warning: this writes `id_mapping` in place
    """

    assert all(edge.type != "neuroglial" for edge in circuit.edges.values()), (
        "External circuits with neuroglial connections are not supported. "
        "Non-external astrocytes may connect to newly created external "
        "neuron–neuron connections, which requires generating an additional "
        "edges file. External astrocytes, if possible, are still another problem"
        "that multiplies the amount of additional files required."
    )

    new_nodes = {}

    write_edge_configs = []
    for name, edge in circuit.edges.items():
        if edge.source.type != "virtual" and edge.target.name in id_mapping:
            wanted_src_ids = circuit.nodes[edge.source.name].ids()

            if edge.source.name in id_mapping:
                wanted_src_ids = wanted_src_ids[
                    _isin(
                        wanted_src_ids, id_mapping[edge.source.name].index.to_numpy(), invert=True
                    )
                ]

            # only keep ids that are used; this is duplicating work in _copy_edge_attributes
            # but the alternative is that it keeps track of the new id_mapping; which
            # seemed less ideal
            with h5py.File(_get_storage_path(edge)) as h5:
                all_sgids = h5[f"edges/{name}/source_node_id"]
                all_tgids = h5[f"edges/{name}/target_node_id"]

                # overwrite wanted_src_ids with a DataFrame; the numpy array is not needed
                wanted_src_ids = _get_subcircuit_external_ids(
                    all_sgids,
                    all_tgids,
                    wanted_src_ids,
                    id_mapping[edge.target.name].index.to_numpy(),
                )

            if len(wanted_src_ids) == 0:
                continue

            new_name = f"external_{name}"
            while new_name in existing_edge_pop_names:
                L.debug("%s already exists as an edge population", new_name)
                new_name = "external_" + new_name
            new_source_pop_name = f"external_{edge.source.name}"
            while new_source_pop_name in existing_node_pop_names:
                L.debug("%s already exists as an node population", new_source_pop_name)
                new_source_pop_name = "external_" + new_source_pop_name
            node_pop_name_mapping[new_source_pop_name] = edge.source.name

            output_path = output / (new_name + ".h5")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            L.debug(
                "Writing edges %s for %s -> %s [%s]",
                name,
                edge.source.name,
                edge.target.name,
                output_path,
            )

            if new_source_pop_name in id_mapping:
                # If mapping already exists, only add new IDs w/o changing existing!
                # (May happen if different target populations have same external source population)
                existing_mapping = id_mapping[new_source_pop_name]
                is_existing = _isin(wanted_src_ids.index, existing_mapping.index)
                wanted_src_ids.loc[is_existing] = existing_mapping.loc[
                    wanted_src_ids.loc[is_existing].index
                ]
                new_ids = (
                    np.arange(np.sum(~is_existing))
                    + wanted_src_ids[NEW_IDS].loc[is_existing].max()
                    + 1
                )  # New node IDs begin at the lowest unused value (max + 1)
                wanted_src_ids.loc[~is_existing, NEW_IDS] = new_ids

                # And merge new into existing
                id_mapping[new_source_pop_name] = pd.concat(
                    [existing_mapping, wanted_src_ids.loc[~is_existing]], axis=0
                )
            else:
                id_mapping[new_source_pop_name] = wanted_src_ids

            write_edge_config = WriteEdgeConfig(
                output_path=str(output_path),
                input_path=_get_storage_path(edge),
                src_node_name=new_source_pop_name,
                dst_node_name=edge.target.name,
                src_edge_name=name,
                dst_edge_name=new_name,
                src_mapping=wanted_src_ids,
                dst_mapping=id_mapping[edge.target.name],
                edge_type=edge.type,
            )
            write_edge_configs.append(write_edge_config)

            new_nodes[new_source_pop_name] = (
                edge.source.name,
                wanted_src_ids.index.to_numpy(),
            )

    new_edges_files = _orchestrate_write_subcircuit_edges(write_edge_configs=write_edge_configs)

    new_node_files = {}
    # write new virtual nodes from originally non-virtual populations
    for population_name, id_tuple in new_nodes.items():
        # Get all properties of the subset of the node population that is relevant
        orig_population_name, ids = id_tuple
        df = circuit.nodes[orig_population_name].get(ids).reset_index(drop=True)
        nodes_path = Path(output) / population_name / "nodes.h5"
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        new_node_files[population_name] = _save_sonata_nodes(nodes_path, df, population_name)

    return new_node_files, new_edges_files


def _write_subcircuit_virtual(
    output,
    circuit,
    edge_populations_to_paths,
    id_mapping,
    node_pop_name_mapping,
    list_of_sources_to_ignore=(),
):
    """write all node/edge populations that have virtual nodes as source

    Note: the id_mapping dictionary is updated with the used virtual nodes
    """
    # pylint: disable=too-many-locals
    new_node_files = {}

    virtual_populations = {
        name: edge
        for name, edge in circuit.edges.items()
        if (
            edge.source.type == "virtual"
            and edge.target.name in id_mapping
            and edge.source.name not in list_of_sources_to_ignore
        )
    }

    # gather the ids of the virtual populations that are used; within a circuit
    # it's possible that a virtual population points to multiple target populations
    pop_used_source_node_ids = collections.defaultdict(list)
    for name, edge in virtual_populations.items():
        target_node_ids = id_mapping[edge.target.name].index.to_numpy()
        target_node_ids = bluepysnap.circuit_ids.CircuitNodeIds.from_dict(
            {edge.target.name: target_node_ids}
        )

        pop_used_source_node_ids[edge.source.name].append(
            edge.afferent_nodes(target_node_ids, unique=True)
        )

    pop_used_source_node_ids = {
        name: np.unique(np.concatenate(ids))
        for name, ids in pop_used_source_node_ids.items()
        if len(np.concatenate(ids)) > 0  # Exclude empty sources
    }

    # Remove edge populations with empty sources
    virtual_populations = {
        name: edge
        for name, edge in virtual_populations.items()
        if edge.source.name in pop_used_source_node_ids
    }

    # update the mappings with the virtual nodes
    for name, ids in pop_used_source_node_ids.items():
        id_mapping[name] = pd.DataFrame({NEW_IDS: range(len(ids))}, index=ids)
        # Virtual input sources retain their name unchanged
        node_pop_name_mapping[name] = name

    # write the edges that have the virtual populations as source

    write_edge_configs = []
    for edge_pop_name, edge in virtual_populations.items():
        write_edge_config = WriteEdgeConfig(
            output_path=Path(output) / edge_populations_to_paths[edge_pop_name],
            input_path=_get_storage_path(edge),  # Where to read from
            src_node_name=edge.source.name,
            dst_node_name=edge.target.name,
            src_edge_name=edge_pop_name,
            dst_edge_name=edge_pop_name,
            src_mapping=id_mapping[edge.source.name],
            dst_mapping=id_mapping[edge.target.name],
            edge_type=edge.type,
        )
        write_edge_configs.append(write_edge_config)

    new_edges_files = _orchestrate_write_subcircuit_edges(write_edge_configs=write_edge_configs)

    # write virtual nodes based on virtual populations
    for population_name, ids in pop_used_source_node_ids.items():
        # Get all properties of the subset of the node population that is relevant
        df = circuit.nodes[population_name].get(ids).reset_index(drop=True)
        nodes_path = Path(output) / population_name / "nodes.h5"
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        new_node_files[population_name] = _save_sonata_nodes(nodes_path, df, population_name)

    return new_node_files, new_edges_files


def _update_config_with_new_paths(output, config, new_population_files, type_):
    """Update config file with the new paths

    Args:
        output: path to output
        config(dict): SONATA config
        new_population_files(dict): population -> path mapping of updated populations
        type_(str): 'nodes' or 'edges'
    """
    assert type_ in (
        "nodes",
        "edges",
    ), f'{type_} must be "nodes" or "edges"'
    output = str(output)

    config = copy.deepcopy(config)
    config["manifest"] = {"$BASE_DIR": "./"}

    def _strip_base_path(path):
        assert path.startswith(output), f"missing output path ({output}) in {path}"
        path = path[len(output) :]
        if path.startswith("/"):
            path = path[1:]
        return path

    str_type = f"{type_}_file"
    if type_ == "nodes":
        default_type = "virtual"
    else:  # Must be edges. This is checked above.
        default_type = "chemical"
    old_population_list = copy.deepcopy(config["networks"][type_])
    config["networks"][type_] = []
    for new_pop_name, new_pop_path in new_population_files.items():
        if (
            new_pop_path == DELETED_EMPTY_EDGES_FILE
            or new_pop_path == DELETED_EMPTY_EDGES_POPULATION
        ):
            continue

        updated_path = _strip_base_path(str(new_pop_path))

        matched_originals = [
            _entry["populations"][new_pop_name]
            for _entry in old_population_list
            if new_pop_name in _entry["populations"]
        ]
        assert 0 <= len(matched_originals) <= 1
        entry = {
            str_type: str(Path("$BASE_DIR") / updated_path),
            "populations": {new_pop_name: copy.deepcopy(matched_originals[0])}
            if len(matched_originals) == 1
            else {new_pop_name: {}},
        }
        config["networks"][type_].append(entry)
        config["networks"][type_][-1]["populations"][new_pop_name].setdefault("type", default_type)
    return config


def _update_node_sets(node_sets, id_mapping):
    """using the `id_mapping`, update impacted `node_sets`

    Note: impacted means they have a 'population', and they have 'node_ids' that changed
    """
    ret = {}
    for name, rule in node_sets.items():
        # Note: 'node_id' predicates without 'population' aren't copied, since it
        # doesn't makes sense to pick a node_id without specifying its population
        if isinstance(rule, dict) and "node_id" in rule:
            if "population" not in rule:
                L.warning("No population key in nodeset %s, ignoring", name)
                continue

            if rule["population"] not in id_mapping:
                continue

            mapping = id_mapping[rule["population"]]
            ret[name] = rule
            ret[name]["node_id"] = (
                mapping.loc[mapping.index.intersection(rule["node_id"])][NEW_IDS]
                .sort_values()
                .to_list()
            )
        else:
            ret[name] = rule

    return ret


def _mapping_to_parent_dict(id_mapping, node_pop_name_mapping):
    mapping = {}
    for population, df in id_mapping.items():
        mapping[population] = {
            PARENT_IDS: df.index.to_list(),
            NEW_IDS: df[NEW_IDS].to_list(),
            PARENT_NAME: node_pop_name_mapping[population],
        }
    return mapping


def _make_parent_the_original_mapping(this_mapping):
    for this_pop in this_mapping.keys():
        this_mapping[this_pop][ORIG_IDS] = this_mapping[this_pop][PARENT_IDS]
        this_mapping[this_pop][ORIG_NAME] = this_mapping[this_pop][PARENT_NAME]


def _add_mapping_to_original(this_mapping, parent_mapping):
    for this_pop in this_mapping.keys():
        parent_pop = this_mapping[this_pop][PARENT_NAME]

        backwards_mapped = pd.Series(
            parent_mapping[parent_pop][ORIG_IDS], index=parent_mapping[parent_pop][NEW_IDS]
        )
        orig_ids = backwards_mapped[this_mapping[this_pop][PARENT_IDS]]
        orig_name = parent_mapping[parent_pop][ORIG_NAME]

        this_mapping[this_pop][ORIG_IDS] = orig_ids.to_list()
        this_mapping[this_pop][ORIG_NAME] = orig_name


def _write_mapping(output, parent_circ, id_mapping, node_pop_name_mapping):
    """write the id mappings between the old and new populations for future analysis"""
    this_mapping = _mapping_to_parent_dict(id_mapping, node_pop_name_mapping)

    provenance = parent_circ.config.get("components", {}).get("provenance", {})
    if "id_mapping" in provenance:
        # Currently, bluepysnap does not seem to resolve $BASE_DIR for entries in "provenance".
        # Therefore I decided to not prepend it and just assume the file exists near the circuit config.
        parent_root = Path(parent_circ._circuit_config_path).parent
        parent_mapping = utils.load_json(parent_root / provenance["id_mapping"])
        _add_mapping_to_original(this_mapping, parent_mapping)
    else:
        _make_parent_the_original_mapping(this_mapping)

    mapping_fn = "id_mapping.json"
    utils.dump_json(output / mapping_fn, this_mapping)
    return mapping_fn


def split_subcircuit(
    output,
    node_set_name,
    circuit,
    do_virtual,
    create_external,
    list_of_virtual_sources_to_ignore=(),
):
    """Split a single subcircuit out of circuit, based on nodeset

    Args:
        output(str): path where files will be written
        node_set_name(str): name of nodeset to extract
        circuit(bluepysnap.Circuit|str): Sonata circuit object or path to circuit_config sonata file
        do_virtual(bool): whether to split out the virtual nodes that target the cells
            contained in the specified nodeset
        create_external(bool): whether to create new virtual populations of all the
            incoming connections
        list_of_virtual_sources_to_ignore (list[str] or tuple[str], default=()): Only considered if
            do_virtual==True. List of names of virtual source node populations. Virtual edge
            populations associated with these sources will NOT be extracted into the subcircuit.
    """
    # pylint: disable=too-many-locals
    output = Path(output)

    if isinstance(circuit, (str, Path)):
        circuit = bluepysnap.Circuit(circuit)
    else:
        assert isinstance(circuit, bluepysnap.Circuit), "Path or sonata circuit object required!"

    node_pop_to_paths, edge_pop_to_paths = _gather_layout_from_networks(circuit.config["networks"])

    # TODO: remove backward compatibility with snap 1.0, when the dependency can be updated.
    #  In snap 2.0 it's possible to simplify:
    #    pop.get(pop.ids(node_set_name, raise_missing_property=False))
    #  with:
    #    pop.get(node_set_name, raise_missing_property=False)
    split_populations = {
        pop_name: pop.get(pop.ids(node_set_name, raise_missing_property=False))
        for pop_name, pop in circuit.nodes.items()
        if not pop.type == "virtual"
    }
    split_populations = {pop_name: df for pop_name, df in split_populations.items() if not df.empty}

    id_mapping = _get_node_id_mapping(split_populations)
    # Intrinsic input sources retain their name unchanged
    node_pop_name_mapping = {pop_name: pop_name for pop_name in split_populations.keys()}

    # TODO: should function `_write_subcircuit_biological`,
    # `_write_subcircuit_external`, `_write_subcircuit_virtual`
    # handle node updates and config updates?

    new_node_files, new_edge_files = _write_subcircuit_biological(
        output, circuit, node_pop_to_paths, edge_pop_to_paths, split_populations, id_mapping
    )

    if do_virtual:
        new_virtual_node_files, new_virtual_edge_files = _write_subcircuit_virtual(
            output,
            circuit,
            edge_pop_to_paths,
            id_mapping,
            node_pop_name_mapping,
            list_of_virtual_sources_to_ignore,
        )

        new_node_files.update(new_virtual_node_files)
        new_edge_files.update(new_virtual_edge_files)

    existing_node_pop_names = list(new_node_files.keys())
    existing_edge_pop_names = list(new_edge_files.keys())
    if create_external:
        new_virtual_node_files, new_virtual_edge_files = _write_subcircuit_external(
            output,
            circuit,
            id_mapping,
            node_pop_name_mapping,
            existing_node_pop_names,
            existing_edge_pop_names,
        )

        new_node_files.update(new_virtual_node_files)
        new_edge_files.update(new_virtual_edge_files)

    mapping_fn = _write_mapping(output, circuit, id_mapping, node_pop_name_mapping)

    config = copy.deepcopy(circuit.config)

    node_sets = _update_node_sets(utils.load_json(config["node_sets_file"]), id_mapping)
    utils.dump_json(output / "node_sets.json", node_sets)
    config["node_sets_file"] = "$BASE_DIR/node_sets.json"

    # update circuit_config
    config = _update_config_with_new_paths(output, config, new_node_files, type_="nodes")
    config = _update_config_with_new_paths(output, config, new_edge_files, type_="edges")

    # TODO: Should be "$BASE_DIR/" + mapping_fn. But bluepysnap does not seem to resolve
    # $BASE_DIR for entries in "provenance"..? So I don't even try.
    config.setdefault("components", {}).setdefault("provenance", {})["id_mapping"] = mapping_fn
    utils.dump_json(output / "circuit_config.json", config)
