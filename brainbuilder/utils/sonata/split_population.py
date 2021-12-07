"""Split a SONATA node/edge population into sub-populations"""
import itertools
import logging
import os
from pathlib import Path

import h5py
import libsonata
import numpy as np
import pandas as pd
import voxcell

from brainbuilder import utils
from brainbuilder.utils import dump_json

L = logging.getLogger(__name__)

# So as not to exhaust memory, the edges files are loaded/written in chunks of this size
H5_READ_CHUNKSIZE = 500000000
# Name of the unique expected group in sonata nodes and edges files
GROUP_NAME = '0'


def _get_population_name(src, dst, synapse_type='chemical'):
    """Return the population name based off `src` and `dst` node population names."""
    return src if src == dst else f'{src}__{dst}__{synapse_type}'


def _get_edge_file_name(output, new_pop_name):
    """Return the name of the edge file split by population."""
    return os.path.join(output, f'edges_{new_pop_name}.h5')


def _get_node_file_name(output, new_pop_name):
    """Return the name of the node file split by population."""
    return os.path.join(output, f'nodes_{new_pop_name}.h5')


def _get_unique_population(parent):
    """Return the h5 unique population, raise an exception if not unique."""
    population_names = list(parent)
    if len(population_names) != 1:
        raise ValueError(f'Single population is supported only, found {population_names}')
    return parent[population_names[0]]


def _get_unique_group(parent):
    """Return the h5 group 0, raise an exception if non present."""
    if GROUP_NAME not in parent:
        raise ValueError(f'Single group {GROUP_NAME!r} is required')
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
    cell_collection.save_sonata(nodes_path)
    # restore the original index
    df.index -= 1


def _init_edge_group(orig_group, new_group):
    """Copy the empty datasets from orig_group to new_group.

    Args:
        orig_group (h5py.Group): original group, e.g. /edges/default/0
        new_group (h5py.Group): new group, e.g. /edges/L2_X__L6_Y__chemical/0

    """
    for name, attr in orig_group.items():
        if isinstance(attr, h5py.Dataset):
            utils.create_appendable_dataset(new_group, name, attr.dtype)
        elif isinstance(attr, h5py.Group) and name == 'dynamics_params':
            new_group.create_group(name)
            for k, values in attr.items():
                if isinstance(values, h5py.Dataset):
                    utils.create_appendable_dataset(new_group[name], k, values.dtype)
                else:
                    L.warning('Not copying dynamics_params subgroup %s', k)
        else:
            raise ValueError('Only "dynamics_params" group is expected')


def _populate_edge_group(orig_group, new_group, sl, mask):
    """Populate the datasets from orig_group to new_group.

    Args:
        orig_group (h5py.Group): original group, e.g. /edges/default/0
        new_group (h5py.Group): new group, e.g. /edges/L2_X__L6_Y__chemical/0
        sl (slice): slice used to select the dataset range
        mask (np.ndarray): mask used to filter the dataset

    """
    for name, attr in orig_group.items():
        if isinstance(attr, h5py.Dataset):
            utils.append_to_dataset(new_group[name], attr[sl][mask])
        elif isinstance(attr, h5py.Group) and name == 'dynamics_params':
            for k, values in attr.items():
                if isinstance(values, h5py.Dataset):
                    utils.append_to_dataset(new_group[name][k], values[sl][mask])
        else:
            raise ValueError('Only "dynamics_params" group is expected')


def _finalize_edges(new_edges):
    edge_count = len(new_edges['source_node_id'])
    new_edges['edge_type_id'] = np.full(edge_count, -1)
    new_edges['edge_group_id'] = np.full(edge_count, 0)
    new_edges['edge_group_index'] = np.arange(edge_count)


def _copy_edge_attributes(h5in, h5out, src_node_pop, dst_node_pop, id_mapping, h5_read_chunk_size):
    """Copy the attributes from the original edges into the new edge populations"""
    orig_edges = _get_unique_population(h5in['edges'])
    orig_group = _get_unique_group(orig_edges)
    new_edges = h5out.create_group('edges/' + _get_population_name(src_node_pop, dst_node_pop))
    new_group = new_edges.create_group(GROUP_NAME)

    utils.create_appendable_dataset(new_edges, 'source_node_id', np.uint64)
    utils.create_appendable_dataset(new_edges, 'target_node_id', np.uint64)
    new_edges['source_node_id'].attrs['node_population'] = src_node_pop
    new_edges['target_node_id'].attrs['node_population'] = dst_node_pop

    _init_edge_group(orig_group, new_group)

    for start in range(0, len(orig_edges['source_node_id']), h5_read_chunk_size):
        sl = slice(start, start + h5_read_chunk_size)
        sgids = orig_edges['source_node_id'][sl]
        tgids = orig_edges['target_node_id'][sl]
        mask = (np.isin(sgids, id_mapping[src_node_pop].index.to_numpy()) &
                np.isin(tgids, id_mapping[dst_node_pop].index.to_numpy()))
        if np.any(mask):
            utils.append_to_dataset(
                new_edges['source_node_id'],
                id_mapping[src_node_pop].loc[sgids[mask]].new_id.to_numpy()
            )
            utils.append_to_dataset(
                new_edges['target_node_id'],
                id_mapping[dst_node_pop].loc[tgids[mask]].new_id.to_numpy()
            )
            _populate_edge_group(orig_group, new_group, sl, mask)

    _finalize_edges(new_edges)


def _get_node_counts(h5out, new_pop_name):
    source_node_count = target_node_count = 0
    new_edges = h5out['edges'][new_pop_name]
    edge_count = len(new_edges['source_node_id'])
    if edge_count > 0:
        # add 1 because IDs are 0-based
        source_node_count = int(np.max(new_edges['source_node_id']) + 1)
        target_node_count = int(np.max(new_edges['target_node_id']) + 1)
    return edge_count, source_node_count, target_node_count


def _write_indexes(edge_file_name, new_pop_name, source_node_count, target_node_count):
    libsonata.EdgePopulation.write_indices(
        edge_file_name, new_pop_name, source_node_count, target_node_count
    )


def _check_written_edges(h5in, written_edges):
    """Verify that the number of written edges matches the number of initial edges."""
    orig_edges = _get_unique_population(h5in['edges'])
    expected_edges = len(orig_edges['source_node_id'])
    if expected_edges != written_edges:
        msg = f'Written edges mismatch: expected={expected_edges}, actual={written_edges}'
        raise RuntimeError(msg)


def _write_edges(output, edges_path, id_mapping, h5_read_chunk_size=H5_READ_CHUNKSIZE):
    """create all new edge populations in separate files"""
    with h5py.File(edges_path, 'r') as h5in:
        written_edges = 0
        for src_node_pop, dst_node_pop in itertools.product(id_mapping, id_mapping):
            edge_pop_name = _get_population_name(src_node_pop, dst_node_pop)
            edge_file_name = _get_edge_file_name(output, edge_pop_name)
            # write the new edges h5 file
            with h5py.File(edge_file_name, 'w') as h5out:
                _copy_edge_attributes(
                    h5in, h5out, src_node_pop, dst_node_pop, id_mapping, h5_read_chunk_size,
                )
                edge_count, sgid_count, tgid_count = _get_node_counts(h5out, edge_pop_name)
            # after the h5 file is closed, it's indexed if valid, or it's removed if empty
            if edge_count:
                _write_indexes(edge_file_name, edge_pop_name, sgid_count, tgid_count)
                L.debug('Written %s edges to %s', edge_count, edge_file_name)
                written_edges += edge_count
            else:
                os.unlink(edge_file_name)
        # verify that all the edges have been written
        _check_written_edges(h5in, written_edges)


def _write_nodes(output, split_nodes):
    """create all new node populations in separate files"""
    for new_population, df in split_nodes.items():
        df = df.reset_index(drop=True)
        nodes_path = _get_node_file_name(output, new_population)
        _save_sonata_nodes(nodes_path, df, population_name=new_population)
        L.debug('Written %s nodes to %s', len(df), nodes_path)


def _get_node_id_mapping(split_nodes):
    """return a dict split_nodes.keys() -> DataFrame with index old_ids, and colunm new_id"""
    id_mapping = {}
    for new_population, df in split_nodes.items():
        id_mapping[new_population] = pd.DataFrame(index=df.index)
        id_mapping[new_population]['new_id'] = np.arange(len(id_mapping[new_population]))
    return id_mapping


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
    L.debug('Splitting population on %s -> %s', attribute, nodes[attribute].unique())
    split_nodes = dict(tuple(nodes.groupby(attribute)))
    return split_nodes


def _write_circuit_config(output, split_nodes):
    """Write a simple circuit-config.json for all the node/edge populations created"""
    tmpl = {"manifest": {"$BASE_DIR": ".",
                         },
            "networks": {"nodes": [],
                         "edges": [],
                         },
            }

    for src, dst in itertools.product(split_nodes, split_nodes):
        new_pop_name = _get_population_name(src, dst)
        if src == dst:
            tmpl['networks']['nodes'].append(
                {
                    'nodes_file': _get_node_file_name('$BASE_DIR', new_pop_name),
                    'node_types_file': None,
                })

        if os.path.exists(_get_edge_file_name(output, new_pop_name)):
            tmpl['networks']['edges'].append(
                {
                    'edges_file': _get_edge_file_name('$BASE_DIR', new_pop_name),
                    'edge_types_file': None
                })

    filepath = Path(output) / 'circuit_config.json'
    dump_json(filepath, tmpl)
    L.debug('Written circuit config %s', filepath)


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
    _write_edges(output, edges_path, id_mapping)

    _write_circuit_config(output, split_populations)
