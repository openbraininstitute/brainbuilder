"""
Temporary SONATA converters.

https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md
"""

import json
import logging
import os.path

from collections import OrderedDict

import numpy as np
import pandas as pd
import h5py
import six
import transforms3d

from voxcell import CellCollection


L = logging.getLogger('brainbuilder')


def _write_dataframe_by_columns(df, out):
    for name, column in df.iteritems():
        values, dtype = column.values, None
        if values.dtype in (object,):
            dtype = h5py.special_dtype(vlen=six.text_type)
        out.create_dataset(name, data=values, dtype=dtype)


def _write_node_group(node_group, out):
    properties, dynamic_params = node_group
    _write_dataframe_by_columns(properties, out)
    if dynamic_params is not None:
        _write_dataframe_by_columns(dynamic_params, out.create_group('dynamic_params'))


def _write_node_population(node_group, out):
    group_id = 0
    group_size = len(node_group[0])
    node_group_id = np.full(group_size, group_id)
    node_group_index = np.arange(group_size)
    out.create_dataset('node_group_id', data=node_group_id, dtype=np.uint32)
    out.create_dataset('node_group_index', data=node_group_index, dtype=np.uint32)
    out.create_dataset('node_type_id', data=np.full_like(node_group_id, -1, dtype=np.int32))
    _write_node_group(node_group, out.create_group(str(group_id)))


def _load_mecombo_info(filepath):
    COLUMN_FILTER = lambda name: name not in ('morph_name', 'layer', 'fullmtype', 'etype')
    return pd.read_csv(filepath, sep=r'\s+', usecols=COLUMN_FILTER, index_col='combo_name')


def _mvd3_to_node_group(mvd3_path, mecombo_info_path=None):
    """
    Convert MVD3 to a pair of pandas DataFrames with columns named SONATA-way.

    First DataFrame contains 'general' properties placed to node group root:
        - 'x', 'y', 'z'
        - 'rotation_angle_[x|y|z]axis'
        - 'morphology'
        - ...

    Second DataFrame contains emodel parameters placed to subgroup 'dynamic_params'.
    """
    properties = CellCollection.load_mvd3(mvd3_path).as_dataframe().reset_index(drop=True)

    if 'orientation' in properties:
        orientations = properties.pop('orientation')
        angles = np.stack(
            transforms3d.euler.mat2euler(m, axes='szyx') for m in orientations
        )
        properties['rotation_angle_xaxis'] = angles[:, 2]
        properties['rotation_angle_yaxis'] = angles[:, 1]
        properties['rotation_angle_zaxis'] = angles[:, 0]

    if ('me_combo' in properties) and (mecombo_info_path is not None):
        mecombos = properties.pop('me_combo')
        mecombo_info = _load_mecombo_info(mecombo_info_path)
        mecombo_params = mecombo_info.loc[mecombos]
        dynamic_params = pd.DataFrame(index=properties.index)
        for name, series in mecombo_params.iteritems():
            values = series.values
            if name == 'emodel':
                values = [('hoc:' + v) for v in values]
                properties['model_template'] = values
            else:
                dynamic_params[name] = values
    else:
        dynamic_params = None

    return properties, dynamic_params


def write_nodes_from_mvd3(mvd3_path, mecombo_info_path, out_h5_path, population):
    """ Export MVD3 + MEComboInfoFile to SONATA node collection. """
    node_group = _mvd3_to_node_group(mvd3_path, mecombo_info_path)
    with h5py.File(out_h5_path, 'w') as h5f:
        _write_node_population(
            node_group,
            h5f.create_group('/nodes/%s' % population)
        )


def _write_edge_group(group, out):
    # TODO: pick only those used, remap to those mentioned in "spec"
    # conductance -> syn_weight
    # morpho_section_id_pre -> afferent_section_id
    # ...
    for prop in group:
        if prop in ('connected_neurons_post', 'connected_neurons_pre'):
            continue
        L.info("'%s'...", prop)
        group.copy(prop, out)


def _write_edge_index(index, out):
    index.copy('neuron_id_to_range', out, name='node_id_to_ranges')
    index.copy('range_to_synapse_id', out, name='range_to_edge_id')


def _write_edge_population(population, source, target, out):
    properties, indices = population['properties'], population['indexes']
    count = len(properties['connected_neurons_pre'])

    L.info("Writing population-level datasets...")

    L.info("'edge_type_id'...")
    out.create_dataset('edge_type_id', shape=(count,), dtype=np.int8, fillvalue=-1)

    L.info("'source_node_id'...")
    properties.copy('connected_neurons_pre', out, name='source_node_id')
    out['source_node_id'].attrs['node_population'] = six.text_type(source)

    L.info("'target_node_id'...")
    properties.copy('connected_neurons_post', out, name='target_node_id')
    out['target_node_id'].attrs['node_population'] = six.text_type(target)

    L.info("Writing group-level datasets...")
    _write_edge_group(properties, out.create_group('0'))

    L.info("Writing indices...")

    L.info("'source_to_target'...")
    _write_edge_index(
        indices['connected_neurons_pre'], out.create_group('indices/source_to_target')
    )

    L.info("'target_to_source'...")
    _write_edge_index(
        indices['connected_neurons_post'], out.create_group('indices/target_to_source')
    )


def write_edges_from_syn2(syn2_path, population, source, target, out_h5_path):
    """ Export SYN2 to SONATA edge collection. """
    with h5py.File(syn2_path, 'r') as syn2:
        with h5py.File(out_h5_path, 'w') as h5f:
            assert len(syn2['/synapses']) == 1
            syn2_population = next(iter(syn2['/synapses'].values()))
            _write_edge_population(
                syn2_population, source, target,
                h5f.create_group('/edges/%s' % population)
            )


def _normalize_path(base_dir, alias):
    base_dir = os.path.realpath(base_dir)

    def _func(path):
        result = os.path.realpath(path)
        if result.startswith(base_dir):
            result = result.replace(base_dir, alias, 1)
        return result

    return _func


def _populations(filepaths, prefix, normalize_path):
    h5_key = '%ss_file' % prefix
    csv_key = '%s_types_file' % prefix
    return [
        OrderedDict([
            (h5_key, normalize_path(path)),
            (csv_key, None),
        ])
        for path in filepaths
    ]


def write_network_config(base_dir, morph_dir, node_files, edge_files, output_path):
    """ Write SONATA network config """
    content = OrderedDict()
    content['manifest'] = {
        '$BASE_DIR': os.path.realpath(base_dir)
    }
    normalize_path = _normalize_path(base_dir, '$BASE_DIR')
    content['components'] = {
        'morphologies_dir': normalize_path(morph_dir)
    }
    content['networks'] = OrderedDict([
        ('nodes', _populations(node_files, 'node', normalize_path)),
        ('edges', _populations(edge_files, 'edge', normalize_path)),
    ])
    with open(output_path, 'w') as f:
        json.dump(content, f, indent=2)
