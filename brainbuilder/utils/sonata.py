"""
Temporary SONATA converters.

https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md
"""

import json
import logging
import os.path
import glob

from collections import OrderedDict

import numpy as np
import pandas as pd
import h5py
import six

from voxcell import CellCollection
from voxcell.sonata import NodePopulation
from bluepy.v2.impl.target import TargetContext

from brainbuilder.exceptions import BrainBuilderError


L = logging.getLogger('brainbuilder')


def _load_mecombo_info(filepath):
    COLUMN_FILTER = lambda name: name not in ('morph_name', 'layer', 'fullmtype', 'etype')
    return pd.read_csv(filepath, sep=r'\s+', usecols=COLUMN_FILTER, index_col='combo_name')


def _mvd3_to_nodes(cells, name, mecombo_info_path=None):
    """ Convert CellCollection to SONATA NodePopulation. """
    if 'me_combo' in cells.properties:
        if mecombo_info_path is None:
            raise BrainBuilderError("""
                Please specify 'mecombo_info_path' in order to resolve 'me_combo' property
            """)
        mecombo_info = _load_mecombo_info(mecombo_info_path)
        L.info(
            "'me_combo' property would be resolved to 'model_template' and dynamics parameters %s",
            ", ".join("'%s'" % s for s in mecombo_info.columns if s != 'emodel')
        )

    result = NodePopulation(name, size=len(cells.properties))

    if cells.positions is not None:
        result.positions = cells.positions

    if cells.orientations is not None:
        result.orientations = cells.orientations

    for prop, column in cells.properties.iteritems():
        if prop == 'me_combo':
            continue
        result.attributes[prop] = column.values

    if 'me_combo' in cells.properties:
        mecombo_params = mecombo_info.loc[cells.properties['me_combo']]
        for prop, column in mecombo_params.iteritems():
            values = column.values
            if prop == 'emodel':
                values = [('hoc:' + v) for v in values]
                result.attributes['model_template'] = values
            else:
                result.dynamics_attributes[prop] = values

    return result


def write_nodes_from_mvd3(mvd3_path,
                          mecombo_info_path,
                          out_h5_path,
                          population,
                          library_properties):
    """ Export MVD3 + MEComboInfoFile to SONATA node collection. """
    cells = CellCollection.load_mvd3(mvd3_path)
    nodes = _mvd3_to_nodes(cells, population, mecombo_info_path)
    nodes.save(out_h5_path, library_properties)


def _write_edge_group(group, out):
    # TODO: pick only those used, remap to those mentioned in "spec"
    # conductance -> syn_weight
    # ...
    MAPPING = {
        'morpho_section_id_post': 'afferent_section_id',
        'morpho_segment_id_post': 'afferent_segment_id',
        'morpho_offset_segment_post': 'afferent_segment_offset',
        'morpho_section_fraction_post': 'afferent_section_pos',
        'morpho_section_type_post': 'afferent_section_type',

        'morpho_section_id_pre': 'efferent_section_id',
        'morpho_segment_id_pre': 'efferent_segment_id',
        'morpho_offset_segment_pre': 'efferent_segment_offset',
        'morpho_section_fraction_pre': 'efferent_section_pos',
        'morpho_section_type_pre': 'efferent_section_type',

        'morpho_spine_length': 'spine_length',
        'morpho_type_id_pre': 'efferent_morphology_id',

        'position_center_post_x': 'afferent_center_x',
        'position_center_post_y': 'afferent_center_y',
        'position_center_post_z': 'afferent_center_z',
        'position_center_pre_x': 'efferent_center_x',
        'position_center_pre_y': 'efferent_center_y',
        'position_center_pre_z': 'efferent_center_z',
        'position_contour_post_x': 'afferent_surface_x',
        'position_contour_post_y': 'afferent_surface_y',
        'position_contour_post_z': 'afferent_surface_z',
        'position_contour_pre_x': 'efferent_surface_x',
        'position_contour_pre_y': 'efferent_surface_y',
        'position_contour_pre_z': 'efferent_surface_z',
    }
    for prop in group:
        if prop in ('connected_neurons_post', 'connected_neurons_pre'):
            continue
        name = MAPPING.get(prop, prop)
        L.info("'%s' -> '%s'...", prop, name)
        group.copy(prop, out, name=name)


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


def _node_populations(nodes_dir, nodes):
    return [
        OrderedDict([
            ('nodes_file', os.path.join(nodes_dir, population, 'nodes.h5')),
            ('node_types_file', None),
            ('node_sets_file', os.path.join(nodes_dir, population, 'node_sets.json')),
        ])
        for population in nodes
    ]


def _edge_populations(edges_dir, edges_suffix, edges):
    return [
        OrderedDict([
            ('edges_file', os.path.join(edges_dir, population, 'edges%s.h5' % edges_suffix)),
            ('edge_types_file', None),
        ])
        for population in edges
    ]


def write_network_config(base_dir,
                         morph_dir,
                         emodel_dir,
                         nodes_dir, nodes,
                         edges_dir, edges_suffix, edges,
                         output_path):
    """ Write SONATA network config """
    # pylint: disable=too-many-arguments
    content = OrderedDict()
    content['manifest'] = {
        '$BASE_DIR': base_dir,
        '$NETWORK_NODES_DIR': os.path.join('$BASE_DIR', nodes_dir),
        '$NETWORK_EDGES_DIR': os.path.join('$BASE_DIR', edges_dir),
    }
    content['components'] = {
        'morphologies_dir': os.path.join('$BASE_DIR', morph_dir),
        'biophysical_neuron_models_dir': os.path.join('$BASE_DIR', emodel_dir),
    }
    content['networks'] = OrderedDict([
        ('nodes', _node_populations('$NETWORK_NODES_DIR', nodes)),
        ('edges', _edge_populations('$NETWORK_EDGES_DIR', edges_suffix, edges)),
    ])
    with open(output_path, 'w') as f:
        json.dump(content, f, indent=2)


def write_node_set_from_targets(input_dir, output_file):
    """Write SONATA node_set from all target files in a directory.

    This function allows to directly convert a set of target files created from a mvd3 file
    into the corresponding node_set.json file. This is useful if user does not have the yaml target
    rules anymore.

    The 'brainbuilder targets node-sets' should be preferred if possible.
    """
    target = TargetContext.load(glob.glob(input_dir + '/*.target'))
    if not os.path.basename(output_file) == 'node_sets.json':
        basename = os.path.basename(output_file)
        L.warning('basename "%s" is not "node_sets.json" change your config file accordingly.',
                  basename)
    output_dict = {}
    for target_name in target.names:
        # targets are built from a mvd3 file so indexing starts from 1 compare to 0 in SONATA
        node_ids = target.resolve(target_name) - 1
        output_dict[target_name] = {"node_id": node_ids.tolist()}

    with open(output_file, "w") as fd:
        json.dump(output_dict, fd, indent=2)
