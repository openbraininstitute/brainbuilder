"""
Target generation.
"""

import json
import collections
import logging

import click
import six
import yaml

from bluepy.v2 import Circuit
from voxcell import ROIMask

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import bbp

L = logging.getLogger('brainbuilder')


@click.group()
def app():
    """ Tools for working with .target files """


def _synapse_class_name(synclass):
    return {
        'EXC': 'Excitatory',
        'INH': 'Inhibitory',
    }[synclass]


def _layer_name(layer):
    return "Layer%s" % layer


def _column_name(column):
    return "mc%d_Column" % column


def write_default_targets(cells, output):
    """ Write default property-based targets. """
    bbp.write_target(output, 'Mosaic', include_targets=['All'])
    bbp.write_target(output, 'All', include_targets=sorted(cells['mtype'].unique()))
    bbp.write_property_targets(output, cells, 'synapse_class', mapping=_synapse_class_name)
    bbp.write_property_targets(output, cells, 'mtype')
    bbp.write_property_targets(output, cells, 'etype')
    bbp.write_property_targets(output, cells, 'region')


def write_query_targets(query_based, circuit, output, allow_empty=False):
    """ Write targets based on BluePy-like queries. """
    for name, query in six.iteritems(query_based):
        gids = circuit.cells.ids(query)
        if len(gids) < 1:
            msg = "Empty target: {} {}".format(name, query)
            if allow_empty:
                L.warning(msg)
            else:
                raise BrainBuilderError(msg)
        bbp.write_target(output, name, gids=gids)


def _load_targets(filepath):
    """
    Load target definition YAML, e.g.:

    >
      targets:
        # BluePy-like queries a.k.a. "smart targets"
        query_based:
            mc2_Column: {'region': '@^mc2'}
            Layer1: {'region': '@1$'}

        # 0/1 masks registered in the atlas
        atlas_based:
            cylinder: '{S1HL-cylinder}'
    """
    with open(filepath) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)['targets']
    return (
        content.get('query_based'),
        content.get('atlas_based'),
    )


@app.command()
@click.argument("mvd3")
@click.option("--atlas", help="Atlas URL / path", default=None, show_default=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("-t", "--targets", help="Path to target definition YAML file", default=None)
@click.option("--allow-empty", is_flag=True, help="Allow empty targets", show_default=True)
@click.option("-o", "--output", help="Path to output .target file", required=True)
def from_mvd3(mvd3, atlas, atlas_cache, targets, allow_empty, output):
    """ Generate .target file from MVD3 (and target definition YAML) """
    # pylint: disable=too-many-locals
    circuit = Circuit({'cells': mvd3})
    cells = circuit.cells.get()
    with open(output, 'w') as f:
        write_default_targets(cells, f)
        if targets is None:
            if 'layer' in cells:
                bbp.write_property_targets(f, cells, 'layer', mapping=_layer_name)
            if 'hypercolumn' in cells:
                bbp.write_property_targets(f, cells, 'hypercolumn', mapping=_column_name)
        else:
            query_based, atlas_based = _load_targets(targets)
            if query_based is not None:
                write_query_targets(query_based, circuit, f, allow_empty=allow_empty)
            if atlas_based is not None:
                from voxcell.nexus.voxelbrain import Atlas
                if atlas is None:
                    raise BrainBuilderError("Atlas not provided")
                atlas = Atlas.open(atlas, cache_dir=atlas_cache)
                xyz = cells[['x', 'y', 'z']].values
                for name, dset in six.iteritems(atlas_based):
                    mask = atlas.load_data(dset, cls=ROIMask).lookup(xyz)
                    bbp.write_target(f, name, cells.index[mask])


@app.command()
@click.argument("mvd3")
@click.option("-t", "--targets", help="Path to target definition YAML file", default=None)
@click.option("--allow-empty", is_flag=True, help="Allow empty targets", show_default=True)
@click.option("-o", "--output", help="Path to output JSON file", required=True)
def node_sets(mvd3, targets, allow_empty, output):
    """Generate JSON node sets from MVD3 (and target definition YAML)"""

    result = collections.OrderedDict()

    def _add_node_sets(to_add):
        for name, query in sorted(to_add.items()):
            if name in result:
                raise BrainBuilderError("Duplicate node set: '%s'" % name)
            count = cells.count(query)
            if count > 0:
                L.info("Target '%s': %d cell(s)", name, count)
            else:
                msg = "Empty target: {} {}".format(name, query)
                if allow_empty:
                    L.warning(msg)
                else:
                    raise BrainBuilderError(msg)
            result[name] = query

    cells = Circuit({'cells': mvd3}).cells

    _add_node_sets({
        'All': {},
        'Excitatory': {'synapse_class': 'EXC'},
        'Inhibitory': {'synapse_class': 'INH'},
    })

    for prop in ['mtype', 'etype', 'region']:
        _add_node_sets({
            val: {prop: val} for val in cells.get(properties=prop).unique()
        })

    if targets is not None:
        query_based, _ = _load_targets(targets)
        if query_based is not None:
            _add_node_sets(query_based)

    with open(output, 'w') as f:
        json.dump(result, f, indent=2)
