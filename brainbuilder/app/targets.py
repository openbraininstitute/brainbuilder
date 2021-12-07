"""Target generation."""
# pylint: disable=import-outside-toplevel
import collections
import logging

import click

from bluepy import Circuit
from voxcell import ROIMask

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import bbp, load_yaml, dump_json

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
    for name, query in query_based.items():
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
    content = load_yaml(filepath)['targets']

    return (
        content.get('query_based'),
        content.get('atlas_based'),
    )


@app.command()
@click.argument("cells-path")
@click.option("--atlas", help="Atlas URL / path", default=None, show_default=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("-t", "--targets", help="Path to target definition YAML file", default=None)
@click.option("--allow-empty", is_flag=True, help="Allow empty targets", show_default=True)
@click.option("-o", "--output", help="Path to output .target file", required=True)
def from_input(cells_path, atlas, atlas_cache, targets, allow_empty, output):
    """ Generate .target file from MVD3 or SONATA (and target definition YAML) """
    # pylint: disable=too-many-locals
    circuit = Circuit({'cells': cells_path})
    cells = circuit.cells.get()
    with open(output, 'w', encoding='utf-8') as f:
        write_default_targets(cells, f)
        if targets is None:
            if 'layer' in cells:
                bbp.write_property_targets(f, cells, 'layer', mapping=_layer_name)
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
                for name, dset in atlas_based.items():
                    mask = atlas.load_data(dset, cls=ROIMask).lookup(xyz)
                    bbp.write_target(f, name, cells.index[mask])


@app.command()
@click.argument("cells-path")
@click.option("--atlas", help="Atlas URL / path", default=None, show_default=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("-t", "--targets", help="Path to target definition YAML file", default=None)
@click.option("--allow-empty", is_flag=True, help="Allow empty targets", show_default=True)
@click.option("--population", help="Population name", default="default", show_default=True)
@click.option("-o", "--output", help="Path to output JSON file", required=True)
def node_sets(cells_path, atlas, atlas_cache, targets, allow_empty, population, output):
    """Generate JSON node sets from MVD3 or SONATA (and target definition YAML)"""
    # pylint: disable=too-many-locals

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

    cells = Circuit({'cells': cells_path}).cells

    result['All'] = {"population": population}
    _add_node_sets({
        'Excitatory': {'synapse_class': 'EXC'},
        'Inhibitory': {'synapse_class': 'INH'},
    })

    for prop in ['mtype', 'etype', 'region']:
        _add_node_sets({
            val: {prop: val} for val in cells.get(properties=prop).unique()
        })

    if targets is not None:
        query_based, atlas_based = _load_targets(targets)
        if query_based is not None:
            _add_node_sets(query_based)
        if atlas_based is not None:
            from voxcell.nexus.voxelbrain import Atlas
            if atlas is None:
                raise BrainBuilderError("Atlas not provided")
            atlas = Atlas.open(atlas, cache_dir=atlas_cache)
            xyz = cells.get(properties=['x', 'y', 'z'])
            for name, dset in atlas_based.items():
                mask = atlas.load_data(dset, cls=ROIMask).lookup(xyz.values)
                ids = xyz.index[mask] - 1
                result[name] = {"population": population, "node_id": ids.tolist()}

    dump_json(output, result)
