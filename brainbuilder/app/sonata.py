""" Tools for working with SONATA """

import json
import os

import click
from brainbuilder.app._utils import REQUIRED_PATH_DIR, REQUIRED_PATH


@click.group()
def app():
    """ Tools for working with SONATA """


@app.command()
@click.argument("mvd3")
@click.option("--mecombo-info", help="Path to TSV file with ME-combo table", default=None)
@click.option("--population", help="Population name", default="default", show_default=True)
@click.option("--library-properties", required=False, default=None,
              help="Comma-separated properties names that should become library enumerations")
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def from_mvd3(mvd3, mecombo_info, population, library_properties, output):
    """Convert MVD3 to SONATA nodes"""
    from brainbuilder.utils.sonata import write_nodes_from_mvd3

    if library_properties is not None:
        library_properties = [p.strip() for p in library_properties.split(',')]

    write_nodes_from_mvd3(
        mvd3_path=mvd3,
        mecombo_info_path=mecombo_info,
        out_h5_path=output,
        population=population,
        library_properties=library_properties
    )


@app.command()
@click.argument("syn2")
@click.option("--population", help="Population name", default="default", show_default=True)
@click.option("--source", help="Source node population name", default="default", show_default=True)
@click.option("--target", help="Target node population name", default="default", show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def from_syn2(syn2, population, source, target, output):
    """Convert SYN2 to SONATA edges"""
    from brainbuilder.utils.sonata import write_edges_from_syn2
    write_edges_from_syn2(
        syn2_path=syn2,
        population=population,
        source=source,
        target=target,
        out_h5_path=output
    )


@app.command()
@click.option("--base-dir", help="Path to base directory", required=True)
@click.option("--morph-dir", help="Morphologies directory (relative to BASE_DIR)", required=True)
@click.option(
    "--emodel-dir", help="Cell electrical models directory (relative to BASE_DIR)", required=True)
@click.option("--nodes-dir", help="Node files directory (relative to BASE_DIR)", required=True)
@click.option("--nodes", help="Node population(s) (';'-separated)", required=True)
@click.option("--node-sets", help="Node sets file (JSON)", required=True)
@click.option("--edges-dir", help="Edge files directory (relative to BASE_DIR)", required=True)
@click.option("--edges-suffix", help="Edge file suffix", default="")
@click.option("--edges", help="Edge population(s) (';'-separated)", required=True)
@click.option("-o", "--output", help="Path to output file (JSON)", required=True)
def network_config(
    base_dir, morph_dir, emodel_dir, nodes_dir, nodes,
    node_sets, edges_dir, edges_suffix, edges, output
):
    """Write SONATA network config"""
    # pylint: disable=too-many-arguments
    from brainbuilder.utils.sonata import write_network_config
    write_network_config(
        base_dir=base_dir,
        morph_dir=morph_dir,
        emodel_dir=emodel_dir,
        nodes_dir=nodes_dir,
        nodes=nodes.split(";"),
        node_sets=node_sets,
        edges_dir=edges_dir,
        edges_suffix=edges_suffix,
        edges=edges.split(";"),
        output_path=output
    )


@app.command()
@click.option("--input-dir", help="Path to the input directory containing the targets files",
              required=True)
@click.option("-o", "--output", help="Path to output the .json file", required=True)
def node_set_from_targets(input_dir, output):
    """Convert target files into a single node_set.json like file.

    Please check 'brainbuilder targets node-sets' also.
    """
    from brainbuilder.utils.sonata import write_node_set_from_targets
    write_node_set_from_targets(input_dir, output)


@app.command()
@click.option("--h5-morphs", required=True, type=REQUIRED_PATH_DIR,
              help="Path to h5 morphology directory")
@click.option("-o", "--output", required=True,
              help="Path to output directory for HDF5 morphologies")
def update_morphologies(h5_morphs, output):
    """Update h5 morphologies"""
    from brainbuilder.utils import sonata_reindex
    assert not os.path.exists(output), 'output directory must not already exist'

    h5_updates = sonata_reindex.generate_h5_updates(h5_morphs)

    sonata_reindex.write_new_h5_morphs(h5_updates, h5_morphs, output)

    h5_updates_path = os.path.join(output, 'h5_updates.json')
    with open(h5_updates_path, 'w') as fd:
        json.dump(h5_updates, fd, indent=2)

    click.echo('h5_updates output to %s' % h5_updates_path)


@app.command()
@click.option("--h5-updates", required=True, type=REQUIRED_PATH,
              help="h5_updates.json produced by update_morphologies")
@click.option("--nodes", required=True, type=REQUIRED_PATH,
              help="Node file")
@click.option("--population", default="default", show_default=True,
              help="Population name")
@click.argument("edges", nargs=-1, required=True)
def update_edge_population(h5_updates, nodes, population, edges):
    '''Given h5_updates from removing single children, update synapses'''
    from brainbuilder.utils import sonata_reindex
    from voxcell.sonata import NodePopulation

    with open(h5_updates, 'r') as fd:
        h5_updates = json.load(fd)

    for v in h5_updates.values():
        v['new_parents'] = [int(k) for k in v['new_parents']]
        v['new_segment_offset'] = {int(k): vv for k, vv in v['new_segment_offset'].items()}

    morphologies = NodePopulation.load(nodes).to_dataframe()['morphology']
    for edge in edges:
        sonata_reindex.apply_edge_updates(morphologies, edge, h5_updates, population)


@app.command()
@click.option("--morph-path", required=True, type=REQUIRED_PATH_DIR,
              help="path to h5 morphology files")
@click.option("--population", default="default", show_default=True,
              help="Population name")
@click.option("--nodes", required=True, type=REQUIRED_PATH,
              help="Node file")
@click.argument("edges", nargs=-1, required=True)
def update_edge_pos(morph_path, population, nodes, edges):
    '''Using: section_id, segment_id and offset, create the sonata *_section_pos'''
    from brainbuilder.utils import sonata_reindex
    from voxcell.sonata import NodePopulation

    morphologies = NodePopulation.load(nodes).to_dataframe()['morphology']
    sonata_reindex.write_sonata_pos(morph_path, morphologies, population, edges)
