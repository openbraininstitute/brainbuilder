""" Tools for working with SONATA """

import click


@click.group()
def app():
    """ Tools for working with SONATA """


@app.command()
@click.argument("mvd3")
@click.option("--mecombo-info", help="Path to TSV file with ME-combo table", default=None)
@click.option("--population", help="Population name", default="default", show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def from_mvd3(mvd3, mecombo_info, population, output):
    """Convert MVD3 to SONATA nodes"""
    from brainbuilder.utils.sonata import write_nodes_from_mvd3
    write_nodes_from_mvd3(
        mvd3_path=mvd3,
        mecombo_info_path=mecombo_info,
        out_h5_path=output,
        population=population
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
@click.option("--base-dir", help="Base directory", required=True)
@click.option("--morph-dir", help="Morphologies directory", required=True)
@click.option("--nodes", help="Nodes H5 filepath(s)", required=True, multiple=True)
@click.option("--edges", help="Edges H5 filepath(s)", required=True, multiple=True)
@click.option("-o", "--output", help="Path to output file (JSON)", required=True)
def network_config(base_dir, morph_dir, nodes, edges, output):
    """Write SONATA network config"""
    from brainbuilder.utils.sonata import write_network_config
    write_network_config(
        base_dir=base_dir,
        morph_dir=morph_dir,
        node_files=nodes,
        edge_files=edges,
        output_path=output
    )
