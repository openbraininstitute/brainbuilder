""" Tools for working with SYN2 """

import click


@click.group()
def app():
    """ Tools for working with SYN2 """
    pass


@app.command()
@click.argument("syn2")
@click.option("--source", help="Source node population name", default="default", show_default=True)
@click.option("--target", help="Target node population name", default="default", show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def to_sonata(syn2, source, target, output):
    """ Convert to SONATA """
    from brainbuilder.utils.sonata import write_edges_from_syn2
    write_edges_from_syn2(
        syn2_path=syn2,
        source=source,
        target=target,
        out_h5_path=output
    )
