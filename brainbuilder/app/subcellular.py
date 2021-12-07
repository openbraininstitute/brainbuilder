"""
Genes / proteins assignment.
"""

import click
import numpy as np

from voxcell import CellCollection


@click.group()
def app():
    """ Genes / proteins assignment """


@app.command()
@click.argument("cells-path", type=click.Path(exists=True, file_okay=True))
@click.option("--subcellular-dir", help="The directory to store the subcellular data")
@click.option("--transcriptome", help="Name of nexus transcriptome entity", required=True)
@click.option("--mtype-taxonomy", help="Name of nexus taxonomy entity", required=True)
@click.option("--cell-proteins", help="Name of nexus cell proteins concentration entity",
              required=True)
@click.option("--synapse-proteins", help="Name of cell proteins concentration entity",
              required=True)
@click.option("--seed", type=int, help="Pseudo-random generator seed", default=0, show_default=True)
@click.option("--output", type=str, required=True)
def assign(cells_path, subcellular_dir, transcriptome, mtype_taxonomy, cell_proteins,
           synapse_proteins, seed, output):
    """ Assign subcellular data """
    # pylint: disable=import-outside-toplevel
    from brainbuilder.subcellular import assign as _assign

    cells = CellCollection.load(cells_path)

    np.random.seed(seed)

    _assign(
        cells,
        subcellular_dir=subcellular_dir,
        transcriptome=transcriptome,
        mtype_taxonomy=mtype_taxonomy,
        cell_proteins=cell_proteins,
        synapse_proteins=synapse_proteins,
        output_path=output
    )
