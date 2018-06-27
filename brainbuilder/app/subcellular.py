"""
Genes / proteins assignment.
"""

import logging
import warnings

from collections import defaultdict

import click
import numpy as np
import pandas as pd
import tables
import six

from voxcell import CellCollection

L = logging.getLogger('brainbuilder')


@click.group()
def app():
    """ Genes / proteins assignment """
    pass


class CellInfo(tables.IsDescription):
    """ '/cells' table schema for HDF5 file with subcellular info. """
    gid = tables.Int32Col(pos=0)
    gene_expressions = tables.StringCol(64, pos=1)
    cell_proteins = tables.StringCol(64, pos=2)
    synapse_proteins = tables.StringCol(64, pos=3)


def _assign_gene_expressions(cells, library):
    L.info("Assigning gene expressions...")

    mapping = defaultdict(list)
    with tables.open_file(library, 'r') as h5f:
        for t in h5f.list_nodes('/gene_expressions'):
            for mtype in t.attrs['mtype'].split("|"):
                mapping[mtype].append(t.name)

    result = pd.Series(index=cells.index, dtype='S')
    for mtype, group in cells.groupby('mtype'):
        result[group.index] = np.random.choice(mapping[mtype], size=len(group))

    return result.values


def _assign_cell_proteins(cells, library):
    L.info("Assigning cell proteins...")
    with tables.open_file(library, 'r') as h5f:
        pool = [t.name for t in h5f.list_nodes('/cell_proteins')]
    return np.random.choice(pool, size=len(cells))


def _assign_synapse_proteins(cells, library):
    L.info("Assigning synapse proteins...")
    with tables.open_file(library, 'r') as h5f:
        pool = [t.name for t in h5f.list_nodes('/synapse_proteins')]
    return np.random.choice(pool, size=len(cells))


@app.command()
@click.argument("mvd3")
@click.option(
    "--gene-expressions", help="Path to HDF5 with gene expressions", required=True)
@click.option(
    "--gene-mapping", help="Path to HDF5 with gene/protein correspondence", required=True)
@click.option(
    "--cell-proteins", help="Path to HDF5 with cell protein concentrations", required=True)
@click.option(
    "--synapse-proteins", help="Path to HDF5 synapse protein concentrations", required=True)
@click.option(
    "--seed", type=int, help="Pseudo-random generator seed", default=0, show_default=True)
@click.option(
    "-o", "--output", help="Path to output subcellular file", required=True)
def assign(mvd3, gene_expressions, gene_mapping, cell_proteins, synapse_proteins, seed, output):
    """ Assign subcellular data """
    # pylint: disable=too-many-locals

    np.random.seed(seed)

    cells = CellCollection.load_mvd3(mvd3).as_dataframe()

    library = {
        'gene_expressions': gene_expressions,
        'gene_mapping': gene_mapping,
        'cell_proteins': cell_proteins,
        'synapse_proteins': synapse_proteins,
    }

    warnings.simplefilter('ignore', tables.NaturalNameWarning)
    with tables.open_file(output, 'w') as out:
        table = out.create_table('/', 'cells', CellInfo)
        table.append(list(zip(
            cells.index.values,
            _assign_gene_expressions(cells, library['gene_expressions']),
            _assign_cell_proteins(cells, library['cell_proteins']),
            _assign_synapse_proteins(cells, library['synapse_proteins']),
        )))
        out_lib = out.create_group('/', 'library')
        for title, src_file in six.iteritems(library):
            node = "/" + title
            L.info("Copying %s to /library...", title)
            with tables.open_file(src_file, 'r') as src:
                src.copy_node(node, out_lib, recursive=True)
