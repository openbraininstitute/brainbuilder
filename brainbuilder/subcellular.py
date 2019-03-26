"""
Genes / proteins assignment.
"""

import logging
import warnings

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tables
import six

from subcellular_querier import (EntityGetter, create_gene_expressions,
                                 create_cell_proteins, create_synapse_proteins)
from entity_management.synprot import (CellProteinConcExperiment, SynapticProteinConcExperiment,
                                       TranscriptomeExperiment, MtypeTaxonomy)

L = logging.getLogger('brainbuilder')


class CellInfo(tables.IsDescription):
    """ '/cells' table schema for HDF5 file with subcellular info. """
    gid = tables.Int32Col(pos=0)
    gene_expressions = tables.StringCol(64, pos=1)
    cell_proteins = tables.StringCol(64, pos=2)
    synapse_proteins = tables.StringCol(64, pos=3)


def _create_all(transcriptome, mtype_taxonomy, cell_tsv, synapse_tsv, dir_output):
    gene_expressions = Path(dir_output, 'gene_expressions.h5')
    cell_proteins = Path(dir_output, 'cell_proteins.h5')
    gene_mapping = Path(dir_output, 'gene_mapping.h5')
    synapse_proteins = Path(dir_output, 'synapse_proteins.h5')

    create_gene_expressions(transcriptome, mtype_taxonomy, gene_expressions)
    create_cell_proteins(cell_tsv, gene_expressions, cell_proteins, gene_mapping)
    create_synapse_proteins(synapse_tsv, gene_expressions, synapse_proteins)

    library = {
        'gene_expressions': gene_expressions,
        'gene_mapping': gene_mapping,
        'cell_proteins': cell_proteins,
        'synapse_proteins': synapse_proteins,
    }

    return library


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


def assign(cells, subcellular_dir, transcriptome, mtype_taxonomy, cell_proteins,
           synapse_proteins, output_path):
    # pylint: disable=too-many-locals
    """ Assign subcellular data """
    cells = cells.as_dataframe()

    subcellular_path = Path(subcellular_dir)
    if not subcellular_path.exists():
        subcellular_path.mkdir()

    L.info("Start retrieving data from nexus ...")
    eg = EntityGetter()
    transcriptome_path = eg.get_entity_attachment(TranscriptomeExperiment, transcriptome,
                                                  subcellular_path)

    mtype_taxonomy_path = eg.get_entity_attachment(MtypeTaxonomy, mtype_taxonomy,
                                                   subcellular_path)

    cell_proteins_path = eg.get_entity_attachment(CellProteinConcExperiment, cell_proteins,
                                                  subcellular_path)

    synapse_proteins_path = eg.get_entity_attachment(SynapticProteinConcExperiment,
                                                     synapse_proteins, subcellular_path)

    L.info("Start creating subcellular h5 file ...")
    library = _create_all(transcriptome_path, mtype_taxonomy_path, cell_proteins_path,
                          synapse_proteins_path, subcellular_dir)

    warnings.simplefilter('ignore', tables.NaturalNameWarning)
    L.info('Creating %s file', str(Path(output_path).resolve()))
    with tables.open_file(output_path, 'w') as out:
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
