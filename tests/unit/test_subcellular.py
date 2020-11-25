import subprocess
import shutil
import tempfile
from contextlib import contextmanager

import pandas as pd
import numpy as np
import nose.tools as nt

from mock import patch, Mock
from pathlib import Path

import brainbuilder.subcellular as sub


def assert_h5_equal(ref, testee):
    cmd = ['/usr/bin/h5diff', ref, testee]
    nt.assert_equal(subprocess.call(cmd), 0)


DATA_PATH = Path(Path(__file__).parent, 'data/subcellular')


@contextmanager
def setup_tempdir(prefix, cleanup=True):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir)


def path(file):
    return str(Path(DATA_PATH, file))


def _get_entity_getter_side_effect():
    return [path('transcriptome.csv'), path('mtype_taxonomy.tsv'),
            path('cells.tsv'), path('synapses.tsv')]


def _get_mvd3_content():
    return pd.read_csv(path('mvd3_content.csv'))


def get_cells_mock():
    cells = Mock()
    cells.as_dataframe.return_value = _get_mvd3_content()
    return cells


class TestSubcellular:

    def setUp(self):
        np.random.seed(42)
        self.cells = _get_mvd3_content()

    @staticmethod
    def _get_library(my_dir):
        return {'gene_expressions': str(Path(my_dir, 'gene_expressions.h5')),
                'gene_mapping': str(Path(my_dir, 'gene_mapping.h5')),
                'cell_proteins': str(Path(my_dir, 'cell_proteins.h5')),
                'synapse_proteins': str(Path(my_dir, 'synapse_proteins.h5')),
                }

    @patch('uuid.uuid4', return_value='dummy')
    def test__create_all(self, uuid_name):
        with setup_tempdir('tmp') as my_dir:
            testee = sub._create_all(path('transcriptome.csv'),
                                     path('mtype_taxonomy.tsv'),
                                     path('cells.tsv'),
                                     path('synapses.tsv'),
                                     my_dir)

            ref = self._get_library(my_dir)

            nt.assert_dict_equal(testee, ref)
            assert_h5_equal(path('gene_expressions.h5'), testee['gene_expressions'])
            assert_h5_equal(path('gene_mapping.h5'), testee['gene_mapping'])
            assert_h5_equal(path('cell_proteins.h5'), testee['cell_proteins'])
            assert_h5_equal(path('synapse_proteins.h5'), testee['synapse_proteins'])

    def test__assign_gene_expressions(self):
        library = self._get_library(DATA_PATH)
        testee = sub._assign_gene_expressions(self.cells, library['gene_expressions'])
        ref = ['a00000', 'a00002', 'a00001', 'a00000', 'a00002',
               'a00002', 'a00001', 'a00002', 'a00001', 'a00000']
        nt.assert_list_equal(list(testee), ref)

    def test__assign_cell_proteins(self):
        library = self._get_library(DATA_PATH)
        testee = sub._assign_cell_proteins(self.cells, library['cell_proteins'])
        ref = ['dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy',
               'dummy', 'dummy', 'dummy', 'dummy']
        nt.assert_list_equal(list(testee), ref)

    def test__assign_synapse_proteins(self):
        library = self._get_library(DATA_PATH)
        testee = sub._assign_synapse_proteins(self.cells, library['synapse_proteins'])
        ref = ['dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy',
               'dummy', 'dummy', 'dummy', 'dummy']
        nt.assert_list_equal(list(testee), ref)

    @patch('uuid.uuid4', return_value='dummy')
    @patch('subcellular_querier.EntityGetter.get_entity_attachment',
           side_effect=_get_entity_getter_side_effect())
    def test_assign(self, uuid_name, attachment):
        with setup_tempdir('tmp', cleanup=False) as my_dir:
            cells = get_cells_mock()
            output = str(Path(my_dir, 'subcellular.h5'))
            sub.assign(cells, str(Path(my_dir, 'test')), 'dummy', 'dummy', 'dummy',
                       'dummy', str(output))
            assert_h5_equal(path('subcellular.h5'), output)
