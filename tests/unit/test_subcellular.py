import subprocess
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import numpy as np

from mock import patch, Mock
import pytest

import brainbuilder.subcellular as test_module


def assert_h5_equal(ref, testee):
    cmd = ['h5diff', ref, testee]
    assert subprocess.call(cmd) == 0


DATA_PATH = Path(__file__).parent / 'data/subcellular'


@contextmanager
def setup_tempdir(prefix, cleanup=True):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir)


def _get_entity_getter_side_effect():
    return [DATA_PATH / 'transcriptome.csv', DATA_PATH / 'mtype_taxonomy.tsv',
            DATA_PATH / 'cells.tsv', DATA_PATH / 'synapses.tsv']


def _get_mvd3_content():
    return pd.read_csv(DATA_PATH / 'mvd3_content.csv')


def get_cells_mock():
    cells = Mock()
    cells.as_dataframe.return_value = _get_mvd3_content()
    return cells


@pytest.fixture()
def cells():
    np.random.seed(42)
    return _get_mvd3_content()


def _get_library(my_dir):
    return {'gene_expressions': str(Path(my_dir, 'gene_expressions.h5')),
            'gene_mapping': str(Path(my_dir, 'gene_mapping.h5')),
            'cell_proteins': str(Path(my_dir, 'cell_proteins.h5')),
            'synapse_proteins': str(Path(my_dir, 'synapse_proteins.h5')),
            }


@patch('uuid.uuid4', return_value='dummy')
def test__create_all(uuid4):
    with setup_tempdir('tmp') as my_dir:
        testee = test_module._create_all(DATA_PATH / 'transcriptome.csv',
                                 DATA_PATH / 'mtype_taxonomy.tsv',
                                 DATA_PATH / 'cells.tsv',
                                 DATA_PATH / 'synapses.tsv',
                                 my_dir)

        ref = _get_library(my_dir)

        assert testee == ref
        assert_h5_equal(DATA_PATH / 'gene_expressions.h5', testee['gene_expressions'])
        assert_h5_equal(DATA_PATH / 'gene_mapping.h5', testee['gene_mapping'])
        assert_h5_equal(DATA_PATH / 'cell_proteins.h5', testee['cell_proteins'])
        assert_h5_equal(DATA_PATH / 'synapse_proteins.h5', testee['synapse_proteins'])


def test__assign_gene_expressions(cells):
    library = _get_library(DATA_PATH)
    testee = test_module._assign_gene_expressions(cells, library['gene_expressions'])
    ref = ['a00000', 'a00002', 'a00001', 'a00000', 'a00002',
           'a00002', 'a00001', 'a00002', 'a00001', 'a00000']
    assert list(testee) == ref


def test__assign_cell_proteins(cells):
    library = _get_library(DATA_PATH)
    testee = test_module._assign_cell_proteins(cells, library['cell_proteins'])
    ref = ['dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy',
           'dummy', 'dummy', 'dummy', 'dummy']
    assert list(testee) == ref


def test__assign_synapse_proteins(cells):
    library = _get_library(DATA_PATH)
    testee = test_module._assign_synapse_proteins(cells, library['synapse_proteins'])
    ref = ['dummy', 'dummy', 'dummy', 'dummy', 'dummy', 'dummy',
           'dummy', 'dummy', 'dummy', 'dummy']
    assert list(testee) == ref


@patch('uuid.uuid4', return_value='dummy')
@patch('subcellular_querier.EntityGetter.get_entity_attachment',
       side_effect=_get_entity_getter_side_effect())
def test_assign(uuid4, get_entity_attachment):
    with setup_tempdir('tmp', cleanup=False) as my_dir:
        cells = get_cells_mock()
        output = str(Path(my_dir, 'subcellular.h5'))
        test_module.assign(cells, str(Path(my_dir, 'test')), 'dummy', 'dummy', 'dummy',
                   'dummy', str(output))
        assert_h5_equal(DATA_PATH / 'subcellular.h5', output)
