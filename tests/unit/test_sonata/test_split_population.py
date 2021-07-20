import json
import os.path
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from brainbuilder.utils.sonata import split_population

import utils


DATA_PATH = (Path(__file__).parent / '../data/sonata/split_population/').resolve()


def test__get_population_name():
    assert 'src__dst__chemical' == split_population._get_population_name(src='src', dst='dst')
    assert 'src' == split_population._get_population_name(src='src', dst='src')


def test__write_nodes():
    split_nodes = {'A': pd.DataFrame({'fake_prop': range(10), },
                                      index=np.arange(10)),
                   'B': pd.DataFrame({'fake_prop': range(5), },
                                     index=np.arange(10, 15)),
                   }
    with utils.tempdir('test__write_nodes') as tmp:
        split_population._write_nodes(tmp, split_nodes)
        assert (Path(tmp) / 'nodes_A.h5').exists()
        assert (Path(tmp) / 'nodes_B.h5').exists()

        with h5py.File((Path(tmp) / 'nodes_A.h5'), 'r') as h5:
            assert_array_equal(h5['/nodes/A/0/fake_prop'], np.arange(10))
            assert_array_equal(h5['/nodes/A/node_type_id'], np.full(10, -1))
        with h5py.File((Path(tmp) / 'nodes_B.h5'), 'r') as h5:
            assert_array_equal(h5['/nodes/B/0/fake_prop'], np.arange(5))
            assert_array_equal(h5['/nodes/B/node_type_id'], np.full(5, -1))


def test__get_node_id_mapping():
    split_nodes = {'A': pd.DataFrame(index=np.arange(0, 10)),
                   'B': pd.DataFrame(index=np.arange(10, 15)),
                   }
    ret = split_population._get_node_id_mapping(split_nodes)
    assert len(ret) == 2
    assert ret['A'].new_id.to_list() == list(range(10))
    assert ret['B'].new_id.to_list() == list(range(5))


def test__split_population_by_attribute():
    # nodes.h5 contains 3 nodes with mtypes "L2_X", "L6_Y", "L6_Y"
    nodes_path = DATA_PATH / 'nodes.h5'
    ret = split_population._split_population_by_attribute(nodes_path, 'mtype')
    assert len(ret) == 2
    assert isinstance(ret['L2_X'], pd.DataFrame)

    assert len(ret['L2_X']) == 1
    assert ret['L2_X'].mtype.unique()[0] == 'L2_X'
    assert_array_equal(ret['L2_X'].index, [0])

    assert len(ret['L6_Y']) == 2
    assert ret['L6_Y'].mtype.unique()[0] == 'L6_Y'
    assert_array_equal(ret['L6_Y'].index, [1, 2])


def test__write_circuit_config():
    split_nodes = {'A': pd.DataFrame(index=np.arange(0, 10)),
                   'B': pd.DataFrame(index=np.arange(10, 15)),
                   }
    with utils.tempdir('test__write_circuit_config') as tmp:
        split_population._write_circuit_config(tmp, split_nodes)
        with open(os.path.join(tmp, 'circuit_config.json'), 'r') as fd:
            ret = json.load(fd)
            assert 'manifest' in ret
            assert 'networks' in ret
            assert 'nodes' in ret['networks']
            assert 'nodes' in ret['networks']
            assert len(ret['networks']['edges']) == 0  # no edge files

        open(os.path.join(tmp, 'edges_A.h5'), 'w').close()
        open(os.path.join(tmp, 'edges_B.h5'), 'w').close()
        open(os.path.join(tmp, 'edges_A__B__chemical.h5'), 'w').close()
        split_population._write_circuit_config(tmp, split_nodes)
        with open(os.path.join(tmp, 'circuit_config.json'), 'r') as fd:
            ret = json.load(fd)
            assert len(ret['networks']['edges']) == 3


def test__write_edges():
    # edges.h5 contains the following edges:
    # '/edges/default/source_node_id': [2, 0, 0, 2]
    # '/edges/default/target_node_id': [0, 1, 1, 1]
    edges_path = DATA_PATH / 'edges.h5'
    # iterate over different id_mappings to split the edges in different ways
    for id_mapping, h5_read_chunk_size, expected_dir in [
        (
            {
                # edges: A -> B (2), B -> A, B -> B
                'A': pd.DataFrame({'new_id': np.arange(4)}, index=[5, 4, 3, 0]),
                'B': pd.DataFrame({'new_id': np.arange(2)}, index=[1, 2]),
            },
            10,
            DATA_PATH / '01',
        ),
        (
            {
                # edges: A -> A (4)
                'A': pd.DataFrame({'new_id': np.arange(4)}, index=[3, 2, 1, 0]),
                'B': pd.DataFrame({'new_id': np.arange(2)}, index=[5, 4]),
            },
            10,
            DATA_PATH / '02',
        ),
        (
            {
                # edges: B -> B (4), reduced chunk size
                'A': pd.DataFrame({'new_id': np.arange(3)}, index=[5, 4, 3]),
                'B': pd.DataFrame({'new_id': np.arange(3)}, index=[2, 1, 0]),
            },
            3,
            DATA_PATH / '03',
        ),
        (
            {
                # edges: A -> A, A -> B (3)
                'A': pd.DataFrame({'new_id': np.arange(4)}, index=[2, 0, 4, 5]),
                'B': pd.DataFrame({'new_id': np.arange(2)}, index=[1, 3]),
            },
            10,
            DATA_PATH / '04',
        ),
        (
            {
                # edges: B -> B, B -> A (3)
                'A': pd.DataFrame({'new_id': np.arange(4)}, index=[1, 3, 4, 5]),
                'B': pd.DataFrame({'new_id': np.arange(2)}, index=[2, 0]),
            },
            10,
            DATA_PATH / '05',
        ),
    ]:
        with utils.tempdir('test__write_edges') as tmp:
            split_population._write_edges(tmp, edges_path, id_mapping, h5_read_chunk_size)
            utils.assert_h5_dirs_equal(tmp, expected_dir, pattern='edges_*.h5')


def test_split_population():
    attribute = 'mtype'
    nodes_path = DATA_PATH / 'nodes.h5'
    edges_path = DATA_PATH / 'edges.h5'
    expected_dir = DATA_PATH / '00'
    with utils.tempdir('test_split_population') as tmp:
        split_population.split_population(tmp, attribute, nodes_path, edges_path)
        utils.assert_h5_dirs_equal(tmp, expected_dir)
        utils.assert_json_files_equal(
            Path(tmp, 'circuit_config.json'), Path(expected_dir, 'circuit_config.json')
        )
