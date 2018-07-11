import os
import tempfile
import shutil

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from nose.tools import eq_, raises
from mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_almost_equal
from pandas.util.testing import assert_frame_equal

from voxcell import CellCollection, Hierarchy, VoxelData

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import bbp


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_bind_profile1d_to_atlas():
    dist = VoxelData(np.array([[0.019, np.nan], [0.0, 1.0]]), voxel_dimensions=(10, 10))
    actual = bbp.bind_profile1d_to_atlas(np.arange(100), dist)
    assert_almost_equal(
        actual.raw,
        [[1, np.nan], [0, 99]]
    )


@raises(BrainBuilderError)
def test_bind_profile1d_to_atlas_raises_1():
    dist = VoxelData(np.array([0.5]), voxel_dimensions=(10,))
    bbp.bind_profile1d_to_atlas(np.arange(5), dist)


@raises(BrainBuilderError)
def test_bind_profile1d_to_atlas_raises_2():
    dist = VoxelData(np.array([1.1]), voxel_dimensions=(10,))
    bbp.bind_profile1d_to_atlas(np.arange(100), dist)


# @patch('voxcell.VoxelData.load_nrrd')
# @patch('numpy.loadtxt')
# def test_load_metype_composition(mock_loadtxt, mock_load_nrrd):
#     atlas = VoxelData(np.array([[2, 22, 404], [3, 0, 405]]), voxel_dimensions=(10,))
#     region_map = {'L2': (2, 22), 'L3': (3,), '404': (404,)}

#     relative_distance = VoxelData(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), voxel_dimensions=(10,))

#     MC_density_1d = np.arange(100)
#     mock_loadtxt.return_value = MC_density_1d

#     MC_density_3d = bbp.bind_profile1d_to_atlas(MC_density_1d, relative_distance)
#     mock_load_nrrd.return_value = MC_density_3d

#     total_density, sdist, etypes = bbp.load_metype_composition(
#         os.path.join(DATA_PATH, 'metype_composition.yaml'), atlas, region_map,
#         relative_distance=relative_distance
#     )

#     assert_almost_equal(total_density.raw, [[42 + 0.1 * 10, 42 + 0.1 * 20, 0], [40, 0, 0]])

#     assert_equal(sdist.field.raw, [[1, 2, -1], [0, -1, -1]])
#     assert_frame_equal(
#         sdist.traits,
#         pd.DataFrame([
#             ['L2', 'L23_MC'],
#             ['L2', 'L2_IPC'],
#             ['L3', 'L23_MC'],
#         ], columns=['region', 'mtype'])
#     )
#     assert_frame_equal(
#         sdist.distributions,
#         pd.DataFrame([
#             [0.0, 0.0232558, 0.0454545],
#             [0.0, 0.9767442, 0.9545455],
#             [1.0, 0.0000000, 0.0000000],
#         ]),
#         check_dtype=False
#     )

#     assert_equal(
#         etypes, {
#             ('L2', 'L23_MC'): {'cNAC': 0.6667, 'bNAC': 0.3333},
#             ('L2', 'L2_IPC'): {'cADpyr': 1.0},
#             ('L3', 'L23_MC'): {'cNAC': 0.5, 'bNAC': 0.5},
#         }
#     )


def test_load_neurondb_v3():
    actual = bbp.load_neurondb_v3(os.path.join(DATA_PATH, 'neuronDBv3.dat'))
    expected = pd.DataFrame({
        'morphology': ["morph-a", "morph-b"],
        'layer': ["L1", "L2"],
        'mtype': ["L1_DAC", "L23_PC"],
        'etype': ["bNAC", "dNAC"],
        'me_combo': ["me-combo-a", "me-combo-b"],
    })
    assert_frame_equal(actual, expected, check_like=True)


def test_load_neurondb_v4_as_v3():
    actual = bbp.load_neurondb_v3(os.path.join(DATA_PATH, 'neuronDBv4.dat'))
    expected = pd.DataFrame({
        'morphology': ["morph-a", "morph-b"],
        'layer': [1, 2],
        'mtype': ["L1_DAC", "L23_PC"],
        'etype': ["bNAC", "dNAC"],
        'me_combo': ["me-combo-a", "me-combo-b"],
    })
    assert_frame_equal(actual, expected, check_like=True)


def test_parse_mvd2():
    data = bbp.parse_mvd2(os.path.join(DATA_PATH, 'circuit.mvd2'))
    eq_(len(data['CircuitSeeds']), 1)
    eq_(len(data['ElectroTypes']), 3)
    eq_(len(data['MorphTypes']), 5)
    eq_(len(data['MicroBox Data']), 1)
    eq_(len(data['MiniColumnsPosition']), 5)
    eq_(len(data['Neurons Loaded']), 5)


def test_load_mvd2():
    cells = bbp.load_mvd2(os.path.join(DATA_PATH, 'circuit.mvd2'))

    eq_(cells.positions.shape, (5, 3))

    eq_(cells.orientations.shape, (5, 3, 3))

    assert_almost_equal(
        cells.orientations[0],
        [[ 0.5639686,  0.       , -0.8257962],
         [ 0.       ,  1.       ,  0.       ],
         [ 0.8257962,  0.       ,  0.5639686]],
    )

    eq_(set(cells.properties.columns),
        set(['etype', 'morphology', 'mtype', 'synapse_class', 'morph_class',
             'layer', 'hypercolumn', 'minicolumn', 'me_combo']))

    eq_(list(cells.properties.synapse_class.unique()),
        ['INH', 'EXC'])

    eq_(list(cells.properties.mtype.unique()),
        ['L1_DLAC', 'L23_PC', 'L4_NBC', 'L5_TTPC1', 'L6_LBC'])

    eq_(list(cells.properties.etype.unique()),
        ['cNAC', 'cADpyr', 'dNAC'])

    eq_(list(cells.properties.synapse_class),
        ['INH', 'EXC', 'INH', 'EXC', 'INH'])

    eq_(list(cells.properties.layer),
        ['1', '2', '4', '5', '6'])

    eq_(cells.properties.me_combo[0],
        'cNAC187_L1_DLAC_1_sm080904a3_-_Scale_x1.000_y1.050_z1.000')


def test_roundtrip_mvd2():
    original = bbp.load_mvd2(os.path.join(DATA_PATH, 'circuit.mvd2'))

    cwd = tempfile.mkdtemp()

    try:
        filename = os.path.join(cwd, 'exported.mvd2')
        bbp.save_mvd2(filename, '/here', original)
        restored = bbp.load_mvd2(filename)
        assert_almost_equal(original.orientations, restored.orientations)
        assert_almost_equal(original.positions, restored.positions)
        assert_frame_equal(original.properties, restored.properties)
    finally:
        shutil.rmtree(cwd)


def test_gid2str():
    actual = bbp.gid2str(42)
    eq_(actual, "a42")


def test_write_target():
    out = StringIO()
    bbp.write_target(out, "test", gids=[1, 2], include_targets=["A", "B"])
    actual = out.getvalue()
    expected = "\n".join([
        "",
        "Target Cell test",
        "{",
        "  a1 a2",
        "  A B",
        "}",
        ""
    ])
    eq_(actual, expected)


def test_write_property_targets():
    cells = pd.DataFrame({
            'prop-a': ['A', 'B', 'A'],
            'prop-b': ['X', 'X', 'Y']
        },
        index=[1, 2, 3]
    )
    out = StringIO()
    bbp.write_property_targets(out, cells, 'prop-a')
    bbp.write_property_targets(out, cells, 'prop-b', mapping=lambda x: "z" + x)
    actual = out.getvalue()
    expected = "\n".join([
        "",
        "Target Cell A",
        "{",
        "  a1 a3",
        "}",
        "",
        "Target Cell B",
        "{",
        "  a2",
        "}",
        "",
        "Target Cell zX",
        "{",
        "  a1 a2",
        "}",
        "",
        "Target Cell zY",
        "{",
        "  a3",
        "}",
        ""
    ])
    eq_(actual, expected)


def test_assign_emodels():
    cells = CellCollection()
    cells.properties = pd.DataFrame([
        ('morph-A', 'layer-A', 'mtype-A', 'etype-A', 'prop-A'),
        ('morph-B', 'layer-B', 'mtype-B', 'etype-B', 'prop-B'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'prop'])
    morphdb = pd.DataFrame([
        ('morph-A', 'layer-A', 'mtype-A', 'etype-A', 'me_combo-A'),
        ('morph-B', 'layer-B', 'mtype-B', 'etype-B', 'me_combo-B'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'me_combo'])
    actual = bbp.assign_emodels(cells, morphdb).properties
    expected = pd.DataFrame([
        ('morph-A', 'layer-A', 'mtype-A', 'etype-A', 'prop-A', 'me_combo-A'),
        ('morph-B', 'layer-B', 'mtype-B', 'etype-B', 'prop-B', 'me_combo-B'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'prop', 'me_combo'])
    assert_frame_equal(actual, expected, check_like=True)


def test_assign_emodels_multiple_choice():
    np.random.seed(0)
    cells = CellCollection()
    cells.properties = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A', 'prop-A'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'prop'])
    morphdb = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A', 'me_combo-A1'),
        ('morph-A', 1, 'mtype-A', 'etype-A', 'me_combo-A2'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'me_combo'])
    actual = bbp.assign_emodels(cells, morphdb).properties
    expected = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A', 'prop-A', 'me_combo-A1'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'prop', 'me_combo'])
    assert_frame_equal(actual, expected, check_like=True)


def test_assign_emodels_overwrite():
    cells = CellCollection()
    cells.properties = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A', 'prop-A', 'me_combo-A0'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'prop', 'me_combo'])
    morphdb = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A', 'me_combo-A1'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'me_combo'])
    actual = bbp.assign_emodels(cells, morphdb).properties
    expected = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A', 'prop-A', 'me_combo-A1'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'prop', 'me_combo'])
    assert_frame_equal(actual, expected, check_like=True)


@raises(BrainBuilderError)
def test_assign_emodels_raises():
    cells = CellCollection()
    cells.properties = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A', 'prop-A'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'prop'])
    morphdb = pd.DataFrame([
        ('morph-A', 1, 'mtype-A', 'etype-A1', 'me_combo-A1'),
    ], columns=['morphology', 'layer', 'mtype', 'etype', 'me_combo'])
    actual = bbp.assign_emodels(cells, morphdb)
