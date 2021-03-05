import os

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from nose.tools import eq_, raises, assert_raises, assert_list_equal

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from voxcell import CellCollection

from brainbuilder.exceptions import BrainBuilderError
from brainbuilder.utils import bbp


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_load_cell_composition_v1():
    with assert_raises(ValueError) as e:
        bbp.load_cell_composition(os.path.join(DATA_PATH, 'cell_composition_v1.yaml'))
    assert 'Use cell composition file of version 2' in e.exception.args[0]


def test_load_cell_composition_v2():
    content = bbp.load_cell_composition(os.path.join(DATA_PATH, 'cell_composition_v2.yaml'))
    assert content == {'version': 'v2.0', 'neurons': [{'density': 68750, 'region': 'mc0;Rt',
                                                       'traits': {'layer': 'Rt', 'mtype': 'Rt_RC',
                                                                  'etype': {'cNAD_noscltb': 0.43,
                                                                            'cAD_noscltb': 0.57}}},
                                                      {'density': '{L23_MC}', 'region': 'mc1;Rt',
                                                       'traits': {'layer': 'Rt', 'mtype': 'Rt_RC',
                                                                  'etype': {'cNAD_noscltb': 0.43,
                                                                            'cAD_noscltb': 0.57}}}]}


def test_load_neurondb_v2():
    actual = bbp.load_neurondb(os.path.join(DATA_PATH, 'neuronDBv2.dat'))
    expected = pd.DataFrame({
        'morphology': ["morph-a", "morph-b"],
        'layer': ["L1", "L2"],
        'mtype': ["L1_DAC", "L23_PC"],
    })
    assert_frame_equal(actual, expected, check_like=True)


def test_load_neurondb_v3_as_v2():
    actual = bbp.load_neurondb(os.path.join(DATA_PATH, 'neuronDBv3.dat'))
    expected = pd.DataFrame({
        'morphology': ["morph-a", "morph-b"],
        'layer': ["L1", "L2"],
        'mtype': ["L1_DAC", "L23_PC"],
    })
    assert_frame_equal(actual, expected, check_like=True)


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
        ('morph-A', 1, 'mtype-A', 'etype-A', 'prop-A', 'me_combo-A2'),
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
