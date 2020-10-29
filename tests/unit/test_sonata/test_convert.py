import json
from pathlib import Path
import tempfile
import voxcell
import pandas as pd

from nose.tools import ok_, eq_, assert_raises
from pandas.testing import assert_frame_equal

from brainbuilder.utils.sonata import convert


def test__add_me_info():
    def _mock_cells():
        df = pd.DataFrame({'me_combo': ['me_combo_%d' % i for i in range(10)],
                           'morphology': ['morph_%d' % i for i in range(10)],
                           })
        df.index += 1
        return voxcell.CellCollection.from_dataframe(df)

    cells = _mock_cells()
    mecombo_info = pd.DataFrame({'combo_name': ['me_combo_%d' % i for i in range(10)],
                                 'threshold': ['threshold_%d' % i for i in range(10)],
                                 })
    convert._add_me_info(cells, mecombo_info)
    cells = cells.as_dataframe()

    eq_(len(cells), 10)
    ok_('@dynamics:threshold' in cells)

    cells = _mock_cells()
    mecombo_info = pd.DataFrame({'combo_name': ['me_combo_0' for _ in range(10)],
                                 'threshold': ['threshold_%d' % i for i in range(10)],
                                 })
    assert_raises(AssertionError, convert._add_me_info, cells, mecombo_info)


def test_provide_me_info():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        df = pd.DataFrame({'a': [1, 2]})
        input_cells = voxcell.CellCollection()
        input_cells.add_properties(df)
        input_cells_path = tmp_dir / 'input_cells.h5'
        output_cells_path = tmp_dir / 'output_cells.h5'
        input_cells.save(input_cells_path)
        convert.provide_me_info(input_cells_path, output_cells_path)
        output_cells = voxcell.CellCollection.load(output_cells_path)
        output_cells_df = output_cells.as_dataframe()
        expected_df = pd.DataFrame({
            'a': [1, 2],
            'model_type': ['biophysical', 'biophysical']},
            index=output_cells_df.index)
        assert_frame_equal(output_cells_df, expected_df,
                           check_like=True)  # ignore column ordering


def test_write_network_config():
    expected_config = json.loads('''
{
  "manifest": {
    "$BASE_DIR": "/base/dir",
    "$NETWORK_NODES_DIR": "$BASE_DIR/nodes/dir",
    "$NETWORK_EDGES_DIR": "$BASE_DIR/edges/dir"
  },
  "components": {
    "morphologies_dir": "$BASE_DIR/morph/dir",
    "biophysical_neuron_models_dir": "/emodel/dir"
  },
  "node_sets_file": "$BASE_DIR/node_sets/file",
  "networks": {
    "nodes": [
      {
        "nodes_file": "/nodes/file/nodes.h5",
        "node_types_file": null
      }
    ],
    "edges": [
      {
        "edges_file": "$NETWORK_EDGES_DIR/edges/file/edges_suffix.h5",
        "edge_types_file": null
      }
    ]
  }
}    
    ''')

    with tempfile.NamedTemporaryFile() as tfp:
        convert.write_network_config(
            '/base/dir', 'morph/dir', '/emodel/dir',
            'nodes/dir', ['/nodes/file'], 'node_sets/file',
            'edges/dir', '_suffix', ['edges/file'],
            tfp.name
        )
        actual_config = json.load(tfp)
        assert expected_config == actual_config
