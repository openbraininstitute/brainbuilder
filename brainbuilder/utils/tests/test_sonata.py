import voxcell
import pandas as pd

from nose.tools import ok_, eq_, assert_raises
from brainbuilder.utils import sonata


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
    sonata._add_me_info(cells, mecombo_info)
    cells = cells.as_dataframe()

    eq_(len(cells), 10)
    ok_('@dynamics:threshold' in cells)

    cells = _mock_cells()
    mecombo_info = pd.DataFrame({'combo_name': ['me_combo_0' for _ in range(10)],
                                 'threshold': ['threshold_%d' % i for i in range(10)],
                                 })
    assert_raises(AssertionError, sonata._add_me_info, cells, mecombo_info)
