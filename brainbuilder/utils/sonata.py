"""
Temporary SONATA converters.

https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md
"""

import numpy as np
import pandas as pd
import h5py
import transforms3d

from voxcell import CellCollection


def _write_dataframe_by_columns(df, out):
    for name, column in df.iteritems():
        values, dtype = column.values, None
        if values.dtype in (object,):
            dtype = h5py.special_dtype(vlen=unicode)
        out.create_dataset(name, data=values, dtype=dtype)


def _write_node_group(node_group, out):
    properties, dynamic_params = node_group
    _write_dataframe_by_columns(properties, out)
    if dynamic_params is not None:
        _write_dataframe_by_columns(dynamic_params, out.create_group('dynamic_params'))


def _write_node_population(node_group, out):
    group_id = 0
    group_size = len(node_group[0])
    node_group_id = np.full(group_size, group_id)
    node_group_index = np.arange(group_size)
    out.create_dataset('node_group_id', data=node_group_id, dtype=np.uint32)
    out.create_dataset('node_group_index', data=node_group_index, dtype=np.uint32)
    out.create_dataset('node_type_id', data=np.full_like(node_group_id, -1, dtype=np.int32))
    _write_node_group(node_group, out.create_group(str(group_id)))


def _load_mecombo_info(filepath):
    COLUMN_FILTER = lambda name: name not in ('morph_name', 'layer', 'fullmtype', 'etype')
    return pd.read_csv(filepath, sep=r'\s+', usecols=COLUMN_FILTER, index_col='combo_name')


def _mvd3_to_node_group(mvd3_path, mecombo_info_path=None):
    """
    Convert MVD3 to a pair of pandas DataFrames with columns named SONATA-way.

    First DataFrame contains 'general' properties placed to node group root:
        - 'x', 'y', 'z'
        - 'rotation_angle_[x|y|z]axis'
        - 'morphology'
        - ...

    Second DataFrame contains emodel parameters placed to subgroup 'dynamic_params'.
    """
    properties = CellCollection.load_mvd3(mvd3_path).as_dataframe().reset_index(drop=True)

    if 'orientation' in properties:
        orientations = properties.pop('orientation')
        angles = np.stack(
            transforms3d.euler.mat2euler(m, axes='szyx') for m in orientations
        )
        properties['rotation_angle_xaxis'] = angles[:, 2]
        properties['rotation_angle_yaxis'] = angles[:, 1]
        properties['rotation_angle_zaxis'] = angles[:, 0]

    if ('me_combo' in properties) and (mecombo_info_path is not None):
        mecombos = properties.pop('me_combo')
        mecombo_info = _load_mecombo_info(mecombo_info_path)
        mecombo_params = mecombo_info.loc[mecombos]
        dynamic_params = pd.DataFrame(index=properties.index)
        for name, series in mecombo_params.iteritems():
            values = series.values
            if name == 'emodel':
                values = [('hoc:' + v) for v in values]
                properties['model_template'] = values
            else:
                dynamic_params[name] = values
    else:
        dynamic_params = None

    return properties, dynamic_params


def write_nodes_from_mvd3(mvd3_path, mecombo_info_path, out_h5_path, population):
    """ Export MVD3 + MEComboInfoFile to SONATA node collection. """
    node_group = _mvd3_to_node_group(mvd3_path, mecombo_info_path)
    with h5py.File(out_h5_path, 'w') as h5f:
        _write_node_population(
            node_group,
            h5f.create_group('/nodes/%s' % population)
        )
