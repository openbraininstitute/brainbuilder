"""
Target generation.
"""
from brainbuilder.utils import bbp


def _synapse_class_name(synclass):
    return {
        'EXC': 'Excitatory',
        'INH': 'Inhibitory',
    }[synclass]


def _layer_name(layer):
    return "Layer%d" % layer


def _column_name(column):
    return "mc%d_Column" % column


def write_start_targets(cells, output):
    """ Write default property-based targets. """
    df = cells.as_dataframe()
    bbp.write_target(output, 'Mosaic', include_targets=['All'])
    bbp.write_target(output, 'All', include_targets=sorted(df['mtype'].unique()))
    bbp.write_property_targets(output, df, 'synapse_class', mapping=_synapse_class_name)
    bbp.write_property_targets(output, df, 'mtype')
    bbp.write_property_targets(output, df, 'etype')
    bbp.write_property_targets(output, df, 'region')
    if 'layer' in df:
        bbp.write_property_targets(output, df, 'layer', mapping=_layer_name)
    if 'hypercolumn' in df:
        bbp.write_property_targets(output, df, 'hypercolumn', mapping=_column_name)
