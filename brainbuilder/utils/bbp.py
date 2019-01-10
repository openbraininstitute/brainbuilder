'''compatibility functions with existing BBP formats'''

import logging

from six import iteritems, text_type

import h5py
import lxml.etree
import numpy as np
import pandas as pd

from voxcell import CellCollection

from brainbuilder.exceptions import BrainBuilderError


L = logging.getLogger(__name__)


def load_neurondb(neurondb_filename, with_emodels=False):
    '''load a neurondb file

    Returns:
        A DataFrame where the columns are:
            morphology, layer, mtype [,etype, me_combo]
    '''
    columns = [
        'morphology',
        'layer',
        'mtype',
    ]
    if with_emodels:
        columns.extend(['etype', 'me_combo'])
    return pd.read_csv(
        neurondb_filename, sep=r'\s+', names=columns, usecols=range(len(columns)), na_filter=False
    )


def load_neurondb_v3(neurondb_filename):
    '''load a neurondb v3 file

    Returns:
        A DataFrame where the columns are:
            morphology, layer, mtype, etype, me_combo
    '''
    return load_neurondb(neurondb_filename, with_emodels=True)


def gid2str(gid):
    """ 42 -> 'a42' """
    return "a%d" % gid


def write_target(f, name, gids=None, include_targets=None):
    """ Append contents to .target file. """
    f.write("\nTarget Cell %s\n{\n" % name)
    if gids is not None:
        f.write("  ")
        f.write(" ".join(map(gid2str, gids)))
        f.write("\n")
    if include_targets is not None:
        f.write("  ")
        f.write(" ".join(include_targets))
        f.write("\n")
    f.write("}\n")


def write_property_targets(f, cells, prop, mapping=None):
    """ Append targets based on 'prop' cell property to .target file. """
    for value, gids in sorted(iteritems(cells.groupby(prop).groups)):
        if mapping is not None:
            value = mapping(value)
        write_target(f, value, gids=gids)


def assign_emodels(cells, morphdb):
    """ Assign electrical models to CellCollection based MorphDB. """
    df = cells.as_dataframe()

    ME_COMBO = 'me_combo'
    if ME_COMBO in df:
        L.warning("'%s' property would be overwritten", ME_COMBO)
        del df[ME_COMBO]

    JOIN_COLS = ['morphology', 'layer', 'mtype', 'etype']

    df = df.join(morphdb.set_index(JOIN_COLS)[ME_COMBO], on=JOIN_COLS)

    not_assigned = np.count_nonzero(df[ME_COMBO].isnull())
    if not_assigned > 0:
        raise BrainBuilderError("Could not pick emodel for %d cell(s)" % not_assigned)

    # choose 'me_combo' randomly if several are available
    df = df.sample(frac=1)
    df = df[~df.index.duplicated(keep='first')]

    df = df.sort_index()

    result = CellCollection.from_dataframe(df)
    result.seeds = cells.seeds

    return result


def _parse_recipe(recipe_filename):
    '''parse a BBP recipe and return the corresponding etree'''
    parser = lxml.etree.XMLParser(resolve_entities=False)
    return lxml.etree.parse(recipe_filename, parser=parser)


def _get_recipe_mtypes(recipe_path):
    """ List of mtypes in the order they appear in builder recipe. """
    result = []
    recipe = _parse_recipe(recipe_path)
    for elem in recipe.iterfind('/NeuronTypes/Layer/StructuralType'):
        mtype = elem.attrib['id']
        if mtype not in result:
            result.append(mtype)
    return result


def reorder_mtypes(mvd3_path, recipe_path):
    """ Re-order /library/mtypes to align with builder recipe. """
    recipe_mtypes = _get_recipe_mtypes(recipe_path)
    with h5py.File(mvd3_path, 'a') as h5f:
        mvd3_mtypes = h5f.pop('/library/mtype')[:]
        mapping = np.zeros(len(mvd3_mtypes), dtype=np.uint8)
        for k, mtype in enumerate(mvd3_mtypes):
            mapping[k] = recipe_mtypes.index(mtype)

        mvd3_mtype_index = h5f.pop('/cells/properties/mtype')[:]
        dt = h5py.special_dtype(vlen=text_type)
        h5f['/cells/properties/mtype'] = mapping[mvd3_mtype_index]
        h5f.create_dataset(
            '/library/mtype', data=np.asarray(recipe_mtypes).astype(object), dtype=dt
        )
