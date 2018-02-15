'''compatibility functions with existing BBP formats'''

import logging

from six import iteritems, text_type
from six.moves import zip

import h5py
import lxml.etree
import numpy as np
import pandas as pd

from voxcell import CellCollection
from voxcell.math_utils import angles_to_matrices
from voxcell.utils import deprecate

from brainbuilder.version import VERSION
from brainbuilder.exceptions import BrainBuilderError


L = logging.getLogger(__name__)


def _parse_recipe(recipe_filename):
    '''parse a BBP recipe and return the corresponding etree'''
    parser = lxml.etree.XMLParser(resolve_entities=False)
    return lxml.etree.parse(recipe_filename, parser=parser)


def bind_profile1d_to_atlas(profile1d, relative_distance):
    """
    Bind 1D profile to an atlas.

    Args:
        profile1d: 1D NumPy vector of length 100
        relative_distance: VoxelData with [0..1] relative distance from region bottom to top

    Returns:
        VoxelData with `profile1d` bound to an atlas using `relative_distance`.
    """
    if len(profile1d) != 100:
        raise BrainBuilderError("1D profile should specify 100 numbers")
    mask = ~np.isnan(relative_distance.raw)
    dist = relative_distance.raw[mask]
    if (dist < 0).any() or (dist > 1).any():
        raise BrainBuilderError("Relative distance should be in 0..1 range")
    bins = np.clip(np.floor(100 * dist).astype(int), 0, 99)
    result = np.full_like(relative_distance.raw, np.nan)
    result[mask] = profile1d[bins]
    return relative_distance.with_data(result)


def load_neurondb_v3(neurondb_filename):
    '''load a neurondb v3 file

    Returns:
        A DataFrame where the columns are:
            morphology, layer, mtype, etype, me_combo
    '''
    columns = [
        'morphology',
        'layer',
        'mtype',
        'etype',
        'me_combo',
    ]
    return pd.read_csv(
        neurondb_filename, sep=r'\s+', names=columns, usecols=range(5), na_filter=False
    )


def parse_mvd2(filepath):
    '''loads an mvd2 as a dict data structure with tagged fields'''

    sections = {
        'Neurons Loaded': (
            ('morphology', str),
            ('database', int), ('hyperColumn', int), ('miniColumn', int),
            ('layer', int), ('mtype', int), ('etype', int),
            ('x', float), ('y', float), ('z', float), ('r', float), ('me_combo', str)
        ),
        'MicroBox Data': (
            ('size_x', float), ('size_y', float), ('size_z', float),
            ('layer_6_percentage', float),
            ('layer_5_percentage', float),
            ('layer_4_percentage', float),
            ('layer_3_percentage', float),
            ('layer_2_percentage', float)
        ),
        'MiniColumnsPosition': (('x', float), ('y', float), ('z', float)),
        'CircuitSeeds': (('RecipeSeed', float), ('ColumnSeed', float), ('SynapseSeed', float)),
        'MorphTypes': (('name', str), ('mclass', str), ('sclass', str)),
        'ElectroTypes': (('name', str),),
    }

    result = {}

    section_names = dict((s.lower(), s) for s in sections)

    current_section = 'HEADER'

    with open(filepath) as f:
        for exact_line in f.readlines():
            line = exact_line.strip()

            if line.lower() in section_names:
                current_section = section_names[line.lower()]
            else:
                if current_section in sections:
                    fields = sections[current_section]
                    parsed = dict((field_def[0], field_def[1](value))
                                  for field_def, value in zip(fields, line.split()))

                    result.setdefault(current_section, []).append(parsed)
                else:
                    assert current_section == 'HEADER'
                    result.setdefault(current_section, '')
                    result[current_section] += exact_line + '\n'

    return result


def _matrices_to_angles(matrices):
    """
    Convert 3x3 rotation matrices to rotation angles around Y.

    Use NaN if rotation could not be represented as a single rotation angle.
    """
    phi = np.arccos(matrices[:, 0, 0]) * np.sign(matrices[:, 0, 2])
    mat = angles_to_matrices(phi, axis='y')
    valid = np.all(np.isclose(mat, matrices), axis=(1, 2))
    phi[~valid] = np.nan
    return phi


def load_mvd2(filepath):
    '''loads an mvd2 as a CellCollection'''
    deprecate.fail("""
        This method would be removed in `brainbuilder>=0.8`.
        Please contact NSE team in case you are using it.
    """)
    data = parse_mvd2(filepath)

    cells = CellCollection()

    cells.positions = np.array([[c['x'], c['y'], c['z']] for c in data['Neurons Loaded']])

    angles = np.array([c['r'] for c in data['Neurons Loaded']]) * np.pi / 180
    cells.orientations = angles_to_matrices(angles, axis='y')

    props = pd.DataFrame({
        'synapse_class': [data['MorphTypes'][c['mtype']]['sclass']
                          for c in data['Neurons Loaded']],
        'morph_class': [data['MorphTypes'][c['mtype']]['mclass'] for c in data['Neurons Loaded']],
        'mtype': [data['MorphTypes'][c['mtype']]['name'] for c in data['Neurons Loaded']],
        'etype': [data['ElectroTypes'][c['etype']]['name'] for c in data['Neurons Loaded']],
        'morphology': [c['morphology'] for c in data['Neurons Loaded']],
        'layer': [str(1 + c['layer']) for c in data['Neurons Loaded']],
        'hypercolumn': [c['hyperColumn'] for c in data['Neurons Loaded']],
        'minicolumn': [c['miniColumn'] for c in data['Neurons Loaded']],
        'me_combo': [c['me_combo'] for c in data['Neurons Loaded']],
    })

    cells.add_properties(props)
    return cells


def save_mvd2(filepath, morphology_path, cells):
    '''saves a CellCollection as mvd2

    Rotations might be lost in the process.
    Cells are expected to have the properties:
    'morphology', 'mtype', 'etype', 'morph_class', 'synapse_class', 'me_combo';
    and, optionally, 'hypercolumn', 'minicolumn', 'layer'.
    '''
    # pylint: disable=too-many-locals
    deprecate.fail("""
        This method would be removed in `brainbuilder>=0.8`.
        Please contact NSE team in case you are using it.
    """)
    rotations = 180 * _matrices_to_angles(cells.orientations) / np.pi
    if np.count_nonzero(np.isnan(rotations)):
        L.warning("save_mvd2: some rotations would be lost!")

    optional = {}
    for prop in ('hypercolumn', 'minicolumn', 'layer'):
        if prop in cells.properties:
            optional[prop] = cells.properties[prop]
        else:
            L.warning("save_mvd2: %s not specified, zero will be used", prop)
            optional[prop] = np.zeros(len(cells.properties), dtype=np.int)

    electro_types, chosen_etype = np.unique(cells.properties.etype, return_inverse=True)

    mtype_names, chosen_mtype = np.unique(cells.properties.mtype, return_inverse=True)

    morph_types = []
    for mtype_name in mtype_names:
        mask = (cells.properties.mtype == mtype_name).values
        morph_types.append((mtype_name,
                            cells.properties[mask].morph_class.values[0],
                            cells.properties[mask].synapse_class.values[0]))

    def get_mvd2_neurons():
        '''return the data for all the neurons used in the circuit'''
        data = zip(
            cells.properties.morphology,
            cells.positions,
            rotations,
            chosen_mtype,
            chosen_etype,
            optional['hypercolumn'],
            optional['minicolumn'],
            optional['layer'],
            cells.properties.me_combo,
        )

        for morph, pos, phi, mtype_idx, etype_idx, hypercolumn, minicolumn, layer, me_combo in data:
            yield dict(name=morph, mtype_idx=mtype_idx, etype_idx=etype_idx,
                       rotation=phi, x=pos[0], y=pos[1], z=pos[2], hypercolumn=hypercolumn,
                       minicolumn=minicolumn, layer=int(layer) - 1, me_combo=me_combo)

    with open(filepath, 'w') as fd:
        fd.write("Application:'BrainBuilder {version}'\n"
                 "{morphology_path}\n"
                 "/unknown/\n".format(version=VERSION, morphology_path=morphology_path))

        fd.write('Neurons Loaded\n')
        line = ('{name} {database} {hypercolumn} {minicolumn} {layer} {mtype_idx} '
                '{etype_idx} {x} {y} {z} {rotation} {me_combo}\n')
        fd.writelines(line.format(database=0, **c) for c in get_mvd2_neurons())

        # skipping sections:
        # MicroBox Data
        # MiniColumnsPosition
        # CircuitSeeds

        fd.write('MorphTypes\n')
        fd.writelines('%s %s %s\n' % m for m in morph_types)

        fd.write('ElectroTypes\n')
        fd.writelines('%s\n' % e for e in electro_types)


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


def _get_recipe_mtypes(recipe_path):
    """ List of mtypes in the order they appear in builder recipe. """
    result = []
    recipe = _parse_recipe(recipe_path)  # pylint: disable=protected-access
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
        h5f.create_dataset('/library/mtype', data=recipe_mtypes, dtype=dt)
