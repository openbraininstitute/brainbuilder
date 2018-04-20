"""
CellCollection building.
"""

import re
import logging
import numbers

import numpy as np
import pandas as pd
import h5py
import yaml
import six

from voxcell import CellCollection, OrientationField, VoxelData
from voxcell import traits as tt

from brainbuilder import BrainBuilderError
from brainbuilder.cell_positions import create_cell_positions
from brainbuilder.cell_orientations import apply_random_rotation
from brainbuilder.nexus.voxelbrain import Atlas


L = logging.getLogger('brainbuilder')


def load_recipe(filepath):
    """
    Load me-type composition and rotation from YAML recipe.

    TODO: link to spec

    Example YAML:
    >
      version: v1.0
      composition:
        L1:
          L1_HAC:
            density: HAC.nrrd
            density_factor: 0.9
            etypes:
              bNAC: 0.2
               cIR:  0.8
          L1_DAC:
                ...
        L2:
          L23_PC:
            density: 11000  # cells / mm^3
            etypes:
              cADpyr: 1.0
    """
    with open(filepath, 'r') as f:
        content = yaml.load(f)

    # TODO: validate the content against schema
    assert content['version'] in ('v1.0', 'v1.1', 'v1.2')

    return content['composition'], content.get('rotation')


def load_mtype_taxonomy(filepath):
    """
    Load mtype taxonomy from TSV file.

    TODO: link to spec
    """
    # TODO: validate
    return pd.read_csv(filepath, sep=r'\s+', index_col='mtype')


def _load_density(value, mask, atlas):
    """ Load density as 3D numpy array.

        Args:
            value: one of
                - float value (constant density per `mask`)
                - path to NRRD file (load from file, filter by `mask`)
                - dataset in `atlas` (load from atlas, filter by `mask`)
            mask: 0/1 3D mask
            atlas: Atlas to use for loading atlas datasets

        `value` of form '{name}' is recognized as atlas dataset 'name'.

        Returns:
            3D float32 numpy array of same shape as `mask`.
    """
    if isinstance(value, numbers.Number):
        result = np.zeros_like(mask, dtype=np.float32)
        result[mask] = float(value)
    elif value.startswith("{"):
        assert value.endswith("}")
        dataset = value[1:-1]
        L.info("Loading 3D density profile from '%s' atlas dataset...", dataset)
        result = atlas.load_data(dataset, cls=VoxelData).raw.astype(np.float32)
        result[~mask] = 0
    elif value.endswith(".nrrd"):
        L.info("Loading 3D density profile from '%s'...", value)
        result = VoxelData.load_nrrd(value).raw.astype(np.float32)
        result[~mask] = 0
    else:
        raise BrainBuilderError("Unexpected density value: '%s'" % value)
    return result


def _bind_to_atlas(composition, atlas, include_region_ids=None):
    """
    Bind me-type composition to atlas.

    Returns:
    (total_density, mtype_sdist) tuple:
        total_density: VoxelData in same space as the `atlas`
        mtype_sdist: SpatialDistribution with ('region', 'mtype') traits
    """
    # pylint: disable=too-many-locals

    brain_regions = atlas.load_data('brain_regions')
    hierarchy = atlas.load_hierarchy()

    if include_region_ids is not None:
        include_region_ids = {
            rid
            for root_id in include_region_ids
            for rid in hierarchy.collect('id', root_id, 'id')
        }

    densities, traits = [], []

    L.info("Loading mtype densities...")
    for region in sorted(composition):
        region_ids = hierarchy.collect('acronym', region, 'id')
        if not region_ids:
            raise BrainBuilderError("Region not found: '%s'" % region)
        if include_region_ids is not None:
            region_ids = include_region_ids.intersection(region_ids)
        if not region_ids:
            raise BrainBuilderError("Region filtered out: '%s'" % region)
        region_mask = np.isin(brain_regions.raw, list(region_ids))
        if not np.any(region_mask):
            L.warning("No voxels found for region '%s'", region)
            continue
        for mtype in sorted(composition[region]):
            params = composition[region][mtype]
            density = _load_density(params['density'], region_mask, atlas)
            densities.append(density)
            traits.append((region, mtype))

    total_density = np.sum(np.stack(densities), axis=0)
    ijk = np.nonzero(total_density > 0)
    if len(ijk[0]) == 0:
        raise BrainBuilderError("No voxel with total density > 0")

    L.info("Composing (region, mtype) SpatialDistribution...")
    traits = pd.DataFrame(traits, columns=['region', 'mtype'])

    unique_dist, dist_idx = np.unique(
        np.stack(density[ijk] for density in densities), axis=1, return_inverse=True
    )
    distributions = pd.DataFrame(
        unique_dist / np.sum(unique_dist, axis=0)
    )

    field = np.full_like(brain_regions.raw, -1, dtype=np.int32)
    field[ijk] = dist_idx

    return (
        brain_regions.with_data(total_density),
        tt.SpatialDistribution(
            brain_regions.with_data(field),
            distributions,
            traits
        ),
    )


def _get_etype_ratios(composition):
    """
    Get etype ratios from composition recipe.

    Returns:
        ('region', 'mtype') -> { etype_ratios }
    """
    return {
        (region, mtype): mtype_group['etypes']
        for region, region_group in six.iteritems(composition)
        for mtype, mtype_group in six.iteritems(region_group)
    }


def _pick_etypes(region_mtype, etype_ratios):
    """ Pick etypes for given (region, mtype) pairs using `etype_ratios. """
    result = pd.Series(index=region_mtype.index)
    for key, group in region_mtype.groupby(['region', 'mtype']):
        etypes, prob = zip(*sorted(etype_ratios[key].items()))
        prob = prob / np.sum(prob)
        result[group.index] = np.random.choice(etypes, size=len(group), p=prob, replace=True)
    return result


def _region_to_layer(region):
    """ Convert 'L<k>' region names to integer layer IDs. """
    def _f(name):
        assert name.startswith("L")
        return int(name[1:])
    return region.apply(_f)


def _get_hypercolumn(positions, atlas):
    """ Get hypercolumn ID for given positions. """
    hierarchy = atlas.load_hierarchy()
    brain_regions = atlas.load_data('brain_regions')
    region_ids = brain_regions.lookup(positions)

    result = np.full_like(region_ids, -1, dtype=np.int16)

    REGEX = re.compile(r"hypercolumn (\d+)")
    for h in hierarchy.children:
        column_id = int(REGEX.match(h.data['name']).group(1))
        result[np.isin(region_ids, list(h.get('id')))] = column_id

    assert np.all(result >= 0)
    return result.astype(np.uint16)


def _get_mtype_property(mtype_taxonomy, mtypes, prop):
    return mtype_taxonomy.loc[mtypes, prop].values


def create(
    composition_path,
    mtype_taxonomy_path,
    atlas_url, atlas_cache=None, region_ids=None,
    soma_placement='basic',
    density_factor=1.0,
    assign_layer=False,
    assign_column=False,
):
    """
    Create CellCollection

    # TODO: fill in the details

    \b
    Properties assigned (in addition to position and orientation):
        - region
        - mtype
        - etype
        - morph_class
        - synapse_class
        - layer [if requested]
        - hypercolumn [if requested]
    """
    # pylint: disable=too-many-arguments,too-many-locals
    atlas = Atlas.open(atlas_url, cache_dir=atlas_cache)

    composition, rotation = load_recipe(composition_path)
    mtype_taxonomy = load_mtype_taxonomy(mtype_taxonomy_path)

    total_density, mtype_sdist = _bind_to_atlas(composition, atlas, region_ids)
    etype_ratios = _get_etype_ratios(composition)

    cells = CellCollection()

    L.info("Assigning cell positions...")
    cells.positions = create_cell_positions(
        total_density, density_factor=density_factor, method=soma_placement
    )

    L.info("Total cell count: %d", len(cells.positions))

    L.info("Assigning cell orientations...")
    orientation_field = atlas.load_data('orientation', cls=OrientationField)
    cells.orientations = orientation_field.lookup(cells.positions)

    L.info("Assigning region / mtype...")
    cells.properties = mtype_sdist.collect(cells.positions, names=['region', 'mtype'])

    L.info("Assigning etype...")
    cells.properties['etype'] = _pick_etypes(cells.properties[['region', 'mtype']], etype_ratios)

    L.info("Assigning morph_class / synapse_class...")
    mtypes = cells.properties['mtype']
    cells.properties['morph_class'] = _get_mtype_property(mtype_taxonomy, mtypes, 'mClass')
    cells.properties['synapse_class'] = _get_mtype_property(mtype_taxonomy, mtypes, 'sClass')

    if assign_layer:
        L.info("Assigning layer...")
        cells.properties['layer'] = _region_to_layer(cells.properties['region'])

    if assign_column:
        L.info("Assigning hypercolumn...")
        cells.properties['hypercolumn'] = _get_hypercolumn(cells.positions, atlas)

    if rotation is None:
        L.warning("Applying random uniform rotation around Y-axis for all cells...")
        cells.orientations = apply_random_rotation(
            cells.orientations, axis='y', distr=('uniform', {'low': -np.pi, 'high': np.pi})
        )
    else:
        for region, region_group in six.iteritems(rotation):
            for mtype, mtype_group in six.iteritems(region_group):
                L.info("Applying random rotation for (%s, %s) cells", region, mtype)
                mask = np.logical_and(
                    cells.properties['region'] == region,
                    cells.properties['mtype'] == mtype
                )
                if np.count_nonzero(mask) == 0:
                    raise BrainBuilderError("No (%s, %s) cells in the circuit" % (region, mtype))
                for axis, distr, params in mtype_group:
                    cells.orientations[mask] = apply_random_rotation(
                        cells.orientations[mask], axis=axis, distr=(distr, params)
                    )

    L.info("Done!")
    return cells


def assign_transcriptome(mvd3_path, db_path, output_path):
    """ Assign transcriptome to cells based on their mtype. """
    mapping = {}
    with h5py.File(db_path, 'r') as h5f:
        for mtype, idx in six.iteritems(h5f['/mapping/by_mtype']):
            mapping[mtype] = idx[:]
        genes = h5f['/library/genes'][:]
        expressions = h5f['/library/expressions'][:]

    cells = CellCollection.load(mvd3_path)

    assigned = np.zeros(len(cells.properties), dtype=np.int32)
    for mtype, group in cells.properties.groupby('mtype'):
        assigned[group.index] = np.random.choice(mapping[mtype], size=len(group))

    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('/cells/expressions', data=assigned)
        h5f.create_dataset('/library/genes', data=genes)
        h5f.create_dataset('/library/expressions', data=expressions)
