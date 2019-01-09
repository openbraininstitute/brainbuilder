"""
CellCollection building.

A collection of commands for creating CellCollection and augmenting its properties, in particular:

----

# `brainbuilder cells create2`

Based on YAML cell composition recipe, create MVD3 with:
 - cell positions
 - required cell properties: 'layer', 'mtype', 'etype'
 - additional cell properties prescribed by the recipe

----

# `brainbuilder cells assign_emodels`

Based on `extNeuronDB.dat` file, add 'me_combo' to existing MVD3.
MVD3 is expected to have the following properties already assigned:
 - 'layer', 'mtype', 'etype'

"""

import re
import logging
import numbers

from collections import Mapping

import click
import numpy as np
import pandas as pd
import yaml
import six

from voxcell import CellCollection, OrientationField, VoxelData
from voxcell import traits as tt
from voxcell.nexus.voxelbrain import Atlas
from voxcell.utils import deprecate

from brainbuilder import BrainBuilderError
from brainbuilder.cell_positions import create_cell_positions
from brainbuilder.cell_orientations import apply_random_rotation
from brainbuilder.utils import bbp


L = logging.getLogger('brainbuilder')


@click.group()
def app():
    """ Building CellCollection """
    pass


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
    deprecate.fail("""
        Cell composition recipes v1.x are deprecated.
        Please consider migrating to the latest recipe structure:
        https://bbpteam.epfl.ch/documentation/Circuit%20Building-1.5/bioname.html#cell-composition-yaml
    """)
    with open(filepath, 'r') as f:
        content = yaml.load(f)

    # TODO: validate the content against schema
    assert content['version'] in ('v1.0', 'v1.1', 'v1.2')

    return content['composition'], content.get('rotation')


def load_recipe2(filepath):
    """
    Load cell composition YAML recipe.

    TODO: link to spec
    """
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)

    # TODO: validate the content against schema
    assert content['version'] in ('v2.0',)

    return content


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
        if np.any(result[~mask] > 0):
            L.warning("Non-zero density outside of region mask")
            result[~mask] = 0
    elif value.endswith(".nrrd"):
        L.info("Loading 3D density profile from '%s'...", value)
        result = VoxelData.load_nrrd(value).raw.astype(np.float32)
        if np.any(result[~mask] > 0):
            L.warning("Non-zero density outside of region mask")
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


def _create(
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


@app.command(short_help="Create CellCollection", help=_create.__doc__)
@click.option("--composition", help="Path to ME-type composition YAML", required=True)
@click.option("--mtype-taxonomy", help="Path to mtype taxonomy TSV", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--region-ids", help="Comma-separated region IDs", default=None, show_default=True)
@click.option("--density-factor", help="Density factor", type=float, default=1.0, show_default=True)
@click.option("--soma-placement", help="Soma placement method", default='basic', show_default=True)
@click.option(
    "--assign-layer", is_flag=True, help="Assign 'layer' property", show_default=True)
@click.option(
    "--assign-column", is_flag=True, help="Assign 'hypercolumn' property", show_default=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def create(
    composition, mtype_taxonomy,
    atlas, atlas_cache, region_ids,
    density_factor, soma_placement,
    assign_layer, assign_column,
    seed,
    output
):
    # pylint: disable=missing-docstring,too-many-arguments
    deprecate.fail("""
        `brainbuilder cells create` command is deprecated.
        Please update your cell composition recipe according to:
          https://bbpteam.epfl.ch/documentation/Circuit%20Building-1.5/bioname.html#cell-composition-yaml
        and use `brainbuilder cells place` command instead.
    """)
    if region_ids is not None:
        region_ids = map(int, region_ids.split(","))

    np.random.seed(seed)

    cells = _create(
        composition,
        mtype_taxonomy,
        atlas, atlas_cache, region_ids,
        density_factor=density_factor,
        soma_placement=soma_placement,
        assign_layer=assign_layer,
        assign_column=assign_column,
    )

    L.info("Export to MVD3...")
    cells.save_mvd3(output)


def _check_traits(traits):
    missing = set(['layer', 'mtype', 'etype']).difference(traits)
    if missing:
        raise BrainBuilderError(
            "Missing properties {} for group {}".format(list(missing), traits)
        )


def _create_cell_group(conf, atlas, root_mask, density_factor, soma_placement):
    _check_traits(conf['traits'])

    region_mask = atlas.get_region_mask(conf['region'], with_descendants=True, memcache=True)
    if root_mask is not None:
        region_mask.raw &= root_mask.raw
    if not np.any(region_mask.raw):
        raise BrainBuilderError("Empty region mask for region: '%s'" % conf['region'])

    density = region_mask.with_data(
        _load_density(conf['density'], region_mask.raw, atlas)
    )

    pos = create_cell_positions(density, density_factor=density_factor, method=soma_placement)
    result = pd.DataFrame(pos, columns=['x', 'y', 'z'])

    for prop, value in six.iteritems(conf['traits']):
        if isinstance(value, Mapping):
            values, probs = zip(*six.iteritems(value))
            if not np.allclose(np.sum(probs), 1.0):
                L.warning("Weights don't sum up to 1.0 for %s; renormalizing them", str(value))
                probs = probs / np.sum(probs)
            result[prop] = np.random.choice(values, size=len(pos), p=probs)
        else:
            result[prop] = value

    L.info("%s... [%d cells]", conf['traits'], len(result))
    return result


def _assign_mtype_traits(cells, mtype_taxonomy):
    traits = mtype_taxonomy.loc[cells.properties['mtype']]
    cells.properties['morph_class'] = traits['mClass'].values
    cells.properties['synapse_class'] = traits['sClass'].values


def _assign_region(cells, atlas):
    brain_regions = atlas.load_data('brain_regions')
    region_map = atlas.load_region_map()

    ids, idx = np.unique(brain_regions.lookup(cells.positions), return_inverse=True)
    names = np.array([region_map.get(_id, attr='acronym') for _id in ids])
    cells.properties['region'] = names[idx]


def _as_bool_mask(dset):
    if dset.raw.dtype not in (np.bool, np.uint8, np.int8):
        raise BrainBuilderError("Unexpected datatype for 0/1 mask: %s" % dset.raw.dtype)
    return dset.with_data(dset.raw.astype(bool))


def _place(
    composition_path,
    mtype_taxonomy_path,
    atlas_url, atlas_cache=None, region=None, mask_dset=None,
    soma_placement='basic',
    density_factor=1.0,
):
    """
    Create CellCollection

    # TODO: fill in the details
    """
    # pylint: disable=too-many-locals
    atlas = Atlas.open(atlas_url, cache_dir=atlas_cache)

    recipe = load_recipe2(composition_path)
    mtype_taxonomy = load_mtype_taxonomy(mtype_taxonomy_path)

    # Cache frequently used atlas data
    atlas.load_data('brain_regions', memcache=True)
    atlas.load_region_map(memcache=True)

    if mask_dset is None:
        root_mask = None
    else:
        root_mask = _as_bool_mask(atlas.load_data(mask_dset))

    if region is not None:
        region_mask = atlas.get_region_mask(region, with_descendants=True)
        if root_mask is None:
            root_mask = region_mask
        else:
            root_mask.raw &= region_mask.raw

    L.info("Creating cell groups...")
    groups = [
        _create_cell_group(conf, atlas, root_mask, density_factor, soma_placement)
        for conf in recipe['neurons']
    ]

    L.info("Merging into single CellCollection...")
    merged = pd.concat(groups)
    merged.index = 1 + np.arange(len(merged))
    result = CellCollection.from_dataframe(merged)

    L.info("Total cell count: %d", len(result.positions))

    L.info("Assigning 'morph_class' / 'synapse_class'...")
    _assign_mtype_traits(result, mtype_taxonomy)

    L.info("Assigning 'region'...")
    _assign_region(result, atlas)

    L.info("Done!")

    return result


@app.command(short_help="Create CellCollection", help=_place.__doc__)
@click.option("--composition", help="Path to ME-type composition YAML", required=True)
@click.option("--mtype-taxonomy", help="Path to mtype taxonomy TSV", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--region", help="Region name filter", default=None, show_default=True)
@click.option("--density-factor", help="Density factor", type=float, default=1.0, show_default=True)
@click.option("--soma-placement", help="Soma placement method", default='basic', show_default=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def create2(
    composition, mtype_taxonomy,
    atlas, atlas_cache, region,
    density_factor, soma_placement,
    seed,
    output
):
    # pylint: disable=missing-docstring,too-many-arguments
    deprecate.warn("""
        `brainbuilder cells create2` has been renamed.
        Please consider using `brainbuilder cells place` command instead;
        `create2` alias would be removed in future versions of `brainbuilder`.
    """)
    np.random.seed(seed)

    cells = _place(
        composition,
        mtype_taxonomy,
        atlas, atlas_cache,
        region=region, mask_dset=None,
        density_factor=density_factor,
        soma_placement=soma_placement,
    )

    L.info("Export to MVD3...")
    cells.save_mvd3(output)


@app.command(short_help="Generate cell positions and me-types", help=_place.__doc__)
@click.option("--composition", help="Path to ME-type composition YAML", required=True)
@click.option("--mtype-taxonomy", help="Path to mtype taxonomy TSV", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--region", help="Region name filter", default=None, show_default=True)
@click.option("--mask", help="Dataset with volumetric mask filter", default=None, show_default=True)
@click.option("--density-factor", help="Density factor", type=float, default=1.0, show_default=True)
@click.option("--soma-placement", help="Soma placement method", default='basic', show_default=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def place(
    composition, mtype_taxonomy,
    atlas, atlas_cache, region, mask,
    density_factor, soma_placement,
    seed,
    output
):
    # pylint: disable=missing-docstring,too-many-arguments
    np.random.seed(seed)

    cells = _place(
        composition,
        mtype_taxonomy,
        atlas, atlas_cache,
        region=region, mask_dset=mask,
        density_factor=density_factor,
        soma_placement=soma_placement,
    )

    L.info("Export to MVD3...")
    cells.save_mvd3(output)


@app.command()
@click.argument("mvd3")
@click.option("--morphdb", help="Path to extNeuronDB.dat", required=True)
@click.option("--seed", type=int, help="Pseudo-random generator seed", default=0)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def assign_emodels(mvd3, morphdb, seed, output):
    """ Assign 'me_combo' property """
    np.random.seed(seed)

    mvd3 = CellCollection.load_mvd3(mvd3)
    morphdb = bbp.load_neurondb_v3(morphdb)

    result = bbp.assign_emodels(mvd3, morphdb)
    result.save_mvd3(output)
