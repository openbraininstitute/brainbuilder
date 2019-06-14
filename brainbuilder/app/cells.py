"""
CellCollection building.

A collection of commands for creating CellCollection and augmenting its properties, in particular:

----

# `brainbuilder cells place`

Based on YAML cell composition recipe, create MVD3 with:
 - cell positions
 - required cell properties: 'layer', 'mtype', 'etype'
 - additional cell properties prescribed by the recipe and / or atlas

----

# `brainbuilder cells assign_emodels`

Based on `extNeuronDB.dat` file, add 'me_combo' to existing MVD3.
MVD3 is expected to have the following properties already assigned:
 - 'layer', 'mtype', 'etype'

"""

import logging
import numbers

from collections import Mapping

import click
import numpy as np
import pandas as pd
import yaml
import six

from voxcell import CellCollection, ROIMask, VoxelData
from voxcell.nexus.voxelbrain import Atlas

from brainbuilder import BrainBuilderError
from brainbuilder.cell_positions import create_cell_positions
from brainbuilder.utils import bbp


L = logging.getLogger('brainbuilder')


@click.group()
def app():
    """ Building CellCollection """


def load_recipe(filepath):
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
    elif value.endswith(".nrrd"):
        L.info("Loading 3D density profile from '%s'...", value)
        result = VoxelData.load_nrrd(value).raw.astype(np.float32)
    else:
        raise BrainBuilderError("Unexpected density value: '%s'" % value)

    # Mask away density values outside region mask (NaNs are fine there)
    result[~mask] = 0

    if np.any(np.isnan(result)):
        raise BrainBuilderError("NaN density values within region mask")

    return result


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


def _assign_property(cells, prop, values):
    if prop in cells:
        raise BrainBuilderError("Duplicate property: '%s'" % prop)
    cells[prop] = values


def _assign_mtype_traits(cells, mtype_taxonomy):
    traits = mtype_taxonomy.loc[cells['mtype']]
    _assign_property(cells, 'morph_class', traits['mClass'].values)
    _assign_property(cells, 'synapse_class', traits['sClass'].values)


def _assign_atlas_property(cells, prop, atlas, dset):
    if dset.startswith('~'):
        dset = dset[1:]
        resolve_ids = True
    else:
        resolve_ids = False

    xyz = cells[['x', 'y', 'z']].values
    if dset == 'FAST-HEMISPHERE':
        # TODO: remove as soon as "slow" way of assigning hemisphere
        # (with a volumetric dataset) is available
        values = np.where(xyz[:, 2] < 5700, u'left', u'right')
    else:
        values = atlas.load_data(dset).lookup(xyz)
        if resolve_ids:
            ids, idx = np.unique(values, return_inverse=True)
            rmap = atlas.load_region_map()
            resolved = np.array([rmap.get(_id, attr='acronym') for _id in ids])
            values = resolved[idx]

    _assign_property(cells, prop, values)


def _place(
    composition_path,
    mtype_taxonomy_path,
    atlas_url, atlas_cache=None, region=None, mask_dset=None,
    soma_placement='basic',
    density_factor=1.0,
    atlas_properties=None,
    sort_by=None,
    append_hemisphere=False
):
    # pylint: disable=too-many-arguments, too-many-locals
    atlas = Atlas.open(atlas_url, cache_dir=atlas_cache)

    recipe = load_recipe(composition_path)
    mtype_taxonomy = load_mtype_taxonomy(mtype_taxonomy_path)

    # Cache frequently used atlas data
    atlas.load_data('brain_regions', memcache=True)
    atlas.load_region_map(memcache=True)

    if mask_dset is None:
        root_mask = None
    else:
        root_mask = atlas.load_data(mask_dset, cls=ROIMask)

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
    result = pd.concat(groups)

    L.info("Total cell count: %d", len(result))

    L.info("Assigning 'morph_class' / 'synapse_class'...")
    _assign_mtype_traits(result, mtype_taxonomy)

    for prop, dset in (atlas_properties or []):
        L.info("Assigning '%s'...", prop)
        _assign_atlas_property(result, prop, atlas, dset)

    if append_hemisphere:
        result['region'] = result['region'] + '@' + result['hemisphere']

    if sort_by:
        L.info("Sorting CellCollection...")
        result.sort_values(sort_by, inplace=True)

    L.info("Done!")

    result.index = 1 + np.arange(len(result))
    return CellCollection.from_dataframe(result)


@app.command(short_help="Generate cell positions and me-types")
@click.option("--composition", help="Path to ME-type composition YAML", required=True)
@click.option("--mtype-taxonomy", help="Path to mtype taxonomy TSV", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--region", help="Region name filter", default=None, show_default=True)
@click.option("--mask", help="Dataset with volumetric mask filter", default=None, show_default=True)
@click.option("--density-factor", help="Density factor", type=float, default=1.0, show_default=True)
@click.option("--soma-placement", help="Soma placement method", default='basic', show_default=True)
@click.option(
    "--atlas-property", type=(str, str), multiple=True, help="Property based on atlas dataset")
@click.option("--sort-by", help="Sort by properties (comma-separated)", default=None)
@click.option(
    "--append-hemisphere", is_flag=True, help="Append hemisphere to region name", default=False)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def place(
    composition, mtype_taxonomy,
    atlas, atlas_cache, region, mask,
    density_factor, soma_placement,
    atlas_property, sort_by, append_hemisphere,
    seed,
    output
):
    """
    Create CellCollection

    # TODO: fill in the details
    """

    # pylint: disable=too-many-arguments
    np.random.seed(seed)

    if sort_by is not None:
        sort_by = sort_by.split(",")

    cells = _place(
        composition,
        mtype_taxonomy,
        atlas, atlas_cache,
        region=region, mask_dset=mask,
        density_factor=density_factor,
        soma_placement=soma_placement,
        atlas_properties=atlas_property,
        sort_by=sort_by,
        append_hemisphere=append_hemisphere
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


def _parse_emodel_mapping(filepath):
    import json
    with open(filepath) as f:
        content = json.load(f)
    result = {}
    for emodel, mapping in content.items():
        assert isinstance(mapping['etype'], six.string_types)
        assert isinstance(mapping['layer'], list)
        etype = mapping['etype']
        for layer in mapping['layer']:
            assert isinstance(layer, six.string_types)
            key = (layer, etype)
            assert key not in result
            result[key] = {'emodel': emodel}
    result = pd.DataFrame(result).transpose()
    result.index.names = ['layer', 'etype']
    return result


@app.command()
@click.argument("mvd3")
@click.option("--emodels", help="Path to emodel -> etype mapping", required=True)
@click.option(
    "--threshold-current", type=float, help="Threshold current to use for all cells", default=None)
@click.option(
    "--holding-current", type=float, help="Holding current to use for all cells", default=None)
@click.option("--out-mvd3", help="Path to output MVD3", required=True)
@click.option("--out-tsv", help="Path to output mecombo TSV", required=True)
def assign_emodels2(mvd3, emodels, threshold_current, holding_current, out_mvd3, out_tsv):
    """ Assign 'me_combo' property; write me_combo.tsv """
    import os.path
    mvd3 = CellCollection.load_mvd3(mvd3)
    emodels = _parse_emodel_mapping(emodels)
    cells = mvd3.as_dataframe()
    cells['me_combo'] = cells.apply(lambda row: "{etype}_{layer}_{morph}".format(
        etype=row.etype,
        layer=row.layer,
        morph=os.path.basename(row.morphology)
    ), axis=1)
    me_combos = cells[
        ['morphology', 'layer', 'mtype', 'etype', 'me_combo']
    ].rename(columns={
        'morphology': 'morph_name',
        'mtype': 'fullmtype',
        'me_combo': 'combo_name',
    })
    me_combos['layer'] = me_combos['layer'].astype(six.text_type)
    me_combos = me_combos.join(emodels, on=('layer', 'etype'))
    if me_combos['emodel'].isna().any():
        mismatch = me_combos[me_combos['emodel'].isna()][['layer', 'etype']]
        raise BrainBuilderError("Can not assign emodels for: %s" % mismatch)
    COLUMNS = ['morph_name', 'layer', 'fullmtype', 'etype', 'emodel', 'combo_name']
    if threshold_current is not None:
        me_combos['threshold_current'] = threshold_current
        COLUMNS.append('threshold_current')
    if holding_current is not None:
        me_combos['holding_current'] = holding_current
        COLUMNS.append('holding_current')
    me_combos[COLUMNS].to_csv(out_tsv, sep='\t', index=False)
    result = CellCollection.from_dataframe(cells)
    result.seeds = mvd3.seeds
    result.save_mvd3(out_mvd3)
