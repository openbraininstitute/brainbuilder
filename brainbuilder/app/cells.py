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
import json
import numbers
import os

from collections.abc import Mapping

import click
import numpy as np
import pandas as pd
import yaml

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


def load_mini_frequencies(filepath):
    """
    Load mini frequencies from a TSV file.
    """
    return pd.read_csv(filepath, sep=r'\s+', index_col="layer")


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

    for prop, value in conf['traits'].items():
        if isinstance(value, Mapping):
            values, probs = zip(*value.items())
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


def _assign_mini_frequencies(cells, mini_frequencies):
    """
    Add the mini_frequency column to `cells`.
    """
    mfreqs_cells = mini_frequencies.loc[cells.layer.values]
    _assign_property(cells, "exc_mini_frequency", mfreqs_cells.exc_mini_frequency.values)
    _assign_property(cells, "inh_mini_frequency", mfreqs_cells.inh_mini_frequency.values)


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
    input_path,
    composition_path,
    mtype_taxonomy_path,
    atlas_url,
    mini_frequencies_path=None,
    atlas_cache=None,
    region=None, mask_dset=None,
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

    if mini_frequencies_path is not None:
        mini_frequencies = load_mini_frequencies(mini_frequencies_path)
        L.info("Assigning mini-frequencies")
        _assign_mini_frequencies(result, mini_frequencies)

    for prop, dset in atlas_properties or []:
        L.info("Assigning '%s'...", prop)
        _assign_atlas_property(result, prop, atlas, dset)

    if append_hemisphere:
        result['region'] = result['region'] + '@' + result['hemisphere']

    if sort_by:
        L.info("Sorting CellCollection...")
        result.sort_values(sort_by, inplace=True)

    L.info("Done!")

    result.index = 1 + np.arange(len(result))
    if input_path is None:
        return CellCollection.from_dataframe(result)
    input_cells = CellCollection.load(input_path)
    out_cells = CellCollection.from_dataframe(pd.concat([input_cells.as_dataframe(), result]))
    out_cells.population_name = input_cells.population_name
    return out_cells


@app.command(short_help="Generate cell positions and me-types")
@click.option("--composition", help="Path to ME-type composition YAML", required=True)
@click.option("--mtype-taxonomy", help="Path to mtype taxonomy TSV", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option(
    "--mini-frequencies", help="Path to the mini frequencies TSV", default=None, show_default=True)
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
@click.option("-o", "--output", help="Path to output MVD3 or SONATA. Use .mvd3 file extension for"
                                     " MVD3, otherwise SONATA is used", required=True)
@click.option("--input", "input_path", default=None, help="Existing cells which are extended with"
                                                          "the new positioned cells")
def place(
    composition,
    mtype_taxonomy,
    atlas,
    mini_frequencies,
    atlas_cache, region, mask,
    density_factor, soma_placement,
    atlas_property, sort_by, append_hemisphere,
    seed,
    output,
    input_path
):
    """Places new cells into an existing cells or creates new cells if no existing were provided."""

    # pylint: disable=too-many-arguments, too-many-locals
    np.random.seed(seed)

    if sort_by is not None:
        sort_by = sort_by.split(",")

    cells = _place(
        input_path,
        composition,
        mtype_taxonomy,
        atlas,
        mini_frequencies_path=mini_frequencies,
        atlas_cache=atlas_cache,
        region=region, mask_dset=mask,
        density_factor=density_factor,
        soma_placement=soma_placement,
        atlas_properties=atlas_property,
        sort_by=sort_by,
        append_hemisphere=append_hemisphere
    )

    L.info("Export to %s", output)
    cells.save(output)


@app.command()
@click.argument("cells-path")
@click.option("--morphdb", help="Path to extNeuronDB.dat", required=True)
@click.option("--seed", type=int, help="Pseudo-random generator seed", default=0)
@click.option("-o", "--output", help="Path to output MVD3 or SONATA. Use .mvd3 file extension for"
                                     " MVD3, otherwise SONATA is used", required=True)
def assign_emodels(cells_path, morphdb, seed, output):
    """ Assign 'me_combo' property """
    np.random.seed(seed)

    cells = CellCollection.load(cells_path)
    morphdb = bbp.load_neurondb_v3(morphdb)
    result = bbp.assign_emodels(cells, morphdb)

    result.save(output)


def _parse_emodel_mapping(filepath):
    with open(filepath) as f:
        content = json.load(f)
    result = {}
    for emodel, mapping in content.items():
        assert isinstance(mapping['etype'], str)
        assert isinstance(mapping['layer'], list)
        etype = mapping['etype']
        for layer in mapping['layer']:
            assert isinstance(layer, str)
            key = (layer, etype)
            assert key not in result
            result[key] = {'emodel': emodel}
    result = pd.DataFrame(result).transpose()
    result.index.names = ['layer', 'etype']
    return result


def _write_mecombo_tsv(out_tsv, cells, emodels, **optional_columns):
    """Writes mecombo tsv file with optional columns if provided"""
    me_combos = cells[
        ['morphology', 'layer', 'mtype', 'etype', 'me_combo']
    ].rename(columns={
        'morphology': 'morph_name',
        'mtype': 'fullmtype',
        'me_combo': 'combo_name',
    })
    me_combos['layer'] = me_combos['layer'].astype(str)
    me_combos = me_combos.join(emodels, on=('layer', 'etype'))
    if me_combos['emodel'].isna().any():
        mismatch = me_combos[me_combos['emodel'].isna()][['layer', 'etype']]
        raise BrainBuilderError("Can not assign emodels for: %s" % mismatch)
    COLUMNS = ['morph_name', 'layer', 'fullmtype', 'etype', 'emodel', 'combo_name']
    for column_name, column_value in optional_columns.items():
        if column_value is not None:
            me_combos[column_name] = column_value
            COLUMNS.append(column_name)
    me_combos[COLUMNS].to_csv(out_tsv, sep='\t', index=False)


@app.command()
@click.argument("cells-path")
@click.option("--emodels", help="Path to emodel -> etype mapping", required=True)
@click.option(
    "--threshold-current", type=float, help="Threshold current to use for all cells", default=None)
@click.option(
    "--holding-current", type=float, help="Holding current to use for all cells", default=None)
@click.option("--out-tsv", help="Path to output mecombo TSV", required=True)
@click.option(
    "--out-mvd3", help="Deprecated! Path to output MVD3 file. Use --out-cells-path instead.")
@click.option("--out-cells-path", help="Path to output cells file. Use .mvd3 file extension for"
                                       " MVD3, otherwise SONATA is used")
def assign_emodels2(cells_path, emodels, threshold_current, holding_current,
                    out_tsv=None, out_mvd3=None, out_cells_path=None):
    """ Assign 'me_combo' property; write me_combo.tsv """

    if out_cells_path is None and out_mvd3 is None:
        raise ValueError('Specify output file with --out-cells-path.')
    cells = CellCollection.load(cells_path)
    emodels = _parse_emodel_mapping(emodels)
    cells = cells.as_dataframe()
    cells['me_combo'] = cells.apply(lambda row: "{etype}_{layer}_{morph}".format(
        etype=row.etype,
        layer=row.layer,
        morph=os.path.basename(row.morphology)
    ), axis=1)

    if out_mvd3 is None and not out_cells_path.lower().endswith('mvd3'):
        cells['layer'] = cells['layer'].astype(str)
        cells = cells.join(emodels, on=('layer', 'etype'))
        if cells['emodel'].isna().any():
            mismatch = cells[cells['emodel'].isna()][['layer', 'etype']]
            raise BrainBuilderError("Can not assign emodels for: %s" % mismatch)
        cells.rename({'emodel': 'model_template'}, inplace=True)
        cells['model_template'] = 'hoc:' + cells['model_template']
        if threshold_current is not None:
            cells[cells.SONATA_DYNAMIC_PROPERTY + 'threshold_current'] = threshold_current
        if holding_current is not None:
            cells[cells.SONATA_DYNAMIC_PROPERTY + 'holding_current'] = holding_current

    _write_mecombo_tsv(out_tsv, cells, emodels,
                       threshold_current=threshold_current, holding_current=holding_current)
    result = CellCollection.from_dataframe(cells)
    if out_mvd3 is not None:
        result.save_mvd3(out_mvd3)
    else:
        # must be biophysical here as a emodel/me_combo is provided
        result.properties['model_type'] = "biophysical"
        result.save(out_cells_path)
