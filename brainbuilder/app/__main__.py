"""
Collection of tools for circuit building.
"""

from builtins import input  # pylint: disable=redefined-builtin

import logging
import shutil

import click
import numpy as np
import pandas as pd

from voxcell import CellCollection, VoxelData

from brainbuilder.app import (
    cells as app_cells,
    targets as app_targets,
    nrn as app_nrn
)
from brainbuilder.utils import bbp
from brainbuilder.version import VERSION


APP_NAME = 'brainbuilder'
L = logging.getLogger(APP_NAME)


@click.group()
@click.version_option(version=VERSION, prog_name=APP_NAME)
def main():
    """ Collection of tools for circuit building """
    logging.basicConfig(level=logging.INFO)


@main.group(name="cells")
def _cells():
    """ Building CellCollection """
    pass


@_cells.command(short_help="Create CellCollection", help=app_cells.create.__doc__)
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
    if region_ids is not None:
        region_ids = map(int, region_ids.split(","))

    np.random.seed(seed)

    cells = app_cells.create(
        composition,
        mtype_taxonomy,
        atlas, atlas_cache, region_ids,
        density_factor=density_factor,
        soma_placement=soma_placement,
        assign_layer=assign_layer,
        assign_column=assign_column,
    )

    L.info("Export to MVD3...")
    cells.save(output)


@_cells.command()
@click.argument("mvd3")
@click.option("--morphdb", help="Path to extNeuronDB.dat", required=True)
@click.option("--seed", type=int, help="Pseudo-random generator seed", default=0)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def assign_emodels(mvd3, morphdb, seed, output):
    """ Assign 'me_combo' property """
    np.random.seed(seed)

    mvd3 = CellCollection.load(mvd3)
    morphdb = bbp.load_neurondb_v3(morphdb)

    result = bbp.assign_emodels(mvd3, morphdb)
    result.save(output)


@_cells.command()
@click.argument("mvd3")
@click.option("--db", help="Path to HDF5 with gene expressions", required=True)
@click.option("--seed", type=int, help="Pseudo-random generator seed", default=0)
@click.option("-o", "--output", help="Path to output transcriptome file", required=True)
def assign_transcriptome(mvd3, db, seed, output):
    """ Assign transcriptome """
    np.random.seed(seed)
    app_cells.assign_transcriptome(mvd3, db, output)


@main.group(name="mvd3")
def _mvd3():
    """ Tools for working with MVD3 """
    pass


@_mvd3.command()
@click.argument("mvd3")
@click.option("--recipe", help="Path to builder recipe XML", required=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def reorder_mtypes(mvd3, recipe, output):
    """ Align /library/mtypes with builder recipe """
    tmp_path = output + "~"
    shutil.copy(mvd3, tmp_path)
    bbp.reorder_mtypes(tmp_path, recipe)
    shutil.move(tmp_path, output)


@_mvd3.command()
@click.argument("mvd3")
@click.option("-p", "--prop", help="Property name to use", required=True)
@click.option("-d", "--voxel-data", help="Path NRRD with to volumetric data", required=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def add_property(mvd3, prop, voxel_data, output):
    """ Add property to MVD3 based on volumetric data """
    cells = CellCollection.load(mvd3)
    if prop in cells.properties:
        choice = input(
            "There is already '%s' property in the provided MVD3. Overwrite (y/n)? " % prop
        )
        if choice.lower() not in ('y', 'yes'):
            return
    voxel_data = VoxelData.load_nrrd(voxel_data)
    cells.properties[prop] = voxel_data.lookup(cells.positions)
    cells.save(output)


@_mvd3.command()
@click.argument("mvd3")
@click.option("--seeds", help="Comma-separated circuit seeds (4 floats)", required=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def set_seeds(mvd3, seeds, output):
    """ Set /circuit/seeds """
    seeds = [float(x) for x in seeds.split(",")]
    assert len(seeds) == 4
    mvd3 = CellCollection.load(mvd3)
    mvd3.seeds = np.array(seeds, dtype=np.float64)
    mvd3.save(output)


@_mvd3.command()
@click.argument("mvd3")
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option("-o", "--output", help="Path to output MVD2", required=True)
def to_mvd2(mvd3, morph_dir, output):
    """ Convert to MVD2 """
    cells = CellCollection.load(mvd3)
    bbp.save_mvd2(output, morph_dir, cells)


@_mvd3.command()
@click.argument("mvd3", nargs=-1)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def merge(mvd3, output):
    """ Merge multiple MVD3 files """
    chunks = [CellCollection.load(filepath).as_dataframe() for filepath in mvd3]
    merged = pd.concat(chunks, ignore_index=True)
    merged.index = 1 + np.arange(len(merged))
    CellCollection.from_dataframe(merged).save(output)


@main.group(name="targets")
def _targets():
    """ Tools for working with .target files """
    pass


@_targets.command()
@click.argument("mvd3")
@click.option("-o", "--output", help="Path to output .target file", required=True)
def from_mvd3(mvd3, output):
    """ Generate .target file from MVD3 """
    cells = CellCollection.load(mvd3)
    with open(output, 'w') as f:
        app_targets.write_start_targets(cells, f)


@main.group(name="nrn")
def _nrn():
    """ Tools for working with NRN files """
    pass


@_nrn.command(name="merge", short_help="Merge NRN files", help=app_nrn.merge.__doc__)
@click.argument("nrn_dir")
@click.option(
    "--only", help="merge only the specified file (e.g --only=nrn_positions.h5)", default=""
)
@click.option(
    "--link", is_flag=True, help="make symbolic links instead of copying datasets"
)
def _nrn_merge(nrn_dir, only, link):
    app_nrn.merge(nrn_dir, only, link=link)


if __name__ == '__main__':
    main()
