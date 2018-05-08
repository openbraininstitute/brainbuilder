"""
Building astrocytes.
"""

import logging

import click
import numpy as np
import pandas as pd

from voxcell import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from brainbuilder.cell_positions import create_cell_positions
from brainbuilder.utils.random import parse_distr


L = logging.getLogger('brainbuilder')


@click.group()
def app():
    """ Building astrocytes """
    pass


def _place(density, soma_radii_distr):
    """
    Generate astrocyte soma positions and radii.

    Args:
        density: VoxelData with target density [cell count / mm^3]
        soma_radii_distr: astrocyte radii distribution
    """
    cells = CellCollection()

    L.info("Generating soma positions...")
    cells.positions = create_cell_positions(density)

    cell_count = len(cells.positions)
    L.info("Total cell count: %d", cell_count)

    L.info("Assigning soma radii...")
    cells.properties = pd.DataFrame({
        'radius': soma_radii_distr.rvs(size=cell_count)
    })

    L.info("Done!")
    return cells


@app.command()
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option(
    "--density", help="Atlas data layer with densities", default='astrocytes', show_default=True
)
@click.option("--soma-radii", help="Soma radii distribution", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output MVD3", required=True)
def place(
    atlas, atlas_cache, density,
    soma_radii,
    seed,
    output
):
    """ Generate positions & radii """
    np.random.seed(seed)

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    cells = _place(
        density=atlas.load_data(density),
        soma_radii_distr=parse_distr(soma_radii)
    )

    L.info("Export to MVD3...")
    cells.save(output)
