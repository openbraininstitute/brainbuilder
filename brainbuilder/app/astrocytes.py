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
    cells.save_mvd3(output)


@app.group(name='domains')
def _domains():
    """ Generate astrocyte domains """


@_domains.command()
@click.argument('mvd3')
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def tesselate(mvd3, atlas, atlas_cache, output):
    """ Laguerre tesselation """
    import brainbuilder.ngv.microdomains as _impl

    cells = CellCollection.load_mvd3(mvd3)
    brain_regions = Atlas.open(atlas, cache_dir=atlas_cache).load_data('brain_regions')

    result = _impl.tesselate(cells.positions, cells.properties['radius'], brain_regions)

    _impl.export_structure(result, output)


@_domains.command()
@click.argument('tesselation')
@click.option("--distr", help="Overlap ratio distribution", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def overlap(tesselation, distr, seed, output):
    """ Scale isotropically in all dimensions """
    import brainbuilder.ngv.microdomains as _impl

    np.random.seed(seed)

    source = _impl.load_structure(tesselation)
    result = _impl.overlap(source, overlap_distr=parse_distr(distr))

    _impl.export_structure(result, output)


@_domains.command()
@click.argument('tesselation')
@click.option("-o", "--output", help="Path to output mesh file (.stl)", required=True)
def mesh(tesselation, output):
    """ Generate STL mesh of domain surfaces """
    import brainbuilder.ngv.microdomains as _impl

    source = _impl.load_structure(tesselation)
    _impl.export_meshes(source, output)
