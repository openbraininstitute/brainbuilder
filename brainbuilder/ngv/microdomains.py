"""
Astrocyte microdomains generation.
"""

import h5py
import numpy as np

import tess

from brainbuilder.geometry import ConvexPolyhedron
from brainbuilder import BrainBuilderError


def tesselate(points, radii, brain_regions):
    """
    Create a Laguerre Tesselation out of generator spheres.

    Args:
        points: (N, 3) array with generator points
        radii: (N,) array with generator radii
        brain_regions: VoxelData with brain region IDs (used as a mask)

    Returns:
        (regions, connectivity) pair, where:
            regions: ConvexPolyhedron for each region
            connectivity: list of region ID pairs indicating neighbouring regions

    See also:
        https://tess.readthedocs.io/en/latest/
    """
    cells = tess.Container(points, limits=brain_regions.bbox, radii=radii)

    regions = [
        ConvexPolyhedron(cell.vertices(), cell.face_vertices())
        for cell in cells
    ]

    connectivity = [
        (cell.id, neighbor)
        for cell in cells
        for neighbor in cell.neighbors() if neighbor >= 0
    ]

    return regions, connectivity


def overlap(tesselation, overlap_distr):
    """
    Overlap `tesselation` regions sampling scaling factors from `overlap_distr`.

    Args:
        tesselation: (regions, connectivity) pair obtained from `tesselate`
        overlap_distr: overlap factor distribution

    Returns:
        Tesselation with the overlapping domains and same connectivity as the input one.

    See also:
        *`tesselate`
        * https://bbpteam.epfl.ch/project/
                spaces/display/BBPNSE/Defining+distributions+in+config+files
    """
    regions, connectivity = tesselation

    factors = overlap_distr.rvs(size=len(regions))
    if np.any(factors < 0):
        raise BrainBuilderError("Negative overlap factor sampled")

    scales = np.cbrt(1.0 + factors)  # pylint: disable=assignment-from-no-return
    scaled_regions = [
        region.scale(scale)
        for region, scale in zip(regions, scales)
    ]

    return (scaled_regions, connectivity)


def export_structure(tesselation, filepath):
    """
    Write tesselation to HDF5 file.

    HDF file layout:
        /connectivity    # (M, 2) dataset with neighbourhood relation
        /cell_data
        --/cell_<i>
        -----/points     # (K_i, 3) dataset with corner vertices corresponding to cell <i> region
        -----/triangles  # (T_i, 3) dataset with face triangles (indices in `points`)

    Args:
        tesselation: (regions, connectivity) pair obtained from `tesselate`.
        filepath: path to output file

    See also:
        `tesselate`
    """
    regions, connectivity = tesselation
    with h5py.File(filepath, 'w') as h5f:
        for cell_id, region in enumerate(regions):
            group = h5f.create_group('/cell_data/cell_{:06d}'.format(cell_id))
            group.create_dataset('points', data=region.points)
            group.create_dataset('triangles', data=region.triangles)

        h5f.create_dataset('/connectivity', data=connectivity)


def load_structure(filepath):
    """
    Load tesselation from HDF5 file.

    See also:
        `export_structure`
    """
    with h5py.File(filepath, 'r') as h5f:
        connectivity = h5f['/connectivity'][:]
        cell_count = len(h5f['/cell_data'])
        regions = []
        for cell_id in range(cell_count):
            group = h5f['/cell_data/cell_{:06d}'.format(cell_id)]
            regions.append(
                ConvexPolyhedron(group['points'][:], group['triangles'][:])
            )
    return regions, connectivity


def export_meshes(tesselation, filepath):
    """
    Exports all the faces of tesselation cells to STL format.

    See also:
        https://en.wikipedia.org/wiki/STL_(file_format)
    """
    from stl.mesh import Mesh

    regions, _ = tesselation
    data = np.concatenate([r.face_points for r in regions])

    mesh = Mesh(np.zeros(data.shape[0], dtype=Mesh.dtype))
    mesh.vectors = data
    mesh.save(filepath)
