#!/usr/bin/env python
"""
Build O1 atlas.
"""

import os
import argparse
import json

import numpy as np
from six import iteritems

from voxcell import build, math_utils, VoxelData

COLUMNS = range(7)
LAYERS = range(1, 7)


def _region_id(column, layer=None):
    """ Region ID corresponding to `layer` in `column`. """
    assert column < 10
    if layer is None:
        return 100 + 10 * column
    else:
        assert 0 < layer < 10
        return 100 + 10 * column + layer


def _compact(mask):
    """ Trim zero values on mask borders. """
    aabb = math_utils.minimum_aabb(mask)
    return math_utils.clip(mask, aabb)


def _build_O1_mosaic(hex_side, voxel_size):
    """ Build 2D matrix representing O1 mosaic. """
    hexagon = _compact(build.regular_convex_polygon_mask_from_side(hex_side, 6, voxel_size))
    w, h = hexagon.shape

    hex_center = np.array([
        [1, 1],
        [0, 2],
        [1, 3],
        [2, 2],
        [0, 4],
        [1, 5],
        [2, 4],
    ])
    shift = hex_center * (3 * w // 4, h // 2)

    shape = np.max(shift, axis=0) + (w, h)
    mosaic = np.full(shape, -1, dtype=np.int16)
    for column, (dx, dz) in enumerate(shift):
        mosaic[dx:dx + w, dz:dz + h][hexagon] = column

    offset = -0.5 * np.array([w, h]) * voxel_size
    return mosaic, offset


def _build_brain_regions(hex_side, layer_thickness, voxel_size):
    """ Build 'brain_regions' VoxelData for O1 atlas. """
    mosaic_2D, offset_2D = _build_O1_mosaic(hex_side, voxel_size)
    mosaic_3d_layers = []
    for layer in LAYERS:
        pattern = np.zeros_like(mosaic_2D, dtype=np.uint16)
        for column in COLUMNS:
            pattern[mosaic_2D == column] = _region_id(column, layer)
        mosaic_3d_layers.append(
            np.repeat([pattern], layer_thickness[layer - 1] // voxel_size, axis=0)
        )
    mosaic_3D = np.swapaxes(
        np.vstack(mosaic_3d_layers[::-1]),
        0, 1
    )
    offset_3D = (offset_2D[0], 0, offset_2D[1])
    return VoxelData(mosaic_3D, 3 * (voxel_size,), offset_3D).compact()


def _build_orientation(brain_regions):
    """ Build 'orientation' VoxelData for O1 atlas. """
    raw = np.zeros(brain_regions.raw.shape + (4,), dtype=np.int8)
    raw[:, :, :, 0] = 127
    return brain_regions.with_data(raw)


def _build_height(brain_regions):
    """ Build 'height' VoxelData for O1 atlas. """
    height = brain_regions.indices_to_positions(brain_regions.raw.shape)[1]
    raw = np.full_like(brain_regions.raw, height, dtype=np.float32)
    return brain_regions.with_data(raw)


def _build_distance(brain_regions):
    """ Build 'distance' VoxelData for O1 atlas. """
    raw = np.full_like(brain_regions.raw, 0.0, dtype=np.float32)
    voxel_size = brain_regions.voxel_dimensions[1]
    for j in range(brain_regions.raw.shape[1]):
        raw[:, j, :] = brain_regions.offset[1] + voxel_size * (0.5 + j)
    return brain_regions.with_data(raw)


def _build_relative_distance(brain_regions):
    """ Build 'relative_distance' VoxelData for O1 atlas. """
    raw = _build_distance(brain_regions).raw / _build_height(brain_regions).raw
    return brain_regions.with_data(raw)


def _column_hierarchy(column):
    """ Build 'hierarchy' dict for single hypercolumn. """
    return {
        'id': _region_id(column),
        'acronym': "mc%d_Column" % column,
        'name': "hypercolumn %d" % column,
        'children': [{
            'id': _region_id(column, layer),
            'acronym': "L%d" % layer,
            'name': "hypercolumn %d, layer %d" % (column, layer)
        } for layer in LAYERS]
    }


def _build_hierarchy():
    """ Build 'hierarchy' dict for O1 atlas. """
    return {
        'id': 65535,
        'acronym': "O1",
        'name': "O1 mosaic",
        'children': [
            _column_hierarchy(c) for c in COLUMNS
        ]
    }


def main(args):
    # pylint: disable=missing-docstring
    layer_thickness = [float(x) for x in args.layer_thickness.split(",")]
    assert len(layer_thickness) == len(LAYERS)

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    brain_regions = _build_brain_regions(args.hex_side, layer_thickness, args.voxel_size)
    datasets = {
        'brain_regions': brain_regions,
        'orientation': _build_orientation(brain_regions),
        'height': _build_height(brain_regions),
        'distance': _build_distance(brain_regions),
        'relative_distance': _build_relative_distance(brain_regions),
    }
    hierarchy = _build_hierarchy()

    for name, data in iteritems(datasets):
        data.save_nrrd(os.path.join(output_dir, name + ".nrrd"))

    with open(os.path.join(output_dir, "hierarchy.json"), "w") as f:
        json.dump(hierarchy, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build O1 atlas")
    parser.add_argument(
        '-a', '--hex-side',
        required=True,
        type=float,
        help="Hexagon side (um)"
    )
    parser.add_argument(
        '-l', '--layer-thickness',
        required=True,
        help="Comma-separated L1-L6 layers thickness (um)"
    )
    parser.add_argument(
        '-d', '--voxel-size',
        required=True,
        type=float,
        help="Voxel side (um)"
    )
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        help="Output dir path"
    )
    main(parser.parse_args())
