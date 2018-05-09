import os
import contextlib
import shutil
import tempfile

import nose.tools as nt

from mock import Mock, patch

import numpy as np
import numpy.testing as npt

from brainbuilder import BrainBuilderError
from brainbuilder.geometry import ConvexPolyhedron

import brainbuilder.ngv.microdomains as test_module


def test_tesselation_1():
    brain_regions = Mock()
    brain_regions.bbox = [[0., 0., 0.], [4., 4., 4.]]
    regions, connectivity = test_module.tesselate(
        points=[[1., 1., 1.], [2., 2., 2.]],
        radii=[1., 2.],
        brain_regions=brain_regions
    )
    nt.assert_equals(len(regions), 2)
    region = regions[0]
    npt.assert_equal(
        region.points,
        [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 3.0), (0.0, 3.0, 0.0)]
    )
    npt.assert_equal(
        region.triangles,
        [(3, 2, 1), (2, 0, 1), (0, 3, 1), (3, 0, 2)]
    )
    nt.assert_equals(
        set(connectivity),
        set([(0, 1), (1, 0)])
    )


def test_overlap_1():
    regions = [
        Mock(), Mock()
    ]
    connectivity = [(0, 1), (1, 0)]
    original = (regions, connectivity)

    distr = Mock()
    distr.rvs.return_value = np.array([2., 4.]) ** 3 - 1
    regions2, connectivity2 = test_module.overlap(original, distr)

    regions[0].scale.assert_called_with(2.0)
    regions[1].scale.assert_called_with(4.0)

    nt.assert_equal(connectivity, connectivity2)


@nt.raises(BrainBuilderError)
def test_overlap_2():
    original = ([None], None)

    distr = Mock()
    distr.rvs.return_value = np.array([-1])
    regions2, connectivity2 = test_module.overlap(original, distr)


@contextlib.contextmanager
def tmpdir():
    dirpath = tempfile.mkdtemp()
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


@patch.object(ConvexPolyhedron, '_ensure_outward_normals')
def test_export_load_structure(_):
    regions = [
        ConvexPolyhedron(
            [[0., 0., 0.], ],
            [[0, 0, 0]]
        ),
        ConvexPolyhedron(
            [[1., 1., 1.], [2., 2., 2.]],
            [[0, 1, 0], [0, 0, 1]]
        ),
    ]
    connectivity = [(0, 1), (1, 0)]
    original = (regions, connectivity)

    with tmpdir() as dirpath:
        filepath = os.path.join(dirpath, 'structure.h5')
        test_module.export_structure(original, filepath)
        loaded = test_module.load_structure(filepath)

    for r0, r1 in zip(original[0], loaded[0]):
        npt.assert_equal(r0.points, r1.points)
        npt.assert_equal(r0.triangles, r1.triangles)

    npt.assert_equal(original[1], loaded[1])


@patch.object(ConvexPolyhedron, '_ensure_outward_normals')
def test_export_meshes(_):
    from stl.mesh import Mesh
    regions = [
        ConvexPolyhedron(
            [[0., 0., 0.], ],
            [[0, 0, 0]]
        ),
        ConvexPolyhedron(
            [[1., 1., 1.], [2., 2., 2.]],
            [[0, 1, 0], [0, 0, 1]]
        ),
    ]
    connectivity = [(0, 1), (1, 0)]
    original = (regions, connectivity)

    with tmpdir() as dirpath:
        filepath = os.path.join(dirpath, 'meshes.stl')
        test_module.export_meshes(original, filepath)
        loaded = Mesh.from_file(filepath)

    npt.assert_equal(
        loaded.vectors,
        [
            [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
            [[1., 1., 1.], [2., 2., 2.], [1., 1., 1.]],
            [[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]]
        ]
    )
