import unittest
import nose.tools as nt

import numpy as np
import numpy.testing as npt

import brainbuilder.geometry.polyhedron as test_module


class TestPolyhedron(unittest.TestCase):
    def setUp(self):
        points = [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ]
        triangles = [
            [3, 1, 0],
            [2, 1, 0],  # would be reordered to ensure outward normal
            [2, 3, 0],
            [2, 3, 1],  # would be reordered to ensure outward normal
        ]
        self.test_obj = test_module.ConvexPolyhedron(points, triangles)

    def test_face_points(self):
        expected = [
            [[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]],  # reordered
            [[0., 0., 1.], [0., 1., 0.], [0., 0., 0.]],
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],  # reordered
        ]
        npt.assert_equal(self.test_obj.face_points, expected)

    def test_face_vectors(self):
        expected = [
            [ 0.,  0., -1.],
            [ 0., -1.,  0.],
            [-1.,  0.,  0.],
            [ 1.,  1.,  1.],
        ]
        npt.assert_equal(self.test_obj.face_vectors, expected)

    def test_face_normals(self):
        expected = [
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            [0.57735027,  0.57735027,  0.57735027]
        ]
        npt.assert_almost_equal(self.test_obj.face_normals, expected)

    def test_face_centers(self):
        expected = np.array([
            [1., 1., 0.,],
            [1., 0., 1.,],
            [0., 1., 1.,],
            [1., 1., 1.,],
        ]) / 3.0
        npt.assert_almost_equal(self.test_obj.face_centers, expected)

    def test_face_areas(self):
        expected = [0.5, 0.5, 0.5, np.sqrt(3) / 2]
        npt.assert_almost_equal(self.test_obj.face_areas, expected)

    def test_centroid(self):
        expected = [0.26289171, 0.26289171, 0.26289171]
        npt.assert_almost_equal(self.test_obj.centroid, expected)

    def test_scale(self):
        scaled = self.test_obj.scale(2.0)
        npt.assert_almost_equal(
            scaled.points, 2 * self.test_obj.points - 0.26289171
        )
        npt.assert_equal(
            scaled.triangles, self.test_obj.triangles
        )


def test_flatmap():
    actual = list(test_module._flatmap(lambda x: [x, x + 1], [1, 3]))
    nt.assert_equals(actual, [1, 2, 3, 4])


def test_iter_polygon_triangles_1():
    actual = list(test_module._iter_polygon_triangles([0, 1, 2]))
    nt.assert_equals(actual, [(0, 1, 2)])


def test_iter_polygon_triangles_2():
    actual = list(test_module._iter_polygon_triangles([0, 1, 2, 3, 4]))
    nt.assert_equals(actual, [(0, 1, 2), (0, 2, 3), (0, 3, 4)])


@nt.raises(ValueError)
def test_iter_polygon_triangles_3():
    actual = list(test_module._iter_polygon_triangles([0, 1]))
