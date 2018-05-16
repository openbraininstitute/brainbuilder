""" ConvexPolyhedron class. """

import itertools

import numpy as np


class ConvexPolyhedron(object):
    """
    Class representing a 3D convex solid.

    Face surfaces are triangulated.

    Internally we store
        * (N, 3) array of vertices after triangulation
        * (M, 3) array of vertex indices defining triangular faces

    For each triangle, vertices are ordered in a way that face normals point outward.

    NB: we do not verify convexity assumption for the moment.
    """

    def __init__(self, points, face_vertices):
        self.points = np.array(points)
        self.triangles = np.array(_triangulate(face_vertices))
        self._ensure_outward_normals()

    def _ensure_outward_normals(self):
        """
        Ensure `face_vectors` and `face_normals` will return vectors pointing outwards.

        In order to do that:
            * construct vectors pointing outwards from centroid towards faces
            * calculate row-wise dot product between these vectors and current `face_vectors`
            * rows with negative dot product correspond to `face_vectors` pointing inwards
            * for these triangles change the order of vertices to flip `face_vectors`
        """
        outwards = self.face_centers - self.centroid
        to_flip = np.einsum('ij,ij->i', self.face_vectors, outwards) < 0
        self.triangles[to_flip] = np.fliplr(self.triangles[to_flip])

    @property
    def face_points(self):
        """ XYZ-coordinates for face. """
        return self.points[self.triangles]

    @property
    def face_vectors(self):
        """ Cross-products of vectors defining each face. """
        pts = self.face_points
        return np.cross(
            pts[:, 1, :] - pts[:, 0, :],
            pts[:, 2, :] - pts[:, 0, :]
        )

    @property
    def face_normals(self):
        """ Face normals. """
        vv = self.face_vectors
        return vv / np.linalg.norm(vv, axis=1)[:, np.newaxis]

    @property
    def face_areas(self):
        """ Face areas. """
        return 0.5 * np.linalg.norm(self.face_vectors, axis=1)

    @property
    def face_centers(self):
        """ Centers of triangle faces. """
        return np.mean(self.face_points, axis=1)

    @property
    def centroid(self):
        """ Centroid weighted by the area of the triangle faces. """
        return np.average(self.face_centers, weights=self.face_areas, axis=0)

    def scale(self, alpha):
        """ Construct a new polyhedron by scaling each vertex distance from centroid. """
        c0 = self.centroid
        points = c0 + alpha * (self.points - c0)
        return ConvexPolyhedron(points, self.triangles[:])


def _flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


def _iter_polygon_triangles(vertices):
    n = len(vertices)
    if n < 3:
        raise ValueError("Invalid polygon of %d vertices" % n)
    for i in range(2, n):
        yield (vertices[0], vertices[i - 1], vertices[i])


def _triangulate(vertices):
    return np.array(list(_flatmap(_iter_polygon_triangles, vertices)))
