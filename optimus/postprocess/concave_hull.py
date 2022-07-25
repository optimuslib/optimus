from scipy.spatial import Delaunay
import numpy as _np


def concave_hull(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.

    Parameters
    ----------
    points : numpy ndarray of size (2,N)
        Array of shape (2,N) with the points.
    alpha : float
        The alpha value.
    only_outer : bool
        Specify if we keep only the outer border or also inner edges.

    Returns
    ----------
    edges : set
        The set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edge, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edge or (j, i) in edge:
            # already added
            assert (j, i) in edge, "Can't go twice over same directed edge..."
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edge.remove((j, i))
            return
        edge.add((i, j))

    triangles = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, vertex_b_idx, vertex_c_idx = indices of corner points of the triangle
    for vertex_a_idx, vertex_b_idx, vertex_c_idx in triangles.vertices:
        vertex_a = points[vertex_a_idx]
        vertex_b = points[vertex_b_idx]
        vertex_c = points[vertex_c_idx]
        # Computing radius of triangle circumcircle
        side_a = _np.sqrt(
            (vertex_a[0] - vertex_b[0]) ** 2 + (vertex_a[1] - vertex_b[1]) ** 2
        )
        side_b = _np.sqrt(
            (vertex_b[0] - vertex_c[0]) ** 2 + (vertex_b[1] - vertex_c[1]) ** 2
        )
        side_c = _np.sqrt(
            (vertex_c[0] - vertex_a[0]) ** 2 + (vertex_c[1] - vertex_a[1]) ** 2
        )
        coeff = (side_a + side_b + side_c) / 2.0
        dist = coeff * (coeff - side_a) * (coeff - side_b) * (coeff - side_c)

        if (dist < 0).any():
            dist = 0
        area = _np.sqrt(dist)

        if area > 0:
            circum_radius = side_a * side_b * side_c / (4.0 * area)
        else:
            circum_radius = _np.inf

        if circum_radius < alpha:
            add_edge(edges, vertex_a_idx, vertex_b_idx)
            add_edge(edges, vertex_b_idx, vertex_c_idx)
            add_edge(edges, vertex_c_idx, vertex_a_idx)
    return edges
