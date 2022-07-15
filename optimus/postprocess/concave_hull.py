from scipy.spatial import Delaunay
import numpy as np


def concave_hull(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (2,n) with the points.
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
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in triangles.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges
