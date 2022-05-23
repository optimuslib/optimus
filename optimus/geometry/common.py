"""Common functionality for geometries."""


class Geometry:
    def __init__(self, grid):
        """
        Create a geometry.

        Parameters
        ----------
        grid : bempp.api.grid
            Surface grid of the geometry.
        """

        self.grid = grid

    def number_of_vertices(self):
        """
        The number of vertices in the surface mesh of the geometry.
        """
        return self.grid.leaf_view.vertices.shape[1]
