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

    def subdomain_count(self):
        """
        The number of subdomains in the geometry.
        """
        if isinstance(self.grid, list):
            return len(self.grid)
        else:
            return 1
