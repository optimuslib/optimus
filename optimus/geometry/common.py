"""Common functionality for geometries."""

import bempp.api as _bempp


class Geometry:
    def __init__(self, grid, label="none"):
        """
        Create a geometry.

        Parameters
        ----------
        grid : bempp.api.Grid
            Surface grid of the geometry.
        label : str
            The name of the geometry.
        """

        self.grid = grid
        self.label = label

    def number_of_vertices(self):
        """
        The number of vertices in the surface mesh of the geometry.
        """
        return self.grid.leaf_view.vertices.shape[1]

    def export_mesh(self, filename=None):
        """
        Export the mesh in GMSH format.

        Parameters
        ----------
        filename : str
            The name of the file for export.
        """

        if filename is None:
            filename = "grid_" + self.label + ".msh"
        elif isinstance(filename, str):
            if filename[-4:] != ".msh":
                filename += ".msh"
        else:
            raise TypeError("The file name to export the grid needs to be a string.")

        _bempp.export(filename, grid=self.grid)


class ImportedGeometry(Geometry):
    def __init__(self, bempp_grid, label, filename=None):
        """
        An imported geometry.

        Parameters
        ----------
        bempp_grid : bempp.api.Grid
            Surface grid of the geometry.
        label : str
            The label of the geometry.
        filename : str, None
            The file name of the geometry, if specified.
        """
        super().__init__(bempp_grid, label)
        self.filename = filename
