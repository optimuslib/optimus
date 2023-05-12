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
        filename : str, None
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

    def _correct_elements_group_bempp_grids(self):
        """
        To add mesh group to geometries created by bempp.
        """
        import os

        self.export_mesh("TEMP.msh")
        self.grid = _bempp.import_grid("TEMP.msh")
        os.remove("TEMP.msh")

    def scale_grid(self, scaling_factor):
        """
        Scale the entire grid with a multiplicative factor.

        Parameters
        ----------
        scaling_factor : float
            The scaling factor for the grid.
        """
        from ..utils.conversions import convert_to_float

        scaling = convert_to_float(scaling_factor, "scaling factor")
        vertices_new = self.grid.leaf_view.vertices * scaling

        self.grid = _bempp.grid.grid_from_element_data(
            vertices_new,
            self.grid.leaf_view.elements,
            self.grid.leaf_view.domain_indices,
        )

    def translate_grid(self, translation_vector):
        """
        Translate the entire grid with an additive vector.

        Parameters
        ----------
        translation_vector : numpy.ndarray
            A 3D vector to translate the grid.
        """
        from ..utils.conversions import convert_to_array

        translation = convert_to_array(translation_vector, (3, 1), "translation vector")
        vertices_new = self.grid.leaf_view.vertices + translation

        self.grid = _bempp.grid.grid_from_element_data(
            vertices_new,
            self.grid.leaf_view.elements,
            self.grid.leaf_view.domain_indices,
        )


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
