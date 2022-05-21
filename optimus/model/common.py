"""Common functionality for the models."""

import bempp.api as _bempp
import numpy as _np


class Model:
    def __init__(
        self,
        source,
        geometry,
        material_exterior,
        material_interior,
        formulation,
        preconditioner,
    ):
        """
        Base class for BEM models.
        """

        self.source = source
        self.geometry = geometry
        self.material_exterior = material_exterior
        self.material_interior = material_interior
        self.n_subdomains = self._preprocess_domains()

        _check_validity_formulation(formulation, preconditioner)
        self.formulation = formulation
        self.preconditioner = preconditioner

    def solve(self):
        """
        Solve the model.
        Needs to be overwritten by specific model.
        """
        raise NotImplementedError

    def _preprocess_domains(self):
        """
        Preprocess the input variables for the geometry and materials.

        Returns
        ----------
        n_bounded_domains : int
            The number of bounded subdomains.
        """

        from optimus.geometry.common import Geometry
        from optimus.material.common import Material

        if not isinstance(self.geometry, (list, tuple)):
            self.geometry = (self.geometry,)
        for subdomain in self.geometry:
            if not isinstance(subdomain, Geometry):
                raise TypeError(
                    "The subdomain needs to be specified as an "
                    "Optimus Geometry object."
                )
        n_bounded_domains = len(self.geometry)

        if not isinstance(self.material_interior, (list, tuple)):
            self.material_interior = (self.material_interior,)
        for material in self.material_interior:
            if not isinstance(material, Material):
                raise TypeError(
                    "The material needs to be specified as an Optimus Material object."
                )
        if len(self.material_interior) != n_bounded_domains:
            raise ValueError(
                "The number of geometries and interior materials should be the same."
            )

        return n_bounded_domains


def _vector_to_gridfunction(vector, spaces):
    """
    Convert an array with values representing the coefficients
    of one or several discrete spaces into Bempp grid functions.

    Parameters
    ----------
    vector : np.ndarray
        Vector of coefficients.
    spaces : tuple of bempp.api.FunctionSpace
        The function spaces.

    Returns
    ----------
    gridfunctions : tuple of bempp.api.GridFunction
        The list of grid functions.
    """

    ndofs = [space.global_dof_count for space in spaces]
    partitioning = _np.cumsum(ndofs)
    subvectors = _np.split(vector, partitioning[:-1])
    gridfunctions = [
        _bempp.GridFunction(space, coefficients=vec)
        for vec, space in zip(subvectors, spaces)
    ]

    return gridfunctions


def _check_validity_formulation(formulation, preconditioner):
    """
    Check if the specified formulation and preconditioner is correct.

    Parameters
    ----------
    formulation : str
        Name of the boundary integral formulation.
    preconditioner : str
        Name of the preconditioner.
    """

    if formulation not in ["pmchwt"]:
        raise NotImplementedError("Unknown boundary integral formulation type.")

    if preconditioner not in ["mass"]:
        raise NotImplementedError("Unknown preconditioner type.")
