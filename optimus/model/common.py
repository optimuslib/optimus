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
        self.formulation = formulation
        self.preconditioner = preconditioner

        _check_validity_formulation(self.formulation, self.preconditioner)

    def solve(self):
        """
        Solve the model.
        Needs to be overwritten by specific model.
        """
        raise NotImplementedError


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
