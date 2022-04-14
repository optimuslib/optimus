"""Common functionality for acoustic sources."""

import numpy as _np
import bempp.api as _bempp


class Source:
    def __init__(
        self,
        source_type,
        frequency,
    ):
        """
        The source of the acoustic wave.
        """

        if not isinstance(frequency, (int, float)):
            raise TypeError("Frequency should be specified as a number")
        elif frequency <= 0.0:
            raise ValueError("Frequency should have a positive value.")

        self.type = source_type
        self.frequency = frequency

    def pressure_field(self, locations, medium):
        """
        Calculate the pressure field in the specified locations.
        Needs to be overridden by specific source type.
        """
        raise NotImplementedError

    def normal_pressure_gradient(self, locations, normals, medium):
        """
        Calculate the normal gradient of the pressure field in the
         specified locations.
        Needs to be overridden by specific source type.
        """
        raise NotImplementedError

    def calc_surface_traces(self, medium, spaces):
        """
        Calculate the Dirichlet and Neumann traces of the source on a
         surface grid, where the incident field is given by explicit
         functions for the pressure and its normal gradient.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        spaces : tuple of bempp.api.FunctionSpace
            The discrete spaces on the surface grid.

        Returns
        ----------
        traces : tuple of bempp.api.GridFunctions
            The Dirichlet and Neumann traces.
        """

        def dirichlet_fun(x, n, domain_index, result):
            result[0] = self.pressure_field(x, medium)

        def neumann_fun(x, n, domain_index, result):
            result[0] = self.normal_pressure_gradient(x, n, medium)

        space_dirichlet = spaces[0]
        trace_dirichlet = _bempp.GridFunction(
            space_dirichlet, fun=dirichlet_fun
        )

        space_neumann = spaces[0]
        trace_neumann = _bempp.GridFunction(space_neumann, fun=neumann_fun)

        return trace_dirichlet, trace_neumann


def _convert_to_3n_array(array):
    """
    Convert the input array into a 3xN Numpy array, if possible.
    """

    if not isinstance(array, (tuple, list, _np.ndarray)):
        raise TypeError("Variable needs to be a tuple, list, or Numpy array.")

    array_np = _np.array(array)

    if array_np.ndim == 1:

        if array_np.size == 3:
            return array_np.reshape([3, 1])
        else:
            raise ValueError("Location needs to be three dimensional.")

    elif array_np.ndim == 2:

        if array_np.shape[0] == 3:
            return array_np
        elif array_np.shape[1] == 3:
            return array_np.transpose()
        else:
            raise ValueError("Locations needs to be three dimensional.")

    else:

        raise ValueError("Locations need to be three dimensional.")


def _convert_to_unit_vector(vector):
    """
    Convert a vector into a unit vector.
    For 2D input arrays, the columns will be normalized.
    """

    if not isinstance(vector, _np.ndarray):
        raise TypeError("Vector needs to be a Numpy array.")

    if vector.ndim == 1:
        return vector / _np.linalg.norm(vector)
    elif vector.ndim == 2:
        return vector / _np.linalg.norm(vector, axis=0)
    else:
        raise ValueError("Vector needs to be 1D or 2D.")
