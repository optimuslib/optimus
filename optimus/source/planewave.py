"""Plane wave sources."""

import numpy as _np

from .common import Source as _Source
from ..utils.linalg import convert_to_3n_array as _convert_to_3n_array
from ..utils.linalg import convert_to_unit_vector as _convert_to_unit_vector


def create_planewave(frequency, direction=(1, 0, 0), amplitude=1.0):
    """
    Create a plane wave source.

    Parameters
    ----------
    frequency : float
        The frequency of the acoustic field.
    direction : array like
        The direction of the plane wave.
        Default: positive x direction
    amplitude : float
        The amplitude of the plane wave.
        Default: 1 Pa
    """
    return _PlaneWave(frequency, direction, amplitude)


class _PlaneWave(_Source):
    def __init__(
        self,
        frequency,
        direction,
        amplitude,
    ):

        super().__init__("planewave", frequency)

        if not isinstance(direction, (list, tuple, _np.ndarray)):
            raise TypeError("Wave direction needs to be an array type.")
        direction_vector = _np.array(direction, dtype=float)
        if direction_vector.ndim == 1 and direction_vector.size == 3:
            self.direction_vector = _convert_to_unit_vector(direction_vector)
        elif direction_vector.ndim == 2 and direction_vector.size == 3:
            self.direction_vector = _convert_to_unit_vector(direction_vector.flatten())
        else:
            raise ValueError("Wave direction needs to be a 3D vector.")

        if not isinstance(amplitude, (int, float)):
            raise TypeError("Wave amplitude should be a number.")
        else:
            self.amplitude = float(amplitude)

    def pressure_field(self, medium, locations):
        """
        Calculate the pressure field in the specified locations.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        locations : 3 x N array
            Locations on which to evaluate the pressure field.

        Returns
        ----------
        pressure : N array
            The pressure in the locations.
        """

        points = _convert_to_3n_array(locations)
        wavenumber = medium.wavenumber(self.frequency)

        pressure = self.amplitude * _np.exp(
            1j * wavenumber * _np.dot(self.direction_vector, points)
        )

        return pressure

    def normal_pressure_gradient(self, medium, locations, normals):
        """
        Calculate the normal gradient of the pressure field in the
         specified locations.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        locations : 3 x N array
            Locations on which to evaluate the pressure field.
        normals : 3 x N array
            Unit normal vectors at the locations on which to evaluate the
             pressure field.

        Returns
        ----------
        gradient : 3 x N array
            The normal gradient of the pressure in the locations.
        """

        points = _convert_to_3n_array(locations)
        normals = _convert_to_3n_array(normals)
        unit_normals = _convert_to_unit_vector(normals)
        wavenumber = medium.wavenumber(self.frequency)

        normals = _np.dot(self.direction_vector, unit_normals)
        points = _np.dot(self.direction_vector, points)
        gradient = (
            (self.amplitude * 1j * wavenumber)
            * normals
            * _np.exp(1j * wavenumber * points)
        )

        return gradient

    def calc_surface_traces(
        self,
        medium,
        space_dirichlet=None,
        space_neumann=None,
        dirichlet_trace=True,
        neumann_trace=True,
    ):
        """
        Calculate the surface traces of the source field on the mesh.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        space_dirichlet, space_neumann : bempp.api.FunctionSpace
            The discrete spaces on the surface grid.
        dirichlet_trace, neumann_trace : bool
            Calculate the Dirichlet or Neumann trace of the field.

        Returns
        ----------
        trace : bempp.api.GridFunctions
            The surface traces.
        """
        return super()._calc_surface_traces_from_function(
            medium,
            space_dirichlet,
            space_neumann,
            dirichlet_trace,
            neumann_trace,
        )
