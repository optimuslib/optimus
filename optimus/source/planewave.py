"""Plane wave sources."""

import numpy as _np

from .common import Source as _Source
from ..utils.conversions import convert_to_float as _convert_to_float
from ..utils.conversions import convert_to_array as _convert_to_array
from ..utils.conversions import convert_to_3n_array as _convert_to_3n_array
from ..utils.linalg import normalize_vector as _normalize_vector


def create_planewave(frequency, direction=(1, 0, 0), amplitude=1.0):
    """
    Create a plane wave source.

    Parameters
    ----------
    frequency : float
        The frequency of the acoustic field.
    direction : tuple[fooat]
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

        direction_vector = _convert_to_array(
            direction, shape=(3,), label="wave direction"
        )
        self.direction_vector = _normalize_vector(direction_vector)

        self.amplitude = _convert_to_float(amplitude, label="wave amplitude")

    def pressure_field(self, medium, locations):
        """
        Calculate the pressure field in the specified locations.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        locations : numpy.ndarray
            An array of size (3,N) with the locations on which to evaluate
            the pressure field.

        Returns
        ----------
        pressure : np.ndarray
            An array of size (N,) with the pressure in the locations.
        """

        points = _convert_to_3n_array(locations)
        wavenumber = medium.compute_wavenumber(self.frequency)

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
        locations : numpy.ndarray
            An array of size (3,N) with the locations on which to evaluate
            the pressure field.
        normals : numpy.ndarray
            An array of size (3,N) with the unit normal vectors at the locations
            on which to evaluate the pressure field.

        Returns
        ----------
        gradient : numpy.ndarray
            An array of size (3,N) with the normal gradient of the pressure
            in the locations.
        """

        points = _convert_to_3n_array(locations)
        normals = _convert_to_3n_array(normals)
        unit_normals = _normalize_vector(normals)
        wavenumber = medium.compute_wavenumber(self.frequency)

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
