"""Point sources."""

import numpy as _np

from .common import Source as _Source
from ..utils.conversions import convert_to_float as _convert_to_float
from ..utils.conversions import convert_to_array as _convert_to_array
from ..utils.conversions import convert_to_3n_array as _convert_to_3n_array
from ..utils.linalg import normalize_vector as _normalize_vector


def create_pointsource(frequency, location=(0, 0, 0), amplitude=1.0):
    """
    Create a point source.

    Parameters
    ----------
    frequency : float
        The frequency of the acoustic field.
    location : numpy.ndarray, tuple float, list float
        The location of the point source.
        Default: (0, 0, 0)
    amplitude : float
        The amplitude of the point source.
        Default: 1 Pa at 1 metre
    """

    return _PointSource(frequency, location, amplitude)


class _PointSource(_Source):
    def __init__(
        self,
        frequency,
        location,
        amplitude,
    ):
        super().__init__("pointsource", frequency)

        self.location = _convert_to_array(
            location, shape=(3,), label="point source location"
        )

        self.amplitude = _convert_to_float(amplitude, label="point source amplitude")

    def pressure_field(self, medium, locations):
        """
        Calculate the pressure field in the specified locations.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        locations : numpy.ndarray
            An array of size (3,N) with the locations on which to evaluate the pressure field.

        Returns
        -------
        pressure : np.ndarray
            An array of size (N,) with the pressure in the locations.
        """

        points = _convert_to_3n_array(locations)
        wavenumber = medium.compute_wavenumber(self.frequency)

        point_source_location = _convert_to_3n_array(self.location)
        differences_between_all_points = point_source_location - points

        # differences_between_all_points = (
        #     self.location[:, _np.newaxis, :] - points[:, :, _np.newaxis]
        # )
        distances_between_all_points = _np.linalg.norm(
            differences_between_all_points, axis=0
        )

        pressure = (
            self.amplitude
            * _np.exp(1j * wavenumber * distances_between_all_points)
            / distances_between_all_points
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
        -------
        gradient : numpy.ndarray
            An array of size (3,N) with the normal gradient of the pressure
            in the locations.
        """

        points = _convert_to_3n_array(locations)
        normals = _convert_to_3n_array(normals)
        wavenumber = medium.compute_wavenumber(self.frequency)

        point_source_location = _convert_to_3n_array(self.location)
        differences_between_all_points = point_source_location - points

        distances_between_all_points = _np.linalg.norm(
            differences_between_all_points, axis=0
        )

        greens_function_scaled = (
            _np.exp(1j * wavenumber * distances_between_all_points)
            / distances_between_all_points
        )

        greens_gradient_amplitude_scaled = _np.divide(
            greens_function_scaled * (wavenumber * distances_between_all_points + 1j),
            distances_between_all_points**2,
        )
        greens_gradient_scaled = (
            differences_between_all_points * greens_gradient_amplitude_scaled
        )
        greens_gradient = -1j * self.amplitude * greens_gradient_scaled

        gradient = _np.sum(greens_gradient * normals, axis=0)

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
        -------
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
