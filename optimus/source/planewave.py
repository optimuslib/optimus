"""Plane wave sources."""

import numpy as _np
from .common import Source as _Source
from .common import _convert_to_3n_array
from .common import _convert_to_unit_vector


def create_planewave(frequency, direction=None, amplitude=1.0):
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
        Default: 1
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

        if direction is not None:
            if not isinstance(direction, (list, tuple, _np.ndarray)):
                raise TypeError("Wave direction needs to be an array type.")
            direction_vector = _np.array(direction)
            if direction_vector.ndim == 1 and direction_vector.size == 3:
                self.direction_vector = _convert_to_unit_vector(
                    direction_vector
                )
            elif direction_vector.ndim == 2 and direction_vector.size == 3:
                self.direction_vector = _convert_to_unit_vector(
                    direction_vector.flatten()
                )
            else:
                raise ValueError("Wave direction needs to be a 3D vector.")
        else:
            self.direction_vector = _np.array([1, 0, 0])

        if not isinstance(amplitude, (int, float)):
            raise TypeError("Wave amplitude should be a number.")
        else:
            self.amplitude = amplitude

    def pressure_field(self, locations, medium):
        """
        Calculate the pressure field in the specified locations.
        Parameters
        ----------
        locations : 3 x N array
            Locations on which to evaluate the pressure field.
        medium : class
            The exterior medium properties.
        """

        points = _convert_to_3n_array(locations)
        wavenumber = medium.wavenumber(self.frequency)

        pressure = self.amplitude * _np.exp(
            1j * wavenumber * _np.dot(self.direction_vector, points)
        )

        return pressure

    def normal_pressure_gradient(self, locations, medium, normals):
        """
        Calculate the normal gradient of the pressure field in the
         specified locations.
        Parameters
        ----------
        locations : 3 x N array
            Locations on which to evaluate the pressure field.
        normals : 3 x N array
            Unit normal vectors at the locations on which to evaluate the
             pressure field.
        medium : class
            The exterior medium properties.
        """

        points = _convert_to_3n_array(locations)
        wavenumber = medium.wavenumber(self.frequency)
        normals = _convert_to_3n_array(normals)
        unit_normals = _convert_to_unit_vector(normals)

        normals = _np.dot(self.direction_vector, unit_normals)
        points = _np.dot(self.direction_vector, points)
        gradient = (
            (self.amplitude * 1j * wavenumber)
            * normals
            * _np.exp(1j * wavenumber * points)
        )

        return gradient
