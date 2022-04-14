"""Piston sources."""

import numpy as _np

from .common import Source as _Source
from ..utils.linalg import convert_to_3n_array as _convert_to_3n_array
from ..utils.linalg import convert_to_unit_vector as _convert_to_unit_vector
from .transducers import incident_field as _incident_field


def create_piston(
    frequency,
    radius,
    source_axis=(1, 0, 0),
    number_of_point_sources_per_wavelength=6,
    location=(0, 0, 0),
    velocity=1.0,
):
    """
    Create a plane circular piston source.

    Parameters
    ----------
    frequency : float
        The frequency of the acoustic field.
    radius : float
        The radius of the piston.
    source_axis : array like
        The axis of the piston.
        Default: positive x direction
    number_of_point_sources_per_wavelength : integer
        The number of point sources per wavelength used to discretise the piston.
        Default: 6
    location : array like
        The location of the centroid of the piston.
        Default: global origin
    velocity : complex
        Normal velocity of the piston.
        Default : 1 m/s

    """
    return _Piston(
        frequency,
        radius,
        source_axis,
        number_of_point_sources_per_wavelength,
        location,
        velocity,
    )


class _Piston(_Source):
    def __init__(
        self,
        frequency,
        radius,
        source_axis,
        number_of_point_sources_per_wavelength,
        location,
        velocity,
    ):

        super().__init__("piston", frequency)

        if not isinstance(source_axis, (list, tuple, _np.ndarray)):
            raise TypeError("Piston source axis needs to be an array type.")
        direction_vector = _np.array(source_axis)
        if direction_vector.ndim == 1 and direction_vector.size == 3:
            self.source_axis = _convert_to_unit_vector(direction_vector)
        elif direction_vector.ndim == 2 and direction_vector.size == 3:
            self.source_axis = _convert_to_unit_vector(
                direction_vector.flatten()
            )
        else:
            raise ValueError("Source axis needs to be a 3D vector.")

        if not isinstance(number_of_point_sources_per_wavelength, int):
            raise TypeError(
                "Number of point sources per wavelength needs to be an integer."
            )
        else:
            if number_of_point_sources_per_wavelength < 0:
                raise ValueError(
                    "Number of point sources per wavelength needs to be a positive integer."
                )
            else:
                self.number_of_point_sources_per_wavelength = (
                    number_of_point_sources_per_wavelength
                )

        if not isinstance(location, (list, tuple, _np.ndarray)):
            raise TypeError("Piston location needs to be an array type.")
        location_vector = _np.array(location)
        if location_vector.ndim == 1 and location_vector.size == 3:
            self.location = location_vector
        elif location_vector.ndim == 2 and location_vector.size == 3:
            self.location = location_vector.flatten()
        else:
            raise ValueError("Piston location needs to be a 3D vector.")

        self.radius = radius

        self.velocity = _np.atleast_1d(complex(velocity))

    def pressure_field(self, locations, medium):
        """
        Calculate the pressure field in the specified locations.

        Parameters
        ----------
        locations : 3 x N array
            Locations on which to evaluate the pressure field.
        medium : optimus.material.Material
            The propagating medium.
        """

        points = _convert_to_3n_array(locations)
        incident_field = _incident_field(self, medium, points)
        pressure = incident_field.pressure

        return pressure

    def normal_pressure_gradient(self, locations, normals, medium):
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
        medium : optimus.material.Material
            The propagating medium.
        """

        points = _convert_to_3n_array(locations)
        normals = _convert_to_3n_array(normals)
        unit_normals = _convert_to_unit_vector(normals)

        incident_field = _incident_field(self, medium, points, unit_normals)
        gradient = incident_field.normal_pressure_gradient

        return gradient
