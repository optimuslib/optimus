"""Bowl sources."""

import numpy as _np

from .common import Source as _Source
from ..utils.conversions import convert_to_positive_int as _convert_to_positive_int
from ..utils.conversions import convert_to_positive_float as _convert_to_positive_float
from ..utils.conversions import convert_to_array as _convert_to_array
from ..utils.conversions import convert_to_3n_array as _convert_to_3n_array
from ..utils.linalg import normalize_vector as _normalize_vector
from .transducers import transducer_field as _transducer_field


def create_bowl(
    frequency,
    outer_radius,
    radius_of_curvature,
    source_axis=(1, 0, 0),
    number_of_point_sources_per_wavelength=6,
    location=(0, 0, 0),
    velocity=1.0,
    inner_radius=None,
):
    """
    Create a spherical section bowl source.

    Parameters
    ----------
    frequency : float
        The frequency of the acoustic field.
    outer_radius : float
        The outer radius of the spherical section bowl.
    radius_of_curvature : float
        The radius of curvature of the bowl.
    source_axis : tuple float
        The axis of the bowl.
        Default: positive x direction
    number_of_point_sources_per_wavelength : integer
        The number of point sources per wavelength used to discretise
        the bowl source.
        Default: 6
    location : tuple float
        The location of the centroid of the bowl.
        Default: global origin
    velocity : complex
        Normal velocity of the bowl.
        Default : 1 m/s
    inner_radius : float, None
        The radius of the bowl aperture.
        Default: None
    """

    return _Bowl(
        frequency,
        outer_radius,
        radius_of_curvature,
        source_axis,
        number_of_point_sources_per_wavelength,
        location,
        velocity,
        inner_radius,
    )


class _Bowl(_Source):
    def __init__(
        self,
        frequency,
        outer_radius,
        radius_of_curvature,
        source_axis,
        number_of_point_sources_per_wavelength,
        location,
        velocity,
        inner_radius,
    ):

        super().__init__("bowl", frequency)

        source_axis_vector = _convert_to_array(
            source_axis, shape=(3,), label="bowl source axis"
        )
        self.source_axis = _normalize_vector(source_axis_vector)

        self.number_of_point_sources_per_wavelength = _convert_to_positive_int(
            number_of_point_sources_per_wavelength,
            label="number of point sources per wavelength",
        )

        self.location = _convert_to_array(location, shape=(3,), label="bowl location")

        if inner_radius:
            self.inner_radius = _convert_to_positive_float(inner_radius, "inner radius")
        else:
            self.inner_radius = None

        self.outer_radius = _convert_to_positive_float(outer_radius)

        self.radius_of_curvature = _convert_to_positive_float(radius_of_curvature)

        self.velocity = _np.atleast_1d(complex(velocity))

    def pressure_field(self, medium, locations):
        """Calculate the pressure field in the specified locations.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        locations : numpy.ndarray
            Array of size (3,N) with the locations on which to evaluate
            the pressure field.

        Returns
        -------
        pressure : numpy.ndarray
            Array of size (N,) with the pressure in the locations.
        """

        points = _convert_to_3n_array(locations)
        incident_field = _transducer_field(self, medium, points)
        pressure = incident_field.pressure

        return pressure

    def normal_pressure_gradient(self, locations, normals, medium):
        """Calculate the normal gradient of the pressure field in the
        specified locations.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        locations : numpy.ndarray
            Array of size (3,N) with the locations on which to evaluate the pressure field.
        normals : numpy.ndarray
            Array of size (3,N) with the unit normal vectors at the locations on which to evaluate the pressure field.

        Returns
        -------
        gradient : numpy.ndarray
            Array of size (3,N) with the normal gradient of the pressure
            in the locations.
        """

        points = _convert_to_3n_array(locations)
        normals = _convert_to_3n_array(normals)
        unit_normals = _normalize_vector(normals)

        incident_field = _transducer_field(self, medium, points, unit_normals)
        gradient = incident_field.normal_pressure_gradient

        return gradient

    def pressure_field_and_normal_gradient(self, medium, locations, normals):
        """Calculate the pressure field and the normal gradient of the pressure
        field in the specified locations.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        locations : numpy.ndarray
            An array of size (3,N) with the locations on which to evaluate the pressure field.
        normals : numpy.ndarray
            An array of size (3,N) with the unit normal vectors at the locations on which to evaluate the pressure field.

        Returns
        -------
        pressure : numpy.ndarray
            An array of size (N,) with the pressure in the locations.
        gradient : numpy.ndarray
            An array of size (3,N) with the normal gradient of the pressure in the locations.
        """

        points = _convert_to_3n_array(locations)
        normals = _convert_to_3n_array(normals)
        unit_normals = _normalize_vector(normals)

        incident_field = _transducer_field(self, medium, points, unit_normals)
        pressure = incident_field.pressure
        gradient = incident_field.normal_pressure_gradient

        return pressure, gradient

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

        return super()._calc_surface_traces_from_coefficients(
            medium,
            space_dirichlet,
            space_neumann,
            dirichlet_trace,
            neumann_trace,
        )
