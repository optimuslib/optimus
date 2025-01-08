"""Array sources."""

import numpy as _np

from .common import Source as _Source
from ..utils.conversions import convert_to_positive_int as _convert_to_positive_int
from ..utils.conversions import convert_to_positive_float as _convert_to_positive_float
from ..utils.conversions import convert_to_array as _convert_to_array
from ..utils.conversions import convert_to_3n_array as _convert_to_3n_array
from ..utils.conversions import convert_to_complex_array as _convert_to_complex_array
from ..utils.linalg import normalize_vector as _normalize_vector
from .transducers import transducer_field as _transducer_field


def create_array(
    frequency,
    element_radius=None,
    element_width=None,
    element_height=None,
    element_shape="circular",
    velocity=1.0,
    source_axis=(1, 0, 0),
    number_of_point_sources_per_wavelength=6,
    location=(0, 0, 0),
    centroid_locations=None,
    centroid_locations_filename=None,
    array_type="spherical",
):
    """Create an array source consisting of circular piston elements distributed on a
    spherical section bowl or on a plane.

    Parameters
    ----------
    frequency : float
        The frequency of the acoustic field.
    element_radius : float
        The radius of elements which lie on the spherical section bowl.
    velocity : complex, numpy.ndarray complex
        Array of size (N,) with complex values for the normal velocities of the array elements.
        If one value is specified, this will be repeated for all array elements. Default : 1 m/s
    source_axis : tuple float
        The direction vector of the axis of the bowl. Default: positive x direction
    number_of_point_sources_per_wavelength : integer
        The number of point sources per wavelength used to discretise each piston. Default: 6
    location : tuple float
        The location of the centroid of the bowl. Default: global origin
    centroid_locations : numpy.ndarray
        An array of size (3, N) with the locations of the centroids of the piston elements.
        For a spherical type array, these must be specified in a local coordinate sytem where
        the axis of the transducer is the Cartesian positive z-axis and the focus of the transducer
        is (0,0,0).
        For a planar array, the locations of the centroids of the piston elements must be specified
        in the z=0 plane, whereby the axis of the transducer is the Cartesian positive z-axis.
    centroid_locations_filename : str
        Path and filename containing the centroid locations data. The file extension has to be ".dat".
    array_type : str
        Type of array considered. Can be either planar or spherical.
    """

    return _Array(
        frequency,
        element_radius,
        element_width,
        element_height,
        element_shape,
        velocity,
        source_axis,
        number_of_point_sources_per_wavelength,
        location,
        centroid_locations,
        centroid_locations_filename,
        array_type,
    )


class _Array(_Source):
    def __init__(
        self,
        frequency,
        element_radius,
        element_width,
        element_height,
        element_shape,
        velocity,
        source_axis,
        number_of_point_sources_per_wavelength,
        location,
        centroid_locations,
        centroid_locations_filename,
        array_type,
    ):
        super().__init__("array", frequency)

        source_axis_vector = _convert_to_array(
            source_axis, shape=(3,), label="array source axis"
        )
        self.source_axis = _normalize_vector(source_axis_vector)

        self.number_of_point_sources_per_wavelength = _convert_to_positive_int(
            number_of_point_sources_per_wavelength,
            label="number of point sources per wavelength",
        )

        self.location = _convert_to_array(location, shape=(3,), label="array location")

        self.element_shape = element_shape

        if element_shape is "circular":
            self.element_radius = _convert_to_positive_float(element_radius)
            self.element_width = None
            self.element_height = None
        elif element_shape is "rectangular":
            self.element_radius = None
            self.element_width = _convert_to_positive_float(element_width)
            self.element_height = _convert_to_positive_float(element_height)
        else:
            raise NotImplementedError

        self.centroid_locations = self._calc_centroid_locations(
            centroid_locations, centroid_locations_filename
        )

        self.number_of_elements = self.centroid_locations.shape[1]

        self.velocity = _convert_to_complex_array(
            velocity, shape=(self.number_of_elements,), label="velocity"
        )

        self.array_type = array_type

        self.radius_of_curvature = self._calc_radius_of_curvature(
            self.centroid_locations, self.array_type
        )

        self.element_normals = self._calc_element_normals(
            self.centroid_locations, array_type
        )

    def _calc_centroid_locations(self, centroid_locations, centroid_locations_filename):
        """Calculates centroid locations.

        Parameters
        ----------
        centroid_locations : numpy.ndarray
            An array of size (3, N) with the locations of the centroids of the piston elements.
        centroid_locations_filename : str
            Path and filename containing the centroid locations data. The file extension has to be ".dat".

        Returns
        -------
        centroid_locations : numpy.ndarray
            An array of size (3, N) with the locations of the centroids of the piston elements.
        """

        if centroid_locations is not None and centroid_locations_filename is not None:
            raise ValueError(
                "Specify either the centroid locations or the centroid locations "
                + "filename."
            )
        elif centroid_locations is not None and centroid_locations_filename is None:
            centroid_locations = _convert_to_3n_array(
                centroid_locations, label="element centroid locations"
            )
        elif centroid_locations is None and centroid_locations_filename is not None:
            if centroid_locations_filename.endswith(".dat"):
                centroid_locations = _np.loadtxt(centroid_locations_filename)
            else:
                raise ValueError(
                    "The centroid locations filename must have a dat extension."
                )

        return centroid_locations

    def _calc_element_normals(self, centroid_locations, array_type):
        """Calculates the array element normals from centroid locations depending on the
        array type.

        Parameters
        ----------
        centroid_locations : numpy.ndarray
            An array of size (3, N) with the locations of the centroids of the piston elements.
        array_type : str
            The type of array. Can be spherical or planar.
        Returns
        -------
        element_normals : numpy.ndarray
            An array of size (3, N) with the locations of the centroids of the piston elements.
        """

        if array_type is "spherical":
            element_normals = -_normalize_vector(self.centroid_locations)
        elif array_type is "planar":
            element_normals = _np.zeros((3, centroid_locations.shape[1]), dtype="float")
            element_normals[2, :] = _np.ones(centroid_locations.shape[1], dtype="float")
        else:
            raise NotImplementedError

        return element_normals

    def _calc_radius_of_curvature(self, centroid_locations, array_type):
        """Calculates the radius of curvature of the array transducer from centroid
        locations.

        Parameters
        ----------
        centroid_locations : numpy.ndarray
            An array of size (3, N) with the locations of the centroids of the piston elements.

        Returns
        -------
        radius of curvature : float
            The radius of curvature of the array.
        """

        if array_type is "spherical":
            centroid_locations_l2_norm = _np.linalg.norm(centroid_locations, axis=0)
            centroid_locations_l2_norm_std = _np.std(centroid_locations_l2_norm)
            radius_of_curvature_from_centroid_locations = _np.mean(
                centroid_locations_l2_norm
            )

            radius_of_curvature_tol = 1e-6
            if centroid_locations_l2_norm_std > radius_of_curvature_tol:
                raise ValueError(
                    "Array element centroid locations do not appear to lie on a sphere."
                )
        elif array_type is "planar":
            radius_of_curvature_from_centroid_locations = None

            z_coordinate_tol = 1e-6
            centroid_z_locations_std = _np.std(centroid_locations[2, :])
            if centroid_z_locations_std > z_coordinate_tol:
                raise ValueError(
                    "Array element centroid locations do not appear to lie in the xy plane."
                )

        return radius_of_curvature_from_centroid_locations

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
            An array of size (3,N) with the locations on which to evaluate the pressure field.
        normals : numpy.ndarray
            An array of size (3,N) with the unit normal vectors at the locations on which to evaluate the pressure field.

        Returns
        -------
        gradient : numpy.ndarray
            An array of size (3,N) with the normal gradient of the pressure in the locations.
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
        """Calculate the surface traces of the source field on the mesh.

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
