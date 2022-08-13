"""Common functionality for transducer sources."""

import numpy as _np
from numba import njit as _njit
from numba import prange as _prange

from ..utils.linalg import translate as _translate
from ..utils.linalg import rotate as _rotate


def transducer_field(
    source,
    medium,
    field_locations,
    normals=None,
    verbose=False,
):
    """
    Calculate the field emitted by a transducer source.

    Parameters
    ----------
    source : optimus.source.Source
        The type of acoustic source used.
    medium : optimus.material.Material
        The propagating medium.
    field_locations : numpy.ndarray
        An array of size (3,N) with the coordinates of the locations at
        which the incident field is evaluated.
    normals : numpy.ndarray
        Array of size (3,N) with the coordinates of the unit normal vectors
        for evaluation of the pressure normal gradient on the surface of scatterers.
    verbose : boolean
        Verbosity of output.
        Default: False

    Returns
    ----------
    transducer : _Transducer
        An object with the attributes 'pressure' and 'normal_pressure_gradient'.
    """

    if source.type in ("piston", "bowl", "array"):
        transducer = _Transducer(
            source,
            medium,
            field_locations,
            normals,
            verbose,
        )
    else:
        raise NotImplementedError

    transducer.calc_pressure_field()

    return transducer


class _Transducer:
    def __init__(self, source, medium, field_locations, normals, verbose):
        """
        Functionality to create different types of transducer sources
        and calculate the pressure field emitted from them.
        """
        self.source = source
        self.field_locations = field_locations
        self.normals = normals
        self.density = medium.density
        self.wavenumber = medium.compute_wavenumber(source.frequency)
        self.frequency = source.frequency
        self.verbose = verbose

        self.source_locations, self.source_weights = self.generate_source_points()

        self.pressure = None
        self.normal_pressure_gradient = None

    def generate_source_points(self):
        """
        Generate the source points of the transducer. The field emitted from
        any transducer is modelled by a collection of point sources, each
        with weighting for its amplitude.

        Returns
        ----------
        source_locations : numpy.ndarray
            An array of size (3,N_sourcepoints) with the location of each point source.
        source_weights : numpy.ndarray
            An array of size (N_sourcepoints,) with the weighting of each point source.
        """

        if self.source.type == "piston":
            (
                source_locations_inside_transducer,
                surface_area_weighting,
            ) = self.define_source_points_in_reference_piston(self.source.radius)
        elif self.source.type == "bowl":
            (
                source_locations_inside_transducer,
                surface_area_weighting,
            ) = self.define_source_points_in_reference_bowl()
        elif self.source.type == "array":
            (
                source_locations_inside_transducer,
                surface_area_weighting,
            ) = self.define_source_points_in_reference_array()
        else:
            raise NotImplementedError

        if self.verbose:
            print(
                "Number of point sources in transducer:",
                source_locations_inside_transducer.shape[1],
            )

        source_locations = self.transform_source_points(
            source_locations_inside_transducer
        )

        if self.source.type == "array":
            number_of_sources_per_element = (
                source_locations.shape[1] / self.source.number_of_elements
            )
            velocity_weighting = _np.repeat(
                self.source.velocity, number_of_sources_per_element
            )
        else:
            n_sources = source_locations.shape[1]
            velocity_weighting = _np.full(n_sources, self.source.velocity)

        source_weights = surface_area_weighting * velocity_weighting

        return source_locations, source_weights

    def define_source_points_in_reference_piston(self, radius):
        """
        Define the source points for a reference piston element,
        that is, the source points on a rectangular grid, located
        in the plane z=0, centered at the origin, and inside a
        disk of the specified radius.
        The resolution of the points is determined by the specified
        number of point sources per wavelength. If zero points per
        wavelength is specified, return the center of the disk as
        the only source point.

        The surface area weighting is uniform.

        Parameters
        ----------
        radius : float
            The radius of piston.

        Returns
        -------
        locations_inside_transducer : numpy.ndarray
            An array of size (3,N_points) with the locations of the point source
            inside the reference element.
        surface_area_weighting : float
            The surface area weighting associated to each point source.
        """

        if self.source.number_of_point_sources_per_wavelength == 0:

            locations_inside_transducer = _np.zeros((3, 1))

        else:

            wavelength = 2 * _np.pi / self.wavenumber.real
            distance_between_points = (
                wavelength / self.source.number_of_point_sources_per_wavelength
            )
            n_points_per_diameter = 2 * radius / distance_between_points
            n_point_sources = int(_np.ceil(n_points_per_diameter))

            if self.verbose:
                print(
                    "Number of point sources across element diameter:",
                    n_point_sources,
                )

            coords = _np.linspace(
                -radius,
                radius,
                n_point_sources,
            )

            source_vector = _np.vstack(
                (
                    _np.repeat(coords, n_point_sources),
                    _np.tile(coords, n_point_sources),
                    _np.zeros(n_point_sources**2),
                )
            )
            distance = _np.linalg.norm(source_vector[:2, :], axis=0)
            inside = distance <= radius
            locations_inside_transducer = source_vector[:, inside]

        n_sources = locations_inside_transducer.shape[1]

        surface_area_weighting = _np.pi * radius**2 / n_sources

        return locations_inside_transducer, surface_area_weighting

    def define_source_points_in_reference_bowl(self):
        """
        Define the source points for a reference spherical section
        bowl transducer, that is, the source points on a spherical
        section bowl whose apex is in contact with the z=0 plane,
        at the global origin and whose axis is the Cartesian x-axis.
        The bowl is defined by its outer radius and radius of
        curvature. A circular aperture may be defined by specifying
        an inner radius.

        The resolution of the points is determined by the specified
        number of point sources per wavelength which must be strictly
        positive.

        The surface area weighting is uniform.

        Returns
        -------
        locations_inside_transducer : numpy.ndarray
            An array of size (3,N_points) with the locations of the point source
            inside the reference element.
        surface_area_weighting : float
            The surface area weighting associated to each point source.
        """

        bowl_mesh_parameter_init = (
            2
            * _np.pi
            / (
                self.wavenumber.real
                * _np.abs(self.source.number_of_point_sources_per_wavelength)
                * self.source.radius_of_curvature
            )
        )

        number_of_points_on_sphere = int(
            _np.round(4 * _np.pi / bowl_mesh_parameter_init**2)
        )
        bowl_mesh_parameter = _np.sqrt(4 * _np.pi / number_of_points_on_sphere)
        elevation_angle_discretisation_param = int(
            _np.round(_np.pi / bowl_mesh_parameter)
        )

        azimuthal_angle_increment = (
            4 * elevation_angle_discretisation_param / number_of_points_on_sphere
        )
        elevation_angle = (
            _np.pi * (_np.arange(elevation_angle_discretisation_param) + 1.5)
        ) / elevation_angle_discretisation_param
        azimuthal_angle_discretisation_param = _np.round(
            2 * _np.pi * _np.sin(elevation_angle) / azimuthal_angle_increment
        ).astype(int)
        azimuthal_angle_mesh_param = azimuthal_angle_discretisation_param[
            azimuthal_angle_discretisation_param > 0
        ].sum()
        surface_area_weighting = (
            4
            * _np.pi
            * self.source.radius_of_curvature**2
            / azimuthal_angle_mesh_param
        )

        outer_radius_angle = _np.arcsin(
            self.source.outer_radius / self.source.radius_of_curvature
        )
        outer_radius_threshold = self.source.radius_of_curvature * _np.cos(
            _np.pi - outer_radius_angle
        )

        if self.source.inner_radius is not None:
            if self.source.inner_radius == 0:
                inner_radius_threshold = -self.source.radius_of_curvature
            else:
                inner_radius_angle = _np.arcsin(
                    self.source.inner_radius / self.source.radius_of_curvature
                )
                inner_radius_threshold = self.source.radius_of_curvature * _np.cos(
                    _np.pi - inner_radius_angle
                )
        else:
            inner_radius_threshold = -_np.inf

        x_source = list()
        y_source = list()
        z_source = list()

        for m in range(elevation_angle_discretisation_param):
            z = self.source.radius_of_curvature * _np.cos(elevation_angle[m])
            if inner_radius_threshold <= z <= outer_radius_threshold:
                sintheta = self.source.radius_of_curvature * _np.sin(elevation_angle[m])
                azimuthal_angle = [
                    2 * _np.pi * (n + 1) / azimuthal_angle_discretisation_param[m]
                    for n in range(azimuthal_angle_discretisation_param[m])
                ]
                for azimuthal_angle_idx in azimuthal_angle:
                    x_source.append(sintheta * _np.cos(azimuthal_angle_idx))
                    y_source.append(sintheta * _np.sin(azimuthal_angle_idx))
                    z_source.append(z)

        x_source = _np.array(x_source)
        y_source = _np.array(y_source)
        z_source = _np.array(z_source) + self.source.radius_of_curvature

        locations_inside_transducer = _np.vstack((x_source, y_source, z_source))

        if self.verbose:

            if self.source.inner_radius is None:
                radius_section = self.source.radius_of_curvature - _np.sqrt(
                    self.source.radius_of_curvature**2 - self.source.outer_radius**2
                )
            else:
                radius_section = _np.sqrt(
                    self.source.radius_of_curvature**2 - self.source.inner_radius**2
                ) - _np.sqrt(
                    self.source.radius_of_curvature**2 - self.source.outer_radius**2
                )

            actual_area = 2 * _np.pi * self.source.radius_of_curvature * radius_section
            estimated_area = (
                locations_inside_transducer.shape[1] * surface_area_weighting
            )

            print("Actual transducer area (m^2): {:.14f}".format(actual_area))
            print("Approximated transducer area (m^2): {:.14f}".format(estimated_area))

        return locations_inside_transducer, surface_area_weighting

    def define_source_points_in_reference_array(self):
        """
        Define the source points for a reference spherical section
        array transducer, that is, the source points on a spherical
        section bowl whose apex is in contact with the z=0 plane,
        at the global origin and whose axis is the Cartesian x-axis.
        The array is defined by the location of the element centroids
        and the radius of the elements.

        The resolution of the points is determined by the specified
        number of point sources per wavelength which must be strictly
        positive.

        The surface area weighting is uniform.

        Returns
        -------
        locations_inside_transducer : numpy.ndarray
            An array of size (3,N_points) with the locations of the point source
            inside the reference array.
        surface_area_weighting : float
            The surface area weighting associated to each point source.
        """

        (
            locations_inside_transducer,
            surface_area_weighting,
        ) = self.define_source_points_in_reference_piston(self.source.element_radius)

        n_points_piston = locations_inside_transducer.shape[1]
        n_points_array = n_points_piston * self.source.number_of_elements
        source_locations_array = _np.empty((3, n_points_array), dtype="float")

        for element_number in range(self.source.number_of_elements):

            source_locations_directed = _rotate(
                locations_inside_transducer,
                self.source.element_normals[:, element_number],
            )
            source_locations_transformed = _translate(
                source_locations_directed,
                self.source.centroid_locations[:, element_number],
            )
            indices = range(
                element_number * n_points_piston, (element_number + 1) * n_points_piston
            )
            source_locations_array[:, indices] = source_locations_transformed

        source_locations_array[2, :] += self.source.radius_of_curvature

        return source_locations_array, surface_area_weighting

    def transform_source_points(self, source_locations_on_reference_source):
        """
        Transform the source points from the reference transducer to the actual
        location of the transducer, as specified by the source axis and the source
        location.

        Parameters
        ----------
        source_locations_on_reference_source : numpy.ndarray
            An array of size (3,N_sourcepoints) with the locations of the
            source points on the reference element of the transducer type.

        Returns
        -------
        source_locations_transformed : numpy.ndarray
            An array of size (3,N_sourcepoints) with the locations of the
            source points on the transducer.
        """

        source_locations_directed = _rotate(
            source_locations_on_reference_source, self.source.source_axis
        )

        source_locations_transformed = _translate(
            source_locations_directed, self.source.location
        )

        return source_locations_transformed

    def calc_pressure_field(self):
        """
        Calculate the pressure field and the normal gradient of the
        transducer, in a collection of 3D observation points.

        Uses the following class attributes.
        ----------
        source_locations : numpy.ndarray
            An array of size (3,N_sourcepoints) with the coordinates of the
            locations of the point sources used to discretise the acoustic source.
        source_weights : numpy.ndarray
            An array of size (N_sourcepoints,) with the weighting assigned
            to each point source.
        field_locations : numpy.ndarray
            An array of size (3,N_observationpoints) with the coordinates of the
            locations at which the incident field is evaluated.
        normals : numpy.ndarray
            An array of size (3,N_observationpoints) with the coordinates of the
            normal vectors for evaluation of the pressure normal gradient on the
            surface of scatterers.

        Sets the following class attributes.
        ----------
        pressure: numpy.ndarray
            An array of size (N_observationpoints,) with the pressure in the
            observation points.
        normal_pressure_gradient: numpy.ndarray
            An array of size (3,N_observationpoints) with the normal gradient
            of the pressure in the observation points.
        """

        pressure_value, pressure_gradient = calc_field_from_point_sources(
            self.source_locations,
            self.field_locations,
            self.frequency,
            self.density,
            self.wavenumber,
            self.source_weights,
        )
        if self.normals is not None:
            normal_pressure_gradient = _np.sum(pressure_gradient * self.normals, axis=0)
        else:
            normal_pressure_gradient = None

        self.pressure = pressure_value
        self.normal_pressure_gradient = normal_pressure_gradient


@_njit(parallel=True)
def calc_greens_functions_in_observation_points_numba(
    locations_source, locations_observation, wavenumber, source_weights
):
    """
    Calculate the pressure field and its gradient for a
    summation of point sources describing a transducer,
    according to the Rayleigh integral formula.

    Use Numba for acceleration and parallelisation.

    Parameters
    ----------
    locations_source : numpy.ndarray
        An array of size (3,N_sourcepoints) with the locations of the source points.
    locations_observation : numpy.ndarray
        An array of size (3,N_observationpoints) with the locations of the
        observation points.
    wavenumber : complex
        The wavenumber of the wave field.
    source_weights : numpy.ndarray
        An array of size (N_sourcepoints,) with the weights of each source element.

    Returns
    -------
    greens_function_in_observation_points : numpy.ndarray
        An array of size (N_observationpoints,) with the Green's function of
        the wave field at the observation points with contribution of all
        source locations.
    greens_gradient_in_observation_points : numpy.ndarray
        An array of size (3,N_observationpoints) with the gradient of Green's
        function of the wave field at the observation points with contribution
        of all source locations. When normals is None, gradient related quantities
        are not evaluated.
    """
    greens_function_in_observation_points_scaled = _np.zeros_like(
        locations_observation[0], dtype=_np.complex128
    )
    greens_gradient_in_observation_points_scaled = _np.zeros_like(
        locations_observation, dtype=_np.complex128
    )
    for i in _prange(locations_observation.shape[1]):
        temp_greens_function_in_observation_points_scaled = 0.0
        temp_greens_gradient_in_observation_points_scaled = _np.zeros(
            3, dtype=_np.complex128
        )
        differences_between_all_points = _np.zeros(3, dtype=_np.float64)
        for j in range(locations_source.shape[1]):
            differences_between_all_points[0] = (
                locations_source[0, j] - locations_observation[0, i]
            )
            differences_between_all_points[1] = (
                locations_source[1, j] - locations_observation[1, i]
            )
            differences_between_all_points[2] = (
                locations_source[2, j] - locations_observation[2, i]
            )

            distances_between_all_points = _np.linalg.norm(
                differences_between_all_points
            )

            greens_function_scaled = (
                _np.exp(1j * wavenumber * distances_between_all_points)
                / distances_between_all_points
            )
            temp_greens_function_in_observation_points_scaled += (
                greens_function_scaled * source_weights[j]
            )

            greens_gradient_amplitude_scaled = _np.divide(
                greens_function_scaled
                * (wavenumber * distances_between_all_points + 1j),
                distances_between_all_points**2,
            )
            greens_gradient_scaled = (
                differences_between_all_points * greens_gradient_amplitude_scaled
            )
            temp_greens_gradient_in_observation_points_scaled += (
                greens_gradient_scaled * source_weights[j]
            )

        greens_function_in_observation_points_scaled[
            i
        ] = temp_greens_function_in_observation_points_scaled

        greens_gradient_in_observation_points_scaled[
            :, i
        ] = temp_greens_gradient_in_observation_points_scaled

    greens_function_in_observation_points = (
        greens_function_in_observation_points_scaled / (2 * _np.pi)
    )

    greens_gradient_in_observation_points = (
        -1j / (2 * _np.pi)
    ) * greens_gradient_in_observation_points_scaled
    return (
        greens_function_in_observation_points,
        greens_gradient_in_observation_points,
    )


def calc_field_from_point_sources(
    locations_source,
    locations_observation,
    frequency,
    density,
    wavenumber,
    source_weights,
):
    """
    Calculate the pressure field and its gradient of a point source,
    according to the Rayleigh integral formula.

    Use Numba acceleration.

    Parameters
    ----------
    locations_source : numpy.ndarray
        An array of size (3,N_sourcepoints) with the locations of the source points.
    locations_observation : numpy.ndarray
        An array of size (3,N_observationpoints) with the locations of the
        observation points.
    frequency : float
        The frequency of the wave field.
    density : float
        The density of the propagating medium.
    wavenumber : complex
        The wavenumber of the wave field.
    source_weights : numpy.ndarray
        An array of size (N_sourcepoints,) with the weights of each source element.

    Returns
    -------
    pressure : numpy.ndarray
        An array of size (N_observationpoints,) with the pressure of the
        wave field in the observation points.
    gradient : numpy.ndarray
        An array of size (3,N_observationpoints) with the gradient of the
        pressure field in the observation points.
    """

    if locations_source.ndim == 1:
        locations_source.reshape((3, 1))
    if locations_observation.ndim == 1:
        locations_source.reshape((3, 1))

    def apply_amplitude(values):
        return (2j * _np.pi * frequency * density) * values

    (
        greens_function_in_observation_points,
        greens_gradient_in_observation_points,
    ) = calc_greens_functions_in_observation_points_numba(
        locations_source,
        locations_observation,
        wavenumber,
        source_weights,
    )

    pressure = apply_amplitude(greens_function_in_observation_points)
    gradient = apply_amplitude(greens_gradient_in_observation_points)

    return pressure, gradient


def calc_field_from_point_sources_numpy(
    locations_source,
    locations_observation,
    frequency,
    density,
    wavenumber,
    source_weights,
):
    """
    Calculate the pressure field and its gradient of a point source,
    according to the Rayleigh integral formula.

    Parameters
    ----------
    locations_source : numpy.ndarray
        An array of size (3,N_sourcepoints) with the locations of the source points.
    locations_observation : numpy.ndarray
        An array of size (3,N_observationpoints) with the locations of the
        observation points.
    frequency : float
        The frequency of the wave field.
    density : float
        The density of the propagating medium.
    wavenumber : complex
        The wavenumber of the wave field.
    source_weights : numpy.ndarray
        An array of size (N_sourcepoints,) with the weights of each source element.

    Returns
    -------
    pressure : numpy.ndarray
        An array of size (N_observationpoints,) with the pressure of the
        wave field in the observation points.
    gradient : numpy.ndarray
        An array of size (3,N_observationpoints) with the gradient of the
        pressure field in the observation points.
    """

    if locations_source.ndim == 1:
        locations_source.reshape((3, 1))
    if locations_observation.ndim == 1:
        locations_source.reshape((3, 1))

    def apply_amplitude(values):
        return (2j * _np.pi * frequency * density) * values

    differences_between_all_points = (
        locations_source[:, _np.newaxis, :] - locations_observation[:, :, _np.newaxis]
    )
    distances_between_all_points = _np.linalg.norm(
        differences_between_all_points, axis=0
    )

    greens_function_scaled = _np.divide(
        _np.exp((1j * wavenumber) * distances_between_all_points),
        distances_between_all_points,
    )
    greens_function_in_observation_points_scaled = _np.dot(
        greens_function_scaled, source_weights
    )
    greens_function_in_observation_points = (
        greens_function_in_observation_points_scaled / (2 * _np.pi)
    )

    pressure = apply_amplitude(greens_function_in_observation_points)

    greens_gradient_amplitude_scaled = _np.divide(
        greens_function_scaled * (wavenumber * distances_between_all_points + 1j),
        distances_between_all_points**2,
    )
    greens_gradient_scaled = (
        differences_between_all_points
        * greens_gradient_amplitude_scaled[_np.newaxis, :, :]
    )
    greens_gradient_in_observation_points_scaled = _np.dot(
        greens_gradient_scaled, source_weights
    )
    greens_gradient_in_observation_points = (
        -1j / (2 * _np.pi)
    ) * greens_gradient_in_observation_points_scaled

    gradient = apply_amplitude(greens_gradient_in_observation_points)

    return pressure, gradient
