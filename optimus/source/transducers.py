"""Common functionality for transducer sources."""

import numpy as _np
from numba import njit as _njit
from numba import prange as _prange
import multiprocessing as _mp

from ..utils.linalg import translate as _translate
from ..utils.linalg import rotate as _rotate


def transducer_field(
    source,
    medium,
    field_locations,
    normals=None,
    verbose=False,
):
    """Calculate the field emitted by a transducer source.

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
    -------
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
        """Functionality to create different types of transducer sources and calculate the pressure field emitted from them."""

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
        """Generate the source points of the transducer. The field emitted from
        any transducer is modelled by a collection of point sources, each
        with weighting for its amplitude.

        Returns
        -------
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
        """Define the source points for a reference piston element,
        that is, the source points on a rectangular grid, located
        in the plane z=0, centered at the origin, and inside a
        disk of the specified radius.
        The resolution of the points is determined by the specified
        number of point sources per wavelength. If zero points per
        wavelength is specified, return the center of the disk as
        the only source point. The surface area weighting is uniform.

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

        if n_sources < 1:
            raise TypeError(
                "Number of point sources inside piston transducer must be greater "
                + " 0: increase number_of_point_sources_per_wavelength"
            )

        surface_area_weighting = _np.pi * radius**2 / n_sources

        return locations_inside_transducer, surface_area_weighting

    def define_source_points_in_reference_bowl(self):
        """Define the source points for a reference spherical section
        bowl transducer, that is, the source points on a spherical
        section bowl whose apex is in contact with the z=0 plane,
        at the global origin and whose axis is the Cartesian x-axis.
        The bowl is defined by its outer radius and radius of
        curvature. A circular aperture may be defined by specifying
        an inner radius. The resolution of the points is determined by the specified
        number of point sources per wavelength which must be strictly
        positive. The surface area weighting is uniform.

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

        if locations_inside_transducer.shape[1] < 1:
            raise TypeError(
                "Number of point sources inside bowl transducer must be greater than"
                + " 0: increase number_of_point_sources_per_wavelength"
            )

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
        """Define the source points for a reference spherical section
        array transducer, that is, the source points on a spherical
        section bowl whose apex is in contact with the z=0 plane,
        at the global origin and whose axis is the Cartesian x-axis.
        The array is defined by the location of the element centroids
        and the radius of the elements. The resolution of the points is determined by the specified
        number of point sources per wavelength which must be strictly
        positive. The surface area weighting is uniform.

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
        """Transform the source points from the reference transducer to the actual
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
        """Calculate the pressure field and the normal gradient of the
        transducer, in a collection of 3D observation points.

        Parameters
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

        Returns
        -------
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
            self.verbose,
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
    """Calculate the pressure field and its gradient for a
    summation of point sources describing a transducer,
    according to the Rayleigh integral formula. Use Numba for acceleration and parallelisation.

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
    verbose,
):
    """Calculate the pressure field and its gradient of a point source,
    according to the Rayleigh integral formula. Use Numba or Multiprocessing acceleration.

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
    verbose : boolean
        Display output.

    Returns
    -------
    pressure : numpy.ndarray
        An array of size (N_observationpoints,) with the pressure of the
        wave field in the observation points.
    gradient : numpy.ndarray
        An array of size (3,N_observationpoints) with the gradient of the
        pressure field in the observation points.
    """
    from optimus import global_parameters
    from itertools import repeat

    if locations_source.ndim == 1:
        locations_source.reshape((3, 1))
    if locations_observation.ndim == 1:
        locations_source.reshape((3, 1))

    parallelisation_method = (
        global_parameters.incident_field_parallelisation.parallelisation_method
    )

    if parallelisation_method.lower() == "numba":

        if verbose:
            print("Parallelisation library is: numba")

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

    elif parallelisation_method.lower() in [
        "multiprocessing",
        "mp",
        "multi-processing",
    ]:

        if verbose:
            print("Parallelisation library is: multiprocessing")

        (
            chunks_index_source,
            chunks_index_field,
            number_of_source_chunks,
            number_of_field_chunks,
        ) = chunk_size_index(
            locations_source,
            locations_observation,
        )

        number_of_workers = global_parameters.incident_field_parallelisation.cpu_count

        number_of_observation_locations = locations_observation.shape[1]
        number_of_source_locations = locations_source.shape[1]

        pool = _mp.Pool(number_of_workers)

        source_parallelisation = (
            number_of_source_locations > number_of_observation_locations
        )

        if source_parallelisation:

            if verbose:
                print(
                    "Parallelisation of incident field calculation "
                    "over source locations"
                )

            number_of_parallel_jobs = _np.arange(0, number_of_source_chunks - 1)

            result = pool.starmap(
                calc_field_from_point_sources_mp_source_para,
                zip(
                    number_of_parallel_jobs,
                    repeat(locations_source),
                    repeat(locations_observation),
                    repeat(frequency),
                    repeat(density),
                    repeat(wavenumber),
                    repeat(source_weights),
                    repeat(chunks_index_source),
                    repeat(chunks_index_field),
                    repeat(number_of_observation_locations),
                ),
            )

        else:

            if verbose:
                print(
                    "Parallelisation of incident field calculation "
                    "over observer locations"
                )

            number_of_parallel_jobs = _np.arange(0, number_of_field_chunks - 1)

            result = pool.starmap(
                calc_field_from_point_sources_mp_field_para,
                zip(
                    number_of_parallel_jobs,
                    repeat(locations_source),
                    repeat(locations_observation),
                    repeat(frequency),
                    repeat(density),
                    repeat(wavenumber),
                    repeat(source_weights),
                    repeat(chunks_index_source),
                    repeat(chunks_index_field),
                ),
            )

        pool.close()

        if source_parallelisation:
            pressure_result = _np.stack([r[0] for r in result])
            gradient_result = _np.stack([r[1] for r in result], axis=2)

            pressure = pressure_result.sum(axis=0, dtype=_np.complex)
            gradient = gradient_result.sum(axis=2, dtype=_np.complex)
        else:
            result_as_array = _np.asarray(result)

            pressure = _np.hstack(result_as_array[..., 0])
            gradient = _np.hstack(result_as_array[..., 1])

    else:
        raise NotImplementedError

    return pressure, gradient


def calc_field_from_point_sources_numpy(
    locations_source,
    locations_observation,
    frequency,
    density,
    wavenumber,
    source_weights,
):
    """Calculate the pressure field and its gradient of a point source,
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


def calc_field_from_point_sources_mp_source_para(
    parallelisation_index,
    locations_source,
    locations_observation,
    frequency,
    density,
    wavenumber,
    source_weights,
    chunks_index_source,
    chunks_index_field,
    number_of_observation_locations,
):
    """Computes the pressure and normal pressure gradient at field locations for
    selected source and field positions based on output from chunk_size_index. Used to
    calculate the incident field using multiprocessing and when parallelising over
    source locations.

    Parameters
    ----------
    parallelisation_index : integer
        Index corresponding to the parallel job.
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
    chunks_index_source : numpy.ndarray
        An array of integers containing the indices to infer the source location chunks.
    chunks_index_field : numpy.ndarray
        An array of integers containing the indices to infer the field location chunks.
    number_of_observation_locations : integer
        The total number of observation points.

    Returns
    -------
    pressure : numpy.ndarray
        An array of size (N_observationpoints,) with the pressure of the
        wave field in the observation points.
    gradient : numpy.ndarray
        An array of size (3,N_observationpoints) with the gradient of the
        pressure field in the observation points.
    """

    pressure = _np.empty(number_of_observation_locations, dtype=_np.complex)
    gradient = _np.empty((3, number_of_observation_locations), dtype=_np.complex)

    i1, i2 = chunks_index_source[parallelisation_index : parallelisation_index + 2]

    for i in range(len(chunks_index_field) - 1):

        j1, j2 = chunks_index_field[i : i + 2]

        (pressure[j1:j2], gradient[:, j1:j2],) = calc_field_from_point_sources_numpy(
            locations_source[:, i1:i2],
            locations_observation[:, j1:j2],
            frequency,
            density,
            wavenumber,
            source_weights[i1:i2],
        )

    return pressure, gradient


def calc_field_from_point_sources_mp_field_para(
    parallelisation_index,
    locations_source,
    locations_observation,
    frequency,
    density,
    wavenumber,
    source_weights,
    chunks_index_source,
    chunks_index_field,
):
    """Computes the pressure and normal pressure gradient at field locations for
    selected source and field positions based on output from chunk_size_index. Used to
    calculate the incident field using multiprocessing and when parallelising over
    field locations.

    Parameters
    ----------
    parallelisation_index : integer
        Index corresponding to the parallel job.
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
    chunks_index_source : numpy.ndarray
        An array of integers containing the indices to infer the source location chunks.
    chunks_index_field : numpy.ndarray
        An array of integers containing the indices to infer the field location chunks.

    Returns
    -------
    pressure : numpy.ndarray
        An array of size (N_observationpoints,) with the pressure of the
        wave field in the observation points.
    gradient : numpy.ndarray
        An array of size (3,N_observationpoints) with the gradient of the
        pressure field in the observation points.
    """

    j1, j2 = chunks_index_field[parallelisation_index : parallelisation_index + 2]

    pressure_tmp = _np.ndarray(
        shape=(j2 - j1, len(chunks_index_source) - 1), dtype=_np.complex
    )
    gradient_tmp = _np.ndarray(
        shape=(3, j2 - j1, len(chunks_index_source) - 1), dtype=_np.complex
    )

    for i in range(len(chunks_index_source) - 1):
        i1, i2 = chunks_index_source[i : i + 2]

        (
            pressure_tmp[:, i],
            gradient_tmp[:, :, i],
        ) = calc_field_from_point_sources_numpy(
            locations_source[:, i1:i2],
            locations_observation[:, j1:j2],
            frequency,
            density,
            wavenumber,
            source_weights[i1:i2],
        )

    pressure = pressure_tmp.sum(axis=1, dtype=_np.complex)
    gradient = gradient_tmp.sum(axis=2, dtype=_np.complex)

    return pressure, gradient


def chunk_size_index(locations_source, locations_observation):
    """Computes the chunk sizes and indices used to calculate the incident field using
    multiprocessing and when parallelising over source or observation locations. The
    chunks are allocated based on the global_parameters.incident_field.mem_per_core
    parameter.

    Parameters
    ----------
    locations_source : numpy.ndarray
        An array of size (3,N_sourcepoints) with the locations of the source points.
    locations_observation : numpy.ndarray
        An array of size (3,N_observationpoints) with the locations of the
        observation points.

    Returns
    -------
    chunks_index_source : numpy.ndarray
        An array of integers describing the source point indices corresponding to each
        chunk for parallelisation.
    chunks_index_observation : numpy.ndarray
        An array of integers describing the observation point indices corresponding to each
        chunk for parallelisation.
    number_of_source_chunks : integer
        The number of source chunks used for parallelisation over source locations.
    number_of_observation_chunks : integer
        The number of source chunks used for parallelisation over observation locations.
    """

    from optimus import global_parameters

    mem_per_core = global_parameters.incident_field_parallelisation.mem_per_core

    number_of_bytes_for_complex_numpy_type = 16

    total_dimension_source = locations_source.shape[1]
    total_dimension_field = locations_observation.shape[1]
    chunk_size = int(
        _np.ceil(
            mem_per_core
            / (total_dimension_source * number_of_bytes_for_complex_numpy_type)
        )
    )

    number_of_source_chunks, chunks_index_source = break_in_chunks(
        total_dimension_source, chunk_size
    )
    number_of_observation_chunks, chunks_index_observation = break_in_chunks(
        total_dimension_field, chunk_size
    )

    return [
        chunks_index_source,
        chunks_index_observation,
        number_of_source_chunks,
        number_of_observation_chunks,
    ]


def break_in_chunks(number_of_locations, chunk_size):
    """Computes the chunk sizes and indices used to calculate the incident field using
    multiprocessing and when parallelising over source or observation locations. The
    chunks are allocated based on the global_parameters.incident_field.mem_per_core
    parameter.

    Parameters
    ----------
    number_of_locations : integer
        The total number of source or observer locations.
    chunk_size :  integer
        The size of the chunks the source or observation points are broken down into.

    Returns
    -------
    number_of_chunks : integer
        The number of chunks the source or observation points are broken into.
    chunks_index : numpy.ndarray
        An array of size (N,) containing the indices of the limits of the source or
        observation point chunks.
    """

    minimum_number_of_chunks = 2
    number_of_chunks = max(
        int(_np.ceil(number_of_locations / chunk_size)), minimum_number_of_chunks
    )
    chunks_index = _np.linspace(
        0, number_of_locations, max(2, number_of_chunks), dtype=int
    )

    return number_of_chunks, chunks_index
