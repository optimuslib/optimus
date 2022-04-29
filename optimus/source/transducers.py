"""Common functionality for transducer sources."""

import numpy as _np

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
    field_locations : np.ndarray of size 3 x N
        The coordinates of the locations at which the incident field
        is evaluated.
    normals : np.ndarray of size 3 x N
        The coordinates of the unit normal vectors for evaluation of the
        pressure normal gradient on the surface of scatterers.
    verbose : boolean
        Verbosity of output.
        Default: False

    Returns
    ----------
    An object with the attributes 'pressure' and 'normal_pressure_gradient'.
    """

    if source.type == "piston":
        transducer = _Transducer(
            source,
            medium,
            field_locations,
            normals,
            verbose,
        )
    else:
        raise NotImplementedError

    transducer.generate_source_points()

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
        self.wavenumber = medium.wavenumber(source.frequency)
        self.frequency = source.frequency
        self.verbose = verbose

        self.source_locations = None
        self.source_weights = None

        self.pressure = None
        self.normal_pressure_gradient = None

    def generate_source_points(self):
        """
        Generate the source points of the transducer. The field emitted from
        any transducer is modelled by a collection of point sources, each
        with weighting for its amplitude.

        Sets the following class attributes.
            source_locations : np.ndarray of size 3 X N_sourcepoints
                The 3D location of each point source.
            source_weights : np.ndarray of size N_sourcepoints
                The weighting of each point source.
        """

        source_locations_inside_element = (
            self.define_source_points_in_unit_transducer_element()
        )

        n_sources_per_element = source_locations_inside_element.shape[1]

        if self.verbose:
            print(
                "Number of point sources per element:",
                n_sources_per_element,
            )

        self.source_locations = self.transform_source_points(
            source_locations_inside_element
        )

        n_sources = self.source_locations.shape[1]

        velocity_weighting = _np.full(n_sources, self.source.velocity)

        surface_area_weighting = (
            _np.pi * self.source.radius**2 / n_sources_per_element
        )

        self.source_weights = surface_area_weighting * velocity_weighting

    def define_source_points_in_unit_transducer_element(self):
        """
        Define the source points for a unit transducer element,
        that is, the source points on a rectangular grid, located
        in the plane z=0, centered at the origin, and inside a
        disk of the specified radius. The resolution of the points
        is determined by the specified number of point sources per
        wavelength. If zero points per wavelength is specified,
        return the center of the disk as the only source point.

        Returns
        -------
        source_locations_inside_element : np.ndarray of size 3 x N_points
            The locations of the point source inside the element.
        """

        if self.source.number_of_point_sources_per_wavelength == 0:

            source_locations_inside_element = _np.zeros((3, 1))

        else:

            wavelength = 2 * _np.pi / self.wavenumber.real
            distance_between_points = (
                wavelength / self.source.number_of_point_sources_per_wavelength
            )
            n_points_per_diameter = (
                2 * self.source.radius / distance_between_points
            )
            n_point_sources = int(_np.ceil(n_points_per_diameter))

            if self.verbose:
                print(
                    "Number of point sources across element diameter:",
                    n_point_sources,
                )

            coords = _np.linspace(
                -self.source.radius,
                self.source.radius,
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
            inside = distance <= self.source.radius
            source_locations_inside_element = source_vector[:, inside]

        return source_locations_inside_element

    def transform_source_points(
        self, source_locations_on_unit_disk, element_range=None
    ):
        """
        Transform the source points from the unit disk to the actual
        location of the transducer source. For multi-element arrays,
        the transformation is applied to the specified range of elements.

        Parameters
        ----------
        source_locations_on_unit_disk : np.ndarray of size 3 x N_sourcepoints
            The locations of the source points on the unit disk, located
            in the plane z=0, centered at the origin and with the
            specified radius.
        element_range : list
            The range of transducer elements in the multi-element array.

        Returns
        -------
        source_locations_transformed : np.ndarray of size 3 x N_sourcepoints
            The locations of the source points on the transducer.
        """

        source_locations_directed = _rotate(
            source_locations_on_unit_disk, self.source.source_axis
        )

        source_locations_translated = _translate(
            source_locations_directed, self.source.location
        )

        source_locations_curved = self.apply_curvature(
            source_locations_translated
        )

        if element_range is None:
            transformation = self.get_transformation_matrix()
            source_locations_transformed = (
                transformation @ source_locations_curved
            )
        else:
            raise NotImplementedError

        return source_locations_transformed

    def get_transformation_matrix(self, transducer_element=None):
        """
        Calculate the transformation matrix of the transducer.

        Parameters
        ----------
        transducer_element : int (default: None)
            The element in the transducer array. For a single transducer
            element, specify None.

        Returns
        -------
        transformation : np.ndarray of size 3 x 3
            The 3D transformation matrix.
        """
        if self.source.type == "piston" and transducer_element is None:

            return _np.identity(3)

        elif self.source.type == "bowl" and transducer_element is not None:

            x, y = self.source.centroid_location[:2, transducer_element]
            beta = _np.arcsin(-x / self.source.radius_of_curvature)
            alpha = _np.arcsin(
                y / (self.source.radius_of_curvature * _np.cos(beta))
            )

            mx = _np.array(
                [
                    [1, 0, 0],
                    [0, _np.cos(alpha), -_np.sin(alpha)],
                    [0, _np.sin(alpha), _np.cos(alpha)],
                ]
            )

            my = _np.array(
                [
                    [_np.cos(beta), 0, _np.sin(beta)],
                    [0, 1, 0],
                    [-_np.sin(beta), 0, _np.cos(beta)],
                ]
            )

            return mx @ my

        else:

            raise NotImplementedError

    def apply_curvature(
        self,
        locations,
        radius_of_curvature=None,
        origin_of_curvature=(0, 0, 0),
    ):
        """
        Apply the radius of curvature to the points.

        Parameters
        ----------
        locations : np.ndarray of size 3 x N_points
            The locations of the points to be transformed.
        radius_of_curvature : float
            The radius of curvature.
        origin_of_curvature : array like
            The origin of the curvature.

        Returns
        -------
        locations_curved : np.ndarray of size 3 x N_points
            The transformed locations of the points.

        """
        if None in [radius_of_curvature, origin_of_curvature]:
            return locations
        else:
            raise NotImplementedError

    def calc_pressure_field(self):
        """
        Calculate the pressure field and the normal gradient of the
        transducer, in a collection of 3D observation points.

        Uses the following class attributes
        ----------
        source_locations : np.ndarray of size 3 x N_sourcepoints
            The coordinates of the locations of the point sources used to
            discretise the acoustic source.
        source_weights : np.ndarray of size N_sourcepoints
            The weighting assigned to each point source.
        field_locations : np.ndarray of size 3 x N_observationpoints
            The coordinates of the locations at which the incident field
            is evaluated.
        normals : np.ndarray of size 3 x N_observationpoints
            The coordinates of the normal vectors for evaluation of the
            pressure normal gradient on the surface of scatterers.

        Sets the following class attributes.
        ----------
        pressure: np.ndarray of size N_observationpoints
            The pressure in the observation points.
        normal_pressure_gradient: np.ndarray of size 3 x N_observationpoints
            The normal gradient of the pressure in the observation points.
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
            normal_pressure_gradient = _np.sum(
                pressure_gradient * self.normals, axis=0
            )
        else:
            normal_pressure_gradient = None

        self.pressure = pressure_value
        self.normal_pressure_gradient = normal_pressure_gradient


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

    Parameters
    ----------
    locations_source : np.ndarray of size 3 x N_sourcepoints
        The locations of the source points.
    locations_observation : np.ndarray of size 3 x N_observationpoints
        The locations of the observation points.
    frequency : float
        The frequency of the wave field.
    density : float
        The density of the propagating medium.
    wavenumber : complex
        The wavenumber of the wave field.
    source_weights : np.ndarray of size N_sourcepoints
        Weights of each source element.

    Returns
    -------
    pressure : np.ndarray of size N_observationpoints
        The pressure of the wave field in the observation points.
    gradient : np.ndarray of size 3 x N_observationpoints
        The gradient of the pressure field in the observation points.
    """

    if locations_source.ndim == 1:
        locations_source.reshape((3, 1))
    if locations_observation.ndim == 1:
        locations_source.reshape((3, 1))

    def apply_amplitude(values):
        return (2j * _np.pi * frequency * density) * values

    differences_between_all_points = (
        locations_source[:, _np.newaxis, :]
        - locations_observation[:, :, _np.newaxis]
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
        greens_function_scaled
        * (wavenumber * distances_between_all_points + 1j),
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
