"""Common functionality for transducer sources."""

import numpy as _np

from ..utils.linalg import translate as _translate
from ..utils.linalg import rotate as _rotate


def incident_field(
    source,
    medium,
    locations,
    normals=None,
    num_cpu=None,
    verbose=False,
):
    """
    Calculates the incident field for an OptimUS source.

    Parameters
    ----------
    source : optimus.source.Source
        The type of acoustic source used.
    medium : optimus.material.Material
        The propagating medium.
    locations : 3 by N array
        The coordinates of the locations at which the incident field
        is evaluated.
    normals : 3 by N array
        The coordinates of the normal vectors for evaluation of the
        pressure normal gradient on the surface of scatterers.
    num_cpu : integer
        The number of cpu cores over which the incident calculation
        is to be parallelised.
        Default: None (no custom parallelisation)
    verbose : boolean
        Verbosity of output.
        Default: False
    """

    if source.type == "piston":
        solver_class = PistonSolver
    else:
        raise NotImplementedError

    field_solver = solver_class(
        source,
        medium,
        locations,
        normals,
        num_cpu,
        verbose,
    )

    field_solver.solver()

    return field_solver


class FieldSolver:
    def __init__(
        self,
        source,
        medium,
        field_locations,
        normals,
        num_cpu,
        verbose,
        source_locations=None,
        source_weights=None,
    ):
        """
        Calculates the incident field for an OptimUS source.

        Parameters
        ----------
        source : Class
            The type of acoustic source used.
        medium : Class
            The acoustic medium.
        field_locations : 3 by N array
            The coordinates of the locations at which the incident field
            is evaluated.
        normals :  3 by N array
            The coordinates of the normal vectors for evaluation of the
            pressure normal gradient on the surface of scatterers.
        num_cpu : integer
            The number of cpu cores over which the incident calculation
            is to be parallelised.
            Default: None (no parallelisation)
        verbose : boolean
            Verbosity of output.
            Default: False
        source_locations : 3 x N array
            Locations of the point sources on the transducer.
        source_weights : float
            Weights of the point sources on the transducer.
        """

        self.source = source
        self.field_locations = field_locations
        self.normals = normals
        self.density = medium.density
        self.wavenumber = medium.wavenumber(source.frequency)
        self.frequency = source.frequency
        self.num_cpu = num_cpu
        self.verbose = verbose

        self.source_locations = source_locations
        self.source_weights = source_weights

        self.pressure = None
        self.normal_pressure_gradient = None

    def point_source_generator(self):
        return self.source_locations, self.source_weights

    def solver(self):
        self.pressure = None
        self.normal_pressure_gradient = None


class SourceSolver(FieldSolver):
    def calculate_field(
        self, source_locations, source_weights, field_locations, normals
    ):
        """
        Returns weighted sum of pressures and normal pressure derivatives
        resulting from source_locations locations at field_locations locations.

        Parameters
        ----------
        source_locations : 3 by M array
            The coordinates of the locations of the point sources used to
            discretise the acoustic source.
        source_weights : 1D array of size M
            The weighting assigned to each point source.
        field_locations : 3 by N array
            The coordinates of the locations at which the incident field
            is evaluated.
        normals : 3 by N array
            The coordinates of the normal vectors for evaluation of the
            pressure normal gradient on the surface of scatterers.
        """

        def calculate_pressure(phi_val):
            return 1j * phi_val * 2 * _np.pi * self.frequency * self.density

        source_field_distances_diff = (
            field_locations[:, _np.newaxis, :]
            - source_locations[:, :, _np.newaxis]
        )
        dist = _np.linalg.norm(source_field_distances_diff, axis=0)

        kr = self.wavenumber * dist

        # Compute Green's function
        if source_locations.shape[1] > 1:
            phi = _np.sum(
                _np.divide(
                    source_weights[:, _np.newaxis] * _np.exp(1j * kr), dist
                ),
                axis=0,
                dtype=_np.complex,
            ) / (2 * _np.pi)
        else:
            phi = _np.sum(
                _np.divide(source_weights * _np.exp(1j * kr), dist),
                axis=0,
                dtype=_np.complex,
            ) / (2 * _np.pi)

        pressure = calculate_pressure(phi)

        # Compute grad of Green's function
        # Compute grad of velocity potential quantity
        if source_locations.shape[1] > 1:
            h = (
                1j
                * _np.divide(
                    source_weights[:, _np.newaxis]
                    * _np.exp(1j * kr)
                    * (kr + 1j),
                    dist**3,
                )
                / (2 * _np.pi)
            )
            grad_phi = _np.sum(
                source_field_distances_diff * h[_np.newaxis, :, :],
                axis=1,
                dtype=_np.complex,
            )
        else:
            h = (
                1j
                * _np.divide(
                    source_weights * _np.exp(1j * kr) * (kr + 1j), dist**3
                )
                / (2 * _np.pi)
            )
            diff_locations = field_locations - source_locations
            grad_phi = diff_locations * _np.tile(h, (3, 1))

        grad_pressure = calculate_pressure(grad_phi)

        if normals is not None:
            normal_pressure_gradient = (grad_pressure * normals).sum(axis=0)
        else:
            normal_pressure_gradient = 0

        return pressure, normal_pressure_gradient

    def solver(self):
        """
        Calculates the pressure and normal pressure gradient of the incident
        field of the source.
        """

        if None in [self.source_locations, self.source_weights]:

            source_locs, self.source_weights = self.point_source_generator()

            source_locations_rotated = _rotate(
                source_locs, self.source.source_axis
            )

            self.source_locations = _translate(
                source_locations_rotated, self.source.location
            )

        self.pressure, self.normal_pressure_gradient = self.calculate_field(
            self.source_locations,
            self.source_weights,
            self.field_locations,
            self.normals,
        )


class ArrayPistonSolver(SourceSolver):
    def add_radius_of_curvature(self, location, radius=None):
        """
        Apply the radius of curvature to the points.
        """
        if radius is None:
            return location
        else:
            raise NotImplementedError

    def get_source_no_range(self):
        """
        Return the range of indices of piston elements in multi-element
        transducer sources.
        """
        if self.source.type == "piston":
            source_no_range = range(1)
        else:
            source_no_range = range(self.source.centroid_location.shape[1])
        return source_no_range

    def get_transformation_matrix(self, i):
        """
        Get the transformation matrix for the piston element.

        The transducer object needs to following attributes:
         - centroid_location: 3 x N array
           The locations of the centroids of the N piston elements.
         - radius_of_curvature: float
           The radius of curvature of the bowl transducer.
        """
        x, y = self.source.centroid_location[:2, i]
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

    def point_source_generator(self):
        """
        Returns the coordinates of the point sources used to discretise a
        transducer source that consists on piston elements.

        Returns
        ----------
        locations : 3 x N array
            The locations of all point sources.
        weights : 1 x N array
            The weights of all point sources.
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

        n_sources_per_element = source_locations_inside_element.shape[1]

        if self.verbose:
            print(
                "Number of point sources per element:",
                n_sources_per_element,
            )

        source_locations = []
        for source_no in self.get_source_no_range():
            transformation = self.get_transformation_matrix(source_no)
            locations_transformed = (
                transformation @ source_locations_inside_element
            )
            source_locations.append(locations_transformed)
        source_locations_array = _np.hstack(source_locations)

        source_locations = self.add_radius_of_curvature(source_locations_array)

        velocity_weighting = _np.repeat(
            self.source.velocity, n_sources_per_element
        )
        surface_area_weighting = (
            _np.pi * self.source.radius**2 / n_sources_per_element
        )
        source_weights = surface_area_weighting * velocity_weighting

        return source_locations, source_weights


class PistonSolver(ArrayPistonSolver):
    def get_transformation_matrix(self, i):
        """
        Get the transformation matrix for the piston element.
        For a single piston source, this is the identity.
        """
        return _np.identity(3)
