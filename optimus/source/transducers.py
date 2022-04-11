"""Transducer incident field solver"""
from abc import ABCMeta, abstractmethod
import multiprocessing as mp
import time as time
import sys
import numpy as np

# TODO: generalise this to an N piston array
from .coordinate_transformations import translate, rotate


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
    source : Class
        The type of acoustic source used.
    medium : Class
        The acoustic medium.
    locations : 3 by N array
        The coordinates of the locations at which the incident field is evaluated.
    normals :  3 by N array
        The coordinates of the normal vectors for evaluation of the pressure normal gradient on the surface of scatterers.
    num_cpu : integer
        The number of cpu cores over which the incident calculation is to be parallelised.
        TODO: implement this.
    verbose : verbose
    """
    solverClass = {
        "piston": PistonSolver,
    }

    fieldSolver = solverClass[source.type](
        source,
        medium,
        locations,
        normals,
        num_cpu=num_cpu,
        verbose=verbose,
    )
    fieldSolver.main()
    return fieldSolver


class FieldSolver:
    def __init__(
        self,
        source,
        medium,
        locations,
        normals,
        verbose=False,
        source_locations=None,
        source_weight=None,
        num_cpu=None,
    ):

        """
        Calculates the incident field for an OptimUS source.

        Parameters
        ----------
        source : Class
            The type of acoustic source used.
        medium : Class
            The acoustic medium.
        locations : 3 by N array
            The coordinates of the locations at which the incident field is evaluated.
        normals :  3 by N array
            The coordinates of the normal vectors for evaluation of the pressure normal gradient on the surface of scatterers.
        num_cpu : integer
            The number of cpu cores over which the incident calculation is to be parallelised.
            TODO: implement this.
        verbose : verbose
        """

        self.num_cpu = mp.cpu_count()  # num_cpu or psutil.cpu_count(logical=False)
        self.max_RAM = 2e7  # min(2E7, psutil.virtual_memory().available)
        self.source = source
        # TODO locations needs to be at least 2D - automate change from (3,) or (1,3) to (3,1)
        self.locations = locations
        self.normals = normals
        # FIXME normals needs to be an array of shape (3,...) - this should fail if not with these dimensions
        self.density = medium.density
        self.wavenumber = medium.wavenumber(source.frequency)
        self.frequency = source.frequency
        self.verbose = verbose
        self.source_locations = source_locations
        self.source_weight = source_weight

    @abstractmethod
    def point_source_generator(self):
        return (self.source_locations, self.source_weight)

    @abstractmethod
    def solver(self):
        pass

    def __sizeof__(self):
        return object.__sizeof__(self) + sum(
            sys.getsizeof(v) for v in self.__dict__.values()
        )

    def main(self):

        self.solver()


class SourceSolver(FieldSolver):
    def calculate_field(
        self, source_locations, field_locations, source_weight, normals
    ):
        """
        Returns weighted sum of pressures and normal pressure derivatives resulting from
        source_locations locations at field_locations locations.
        Parameters
        ----------
        source_locations : 3 x N array
            The coordinates of the locations of the point sources used to discretise the acoustic source.
        field_locations : 3 by N array
            The coordinates of the locations at which the incident field is evaluated.
        source_weight : 1D array
            The weigthing assigned to each point source.
        normals :  3 by N array
            The coordinates of the normal vectors for evaluation of the pressure normal gradient on the surface of scatterers.
        """

        def calculate_pressure(phi):
            return 1j * phi * 2 * np.pi * self.frequency * self.density

        # Compute distances between all source and receiver locations
        source_field_distances_diff = (
            field_locations[:, np.newaxis, :] - source_locations[:, :, np.newaxis]
        )
        R = np.sqrt(((source_field_distances_diff) ** 2).sum(axis=0))

        kR = self.wavenumber * R

        if not isinstance(source_weight, np.ndarray):
            source_weight = np.array(
                [source_weight] * source_locations.shape[1]
            ).reshape(1, source_locations.shape[1])

        # Compute Green's function
        if source_locations.shape[1] > 1:
            phi = np.sum(
                np.divide(source_weight[:, np.newaxis] * np.exp(1j * kR), R),
                axis=0,
                dtype=np.complex,
            ) / (2 * np.pi)
        else:
            phi = np.sum(
                np.divide(source_weight * np.exp(1j * kR), R), axis=0, dtype=np.complex
            ) / (2 * np.pi)

        # Compute acoustics pressure
        pressure = calculate_pressure(phi)

        # Compute grad of Green's function
        # Compute grad of velocity potential quantity
        if source_locations.shape[1] > 1:
            H = (
                1j
                * np.divide(
                    source_weight[:, np.newaxis] * np.exp(1j * kR) * (kR + 1j), R**3
                )
                / 2
                / np.pi
            )
            grad_phi = (source_field_distances_diff * H[np.newaxis, :, :]).sum(
                axis=1, dtype=np.complex
            )
        else:
            H = (
                1j
                * np.divide(source_weight * np.exp(1j * kR) * (kR + 1j), R**3)
                / 2
                / np.pi
            )
            grad_phi = (field_locations - source_locations) * np.tile(H, (3, 1))

        # Compute grad of acoustic pressure
        grad_pressure = calculate_pressure(grad_phi)
        self.grad_pressure = grad_pressure

        if normals is not None:
            # Obtain normal derivative of acoustic pressure
            normal_pressure_gradient = (grad_pressure * normals).sum(axis=0)

        else:
            normal_pressure_gradient = 0

        return pressure, normal_pressure_gradient

    def solver(self):
        """
        Returns the pressure and normal pressure gradient of the incident field of the source.
        Parameters
        ----------
        """
        # Generate point sources associated with source_type
        (
            source_locations_init,
            self.source_weight,
        ) = self.point_source_generator()  # FIXME source_locations is an attribute...

        # Rotate source locations according to source axis
        source_locations_rotated = rotate(
            source_locations_init, self.source.source_axis
        )
        # translate source locations according to source axis
        self.source_locations = translate(
            source_locations_rotated, self.source.location
        )
        self.pressure, self.normal_pressure_gradient = self.calculate_field(
            self.source_locations, self.locations, self.source_weight, self.normals
        )


class ArrayPistonSolver(SourceSolver):
    @abstractmethod
    def add_radius_of_curvature(self, location):
        return location

    @abstractmethod
    def get_source_no_range(self):
        # indices of the Number of HIFU array elements
        if self.source.type == "piston":
            source_no_range = range(1)
        else:
            source_no_range = range(self.source.centroid_location.shape[1])
        return source_no_range
        # TODO: implement array type source

    @abstractmethod
    def get_transformation_matrix(self, i):
        x, y = self.source.centroid_location[:2, i]
        beta = np.arcsin(-x / self.source.radius_of_curvature)
        alpha = np.arcsin(y / (self.source.radius_of_curvature * np.cos(beta)))
        m = alpha.size
        Mx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)],
            ]
        )

        My = np.array(
            [
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)],
            ]
        )

        return Mx @ My

    @abstractmethod
    def get_z0_source(self, number_of_point_sources):
        return np.zeros(number_of_point_sources**2)

    def point_source_generator(self):
        """
        Returns the coordinates of the point sources used to discretise a piston type source
        based on the piston radius and the number of elements per wavelength, and the weighting
        for each point source
        Parameters
        ----------
        """

        # In case of 0 grid points per wavelength, default to centroid of piston(s)
        source_locations_inside_element = np.array(self.source.location).reshape((3, 1))

        if self.source.number_of_point_sources_per_wavelength != 0:

            # FIXME this is out by 2 * pi
            number_of_point_sources = 1 + int(
                (
                    self.wavenumber.real
                    * self.source.radius
                    * self.source.number_of_point_sources_per_wavelength
                )
                / np.pi
            )
            print(
                "Number of point sources across element diameter:",
                number_of_point_sources,
            )

            x0 = np.linspace(
                -self.source.radius,
                self.source.radius,
                number_of_point_sources,
            )

            source_vector = np.array(
                [
                    np.repeat(x0, number_of_point_sources),  # x0_source
                    np.tile(x0, number_of_point_sources),  # y0_source
                    self.get_z0_source(number_of_point_sources),  # z0_source
                ]
            )
            square_grid = np.sqrt((source_vector[:2, :] ** 2).sum(axis=0))
            inside = square_grid <= self.source.radius
            if any(inside):
                source_locations_inside_element = source_vector[:, inside]

        number_of_sources_per_element = source_locations_inside_element.shape[1]

        if self.verbose:
            print("\n", 70 * "*")
            print(
                "\n Number of point sources per element: ",
                number_of_sources_per_element,
            )
            print("\n", 70 * "*")

        source_no_range = self.get_source_no_range()

        velocity_weighting, source_locations = tuple(), tuple()

        for source_no in source_no_range:

            # Compute rotation grad_pressurerices
            MxMy = self.get_transformation_matrix(source_no)

            # Carry out coordinate transfornation
            locations_transformed = MxMy @ source_locations_inside_element

            # Generate array of point source locations for each element
            source_locations += (locations_transformed,)

            # Generate velocity weighting for each point source
            velocity_weighting += (
                self.source.velocity[source_no]
                * np.ones(number_of_sources_per_element),
            )

        # Stack data in tuples to arrays
        source_locations_array = np.hstack(source_locations)
        velocity_weighting_array = np.hstack(velocity_weighting)

        source_locations_array = self.add_radius_of_curvature(source_locations_array)

        # surface area weighting associated with each point source
        delta_S = np.pi * self.source.radius**2 / number_of_sources_per_element
        # Combined source weighting involving velocity and surface area
        source_weight = delta_S * velocity_weighting_array

        return (source_locations_array, source_weight)


class PistonSolver(ArrayPistonSolver):
    def get_transformation_matrix(self, i):
        return np.identity(3)
