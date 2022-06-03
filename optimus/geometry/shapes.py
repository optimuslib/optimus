"""Predefined geometry shapes."""

import bempp.api as _bempp
from .common import Geometry as _Geometry


class RegularSphere(_Geometry):
    def __init__(self, refinement_level=1):
        """
        Create a regular sphere geometry, i.e., a structured mesh on a sphere
        with unit radius.

        Parameters
        ----------
        refinement_level : int
            Refinement level.
        """

        grid = _bempp.shapes.regular_sphere(refinement_level)

        super().__init__(grid, label="regular_sphere")

        self.refinement = refinement_level


class Sphere(_Geometry):
    def __init__(self, radius=1, origin=(0, 0, 0), element_size=0.1):
        """
        Create a sphere geometry.

        Parameters
        ----------
        radius : float
            Radius of the sphere.
        origin : tuple
            Center of the sphere.
        element_size : float
            Mesh element size.
        """

        grid = _bempp.shapes.sphere(radius, origin, element_size)

        super().__init__(grid, label="sphere")

        self.radius = radius
        self.origin = origin
        self.meshwidth = element_size


class Ellipsoid(_Geometry):
    def __init__(self, radius=(1, 1, 1), origin=(0, 0, 0), element_size=0.1):
        """
        Create an ellipsoid geometry.

        Parameters
        ----------
        radius : tuple
            Radii of the ellipsoid along (x,y,z) axes.
        origin : tuple
            Center of the ellipsoid.
        element_size : float
            Mesh element size.
        """

        grid = _bempp.shapes.ellipsoid(
            radius[0], radius[1], radius[2], origin, element_size
        )

        super().__init__(grid, label="ellipsoid")

        self.radius = radius
        self.origin = origin
        self.meshwidth = element_size


class Cube(_Geometry):
    def __init__(self, length=1, origin=(0, 0, 0), element_size=0.1):
        """
        Create a cube geometry.

        Parameters
        ----------
        length : float
            The length of the edges.
        origin : tuple
            Position of the vertex with minimum value in each direction.
        element_size : float
            Mesh element size.
        """

        grid = _bempp.shapes.cube(length, origin, element_size)

        super().__init__(grid, label="cube")

        self.length = length
        self.origin = origin
        self.meshwidth = element_size


class ReentrantCube(_Geometry):
    def __init__(self, refinement_factor=0.2, element_size=0.1):
        """
        Create a reentrant cube geometry.

        Parameters
        ----------
        refinement_factor : float
            The refinement factor of the reentry.
        element_size : float
            Mesh element size.
        """

        grid = _bempp.shapes.reentrant_cube(element_size, refinement_factor)

        super().__init__(grid, label="reentrant_cube")

        self.refinement = refinement_factor
        self.meshwidth = element_size


class Almond(_Geometry):
    def __init__(self, element_size=0.01):
        """
        Create a NASA almond geometry.

        Parameters
        ----------
        element_size : float
            Mesh element size.
        """

        grid = _bempp.shapes.almond(element_size)

        super().__init__(grid, label="nasa_almond")

        self.meshwidth = element_size
