"""Predefined geometry shapes."""

import bempp.api as _bempp
from .common import Geometry as _Geometry


class RegularSphere(_Geometry):
    def __init__(self, n=1):
        """
        Create a regular sphere geometry.

        Parameters
        ----------
        n : int
            Refinement level.
        """

        grid = _bempp.shapes.regular_sphere(n)

        super().__init__(grid, label="regular_sphere")

        self.refinement = n


class Sphere(_Geometry):
    def __init__(self, r=1, origin=(0, 0, 0), h=0.1):
        """
        Create a sphere geometry.

        Parameters
        ----------
        r : float
            Radius of the sphere.
        origin : tuple
            Center of the sphere.
        h : float
            Mesh element size.
        """

        grid = _bempp.shapes.sphere(r, origin, h)

        super().__init__(grid, label="sphere")

        self.radius = r
        self.origin = origin
        self.meshwidth = h


class Ellipsoid(_Geometry):
    def __init__(self, r=(1, 1, 1), origin=(0, 0, 0), h=0.1):
        """
        Create an ellipsoid geometry.

        Parameters
        ----------
        r : tuple
            Radii of the ellipsoid.
        origin : tuple
            Center of the ellipsoid.
        h : float
            Mesh element size.
        """

        grid = _bempp.shapes.ellipsoid(r[0], r[1], r[2], origin, h)

        super().__init__(grid, label="ellipsoid")

        self.radius = r
        self.origin = origin
        self.meshwidth = h


class Cube(_Geometry):
    def __init__(self, length=1, origin=(0, 0, 0), h=0.1):
        """
        Create a cube geometry.

        Parameters
        ----------
        length : float
            The length of the edges.
        origin : tuple
            Position of the vertex with minimum value in each direction.
        h : float
            Mesh element size.
        """

        grid = _bempp.shapes.cube(length, origin, h)

        super().__init__(grid, label="cube")

        self.length = length
        self.origin = origin
        self.meshwidth = h


class ReentrantCube(_Geometry):
    def __init__(self, refinement_factor=0.2, h=0.1):
        """
        Create a reentrant cube geometry.

        Parameters
        ----------
        refinement_factor : float
            The refinement factor of the reentry.
        h : float
            Mesh element size.
        """

        grid = _bempp.shapes.reentrant_cube(h, refinement_factor)

        super().__init__(grid, label="reentrant_cube")

        self.refinement = refinement_factor
        self.meshwidth = h


class Almond(_Geometry):
    def __init__(self, h=0.01):
        """
        Create a NASA almond geometry.

        Parameters
        ----------
        h : float
            Mesh element size.
        """

        grid = _bempp.shapes.almond(h)

        super().__init__(grid, label="nasa_almond")

        self.meshwidth = h
