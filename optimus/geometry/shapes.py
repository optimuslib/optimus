"""Predefined geometry shapes."""

import bempp.api as _bempp
from .common import Geometry as _Geometry


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
            Element size.
        """

        grid = _bempp.shapes.sphere(r, origin, h)

        super().__init__(grid)
