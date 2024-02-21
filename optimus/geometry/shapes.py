"""Predefined geometry shapes."""
import bempp.api as _bempp
from .common import Geometry as _Geometry

class RegularSphere(_Geometry):
    def __init__(self, refinement_level=1, label="regular_sphere"):
        """Create a regular sphere geometry, i.e., a structured mesh on a sphere
        with unit radius.

        Parameters
        ----------
        refinement_level : int
            Refinement level.
        label : str
            The name of the geometry.
        """

        grid = _bempp.shapes.regular_sphere(refinement_level)

        super().__init__(grid, "sphere", label)

        self.radius = 1
        self.origin = (0, 0, 0)
        self.refinement = refinement_level
        self._correct_elements_group_bempp_grids()


class Sphere(_Geometry):
    def __init__(self, radius=1, origin=(0, 0, 0), element_size=0.1, label="sphere"):
        """Create a sphere geometry.

        Parameters
        ----------
        radius : float
            Radius of the sphere.
        origin : tuple[float]
            Center of the sphere.
        element_size : float
            Mesh element size.
        label : str
            The name of the geometry.
        """

        grid = _bempp.shapes.sphere(radius, origin, element_size)

        super().__init__(grid, "sphere", label)

        self.radius = radius
        self.origin = origin
        self.meshwidth = element_size
        self._correct_elements_group_bempp_grids()


class Ellipsoid(_Geometry):
    def __init__(
        self, radius=(1, 1, 1), origin=(0, 0, 0), element_size=0.1, label="ellipsoid"
    ):
        """Create an ellipsoid geometry.

        Parameters
        ----------
        radius : tuple[float]
            Radii of the ellipsoid along (x,y,z) axes.
        origin : tuple[float]
            Center of the ellipsoid.
        element_size : float
            Mesh element size.
        label : str
            The name of the geometry.
        """

        grid = _bempp.shapes.ellipsoid(
            radius[0], radius[1], radius[2], origin, element_size
        )

        super().__init__(grid, "ellipsoid", label)

        self.radius = radius
        self.origin = origin
        self.meshwidth = element_size


class Cube(_Geometry):
    def __init__(self, length=1, origin=(0, 0, 0), element_size=0.1, label="cube"):
        """Create a cube geometry.

        Parameters
        ----------
        length : float
            The length of the edges.
        origin : tuple[float]
            Position of the vertex with minimum value in each direction.
        element_size : float
            Mesh element size.
        label : str
            The name of the geometry.
        """

        grid = _bempp.shapes.cube(length, origin, element_size)

        super().__init__(grid, "cube", label)

        self.length = length
        self.origin = origin
        self.meshwidth = element_size


class ReentrantCube(_Geometry):
    def __init__(self, refinement_factor=0.2, element_size=0.1, label="reentrant_cube"):
        """Create a reentrant cube geometry.

        Parameters
        ----------
        refinement_factor : float
            The refinement factor of the reentry.
        element_size : float
            Mesh element size.
        label : str
            The name of the geometry.
        """

        grid = _bempp.shapes.reentrant_cube(element_size, refinement_factor)

        super().__init__(grid, "reentrant_cube", label)

        self.refinement = refinement_factor
        self.meshwidth = element_size


class Almond(_Geometry):
    def __init__(self, element_size=0.01, label="almond"):
        """Create a NASA almond geometry.

        Parameters
        ----------
        element_size : float
            Mesh element size.
        label : str
            The name of the geometry.
        """

        grid = _bempp.shapes.almond(element_size)

        super().__init__(grid, "almond", label)

        self.meshwidth = element_size


class Cuboid(_Geometry):
    def __init__(
        self, length=(1, 1, 1), origin=(0, 0, 0), element_size=0.1, label="cuboid"
    ):
        """Create a cuboid geometry.

        Parameters
        ----------
        length : tuple[float]
            The lengths of the three edges.
        origin : tuple[float]
            The location of the corner with minimum values in all three axes.
        element_size : float
            Mesh element size.
        label : str
            The name of the geometry.
        """

        from ..utils.mesh import generate_grid_from_geo_string

        self.length = length
        self.origin = origin
        self.meshwidth = element_size

        gmsh_string = self._cuboid_gmsh_string()
        grid = generate_grid_from_geo_string(gmsh_string)

        super().__init__(grid, "cuboid", label)

    def _cuboid_gmsh_string(self):
        """Create GMSH string for cuboid shape.
        This function is identical to the equivalent bempp-cl functionality.
        """

        cuboid_stub = """
        Point(1) = {orig0,orig1,orig2,cl};
        Point(2) = {orig0+l0,orig1,orig2,cl};
        Point(3) = {orig0+l0,orig1+l1,orig2,cl};
        Point(4) = {orig0,orig1+l1,orig2,cl};
        Point(5) = {orig0,orig1,orig2+l2,cl};
        Point(6) = {orig0+l0,orig1,orig2+l2,cl};
        Point(7) = {orig0+l0,orig1+l1,orig2+l2,cl};
        Point(8) = {orig0,orig1+l1,orig2+l2,cl};
        Line(1) = {1,2};
        Line(2) = {2,3};
        Line(3) = {3,4};
        Line(4) = {4,1};
        Line(5) = {1,5};
        Line(6) = {2,6};
        Line(7) = {3,7};
        Line(8) = {4,8};
        Line(9) = {5,6};
        Line(10) = {6,7};
        Line(11) = {7,8};
        Line(12) = {8,5};
        Line Loop(1) = {-1,-4,-3,-2};
        Line Loop(2) = {1,6,-9,-5};
        Line Loop(3) = {2,7,-10,-6};
        Line Loop(4) = {3,8,-11,-7};
        Line Loop(5) = {4,5,-12,-8};
        Line Loop(6) = {9,10,11,12};
        Plane Surface(1) = {1};
        Plane Surface(2) = {2};
        Plane Surface(3) = {3};
        Plane Surface(4) = {4};
        Plane Surface(5) = {5};
        Plane Surface(6) = {6};
        Physical Surface(1) = {1};
        Physical Surface(2) = {2};
        Physical Surface(3) = {3};
        Physical Surface(4) = {4};
        Physical Surface(5) = {5};
        Physical Surface(6) = {6};
        Surface Loop (1) = {1,2,3,4,5,6};
        Volume (1) = {1};
        Mesh.Algorithm = 6;
        """

        cuboid_geometry = (
            "l0 = "
            + str(self.length[0])
            + ";\n"
            + "l1 = "
            + str(self.length[1])
            + ";\n"
            + "l2 = "
            + str(self.length[2])
            + ";\n"
            + "orig0 = "
            + str(self.origin[0])
            + ";\n"
            + "orig1 = "
            + str(self.origin[1])
            + ";\n"
            + "orig2 = "
            + str(self.origin[2])
            + ";\n"
            + "cl = "
            + str(self.meshwidth)
            + ";\n"
            + cuboid_stub
        )

        return cuboid_geometry
