"""Get mesh statistics and scale mesh elements."""

import bempp.api as bempp
from ..geometry.common import Geometry as _Geometry
from .conversions import convert_to_float as _convert_to_float
import numpy as np


def _get_mesh_stats(grid, verbose=False):
    """
    Compute the minimum, maximum, median, mean and standard deviation of
    mesh elements for a grid object.

    Parameters
    ----------
    grid : bempp.api.Grid
        The surface grid.
    verbose: boolean
        Print the results.

    Returns
    ----------
    stats : dict
        The mesh statistics.
    """
    elements = list(grid.leaf_view.entity_iterator(0))
    element_size = [
        np.sqrt(
            (
                (
                    element.geometry.corners
                    - np.roll(element.geometry.corners, 1, axis=1)
                )
                ** 2
            ).sum(axis=0)
        )
        for element in elements
    ]
    elements_min = np.min(element_size)
    elements_max = np.max(element_size)
    elements_avg = np.mean(element_size)
    elements_med = np.median(element_size)
    elements_std = np.std(element_size)
    number_of_nodes = grid.leaf_view.entity_count(2)

    if verbose:
        print("\n", 70 * "*")
        print("Number of nodes: {0}.\n".format(number_of_nodes))
        print(
            "Statistics about the element size in the triangular surface grid:\n"
            " Min: {0:.2e}\n Max: {1:.2e}\n AVG: {2:.2e}\n"
            " MED: {3:.2e}\n STD: {4:.2e}\n".format(
                elements_min, elements_max, elements_avg, elements_med, elements_std
            )
        )
        print("\n", 70 * "*")

    return {
        "elements_min": elements_min,
        "elements_max": elements_max,
        "elements_avg": elements_avg,
        "elements_med": elements_med,
        "elements_std": elements_std,
        "number_of_nodes": number_of_nodes,
    }


def get_geometries_stats(geometries, verbose=False):
    """
    Compute the minimum, maximum, median, mean and standard deviation of
    mesh elements for optimus Geometries.

    Parameters
    ----------
    geometries : list or tuple
        The Optimus geometry object(s).
    verbose: boolean
        Print the results.

    Returns
    ----------
    stats : dict
        The mesh statistics.
    """
    elements_min = []
    elements_max = []
    elements_avg = []
    elements_med = []
    elements_std = []
    number_of_nodes = []
    if isinstance(geometries, (list, tuple)):
        for geometry in geometries:
            stats = _get_mesh_stats(geometry.grid, verbose=False)
            elements_min.append(stats["elements_min"])
            elements_max.append(stats["elements_max"])
            elements_avg.append(stats["elements_avg"])
            elements_med.append(stats["elements_med"])
            elements_std.append(stats["elements_std"])
            number_of_nodes.append(stats["number_of_nodes"])
        total_number_of_nodes = np.sum(number_of_nodes)

        stats_total = {
            "elements_min": elements_min,
            "elements_max": elements_max,
            "elements_avg": elements_avg,
            "elements_med": elements_med,
            "elements_std": elements_std,
            "number_of_nodes": number_of_nodes,
        }

        if verbose:
            print("\n", 70 * "*")
            for i, geometry in enumerate(geometries):
                print(
                    "Number of nodes in geometry {0} is {1}.\n".format(
                        i + 1, number_of_nodes[i]
                    )
                )
                print(
                    (
                        "Statistics about the element size in the triangular surface "
                        "grid of geometry {0}:\n"
                        " Min: {1:.2e}\n Max: {2:.2e}\n AVG: {3:.2e}\n"
                        " MED: {4:.2e}\n STD: {5:.2e}\n"
                    ).format(
                        i + 1,
                        elements_min[i],
                        elements_max[i],
                        elements_avg[i],
                        elements_med[i],
                        elements_std[i],
                    )
                )
            print(
                "The total number of nodes in all geometries is {0}.".format(
                    total_number_of_nodes
                )
            )
            print("\n", 70 * "*")
    else:
        stats_total = _get_mesh_stats(geometries.grid, verbose=verbose)

    return stats_total


def scale_mesh(geometry, scaling_factor):
    """
    Scale elements sizes of a grid of an optimus Geometry by a factor.
    The number of nodes remains intact.

    Parameters
    ----------
    geometry: optimus.geometry.common.Geometry
        The geometry to scale.
    scaling_factor : float
        Scaling factor of the sizes of the grid elements.
    """
    scaling = _convert_to_float(scaling_factor, "mesh scaling factor")
    vertices = geometry.grid.leaf_view.vertices * scaling
    elements = geometry.grid.leaf_view.elements
    scaled_grid = bempp.grid_from_element_data(vertices, elements)
    return _Geometry(scaled_grid, label=geometry.label + "_scaled")


def msh_from_string(geo_string):
    """Create a mesh from a string."""
    import os
    import subprocess

    gmsh_command = bempp.GMSH_PATH
    if gmsh_command is None:
        raise RuntimeError("Gmsh is not found. Cannot generate mesh")

    import tempfile

    geo, geo_name = tempfile.mkstemp(suffix=".geo", dir=bempp.TMP_PATH, text=True)
    geo_file = os.fdopen(geo, "w")
    msh_name = os.path.splitext(geo_name)[0] + ".msh"

    geo_file.write(geo_string)
    geo_file.close()

    fnull = open(os.devnull, "w")
    cmd = gmsh_command + " -2 " + geo_name
    try:
        subprocess.check_call(cmd, shell=True, stdout=fnull, stderr=fnull)
    except:
        print("The following command failed: " + cmd)
        fnull.close()
        raise
    os.remove(geo_name)
    fnull.close()
    return msh_name


def generate_grid_from_geo_string(geo_string):
    """Helper routine that implements the grid generation"""
    import os

    msh_name = msh_from_string(geo_string)
    grid = bempp.import_grid(msh_name)
    os.remove(msh_name)
    return grid


def plane_grid(x_axis_lims, y_axis_lims, rotation_axis, rotation_angle, element_size):
    """
    Return a 2D square shaped plane.

    x_axis_lims : list of two float numbers
        The bounding values along the x-axis of plane.
    y_axis_lims : list of two float numbers
        The bounding values along the y-axis of plane.
    rotation_axis : a list of size three populated with 0 or 1,
        It defines the axis of rotation so to construct the desired plane from an x-y plane.
    element_size : float
        Element size.
    """
    stub = """
    Point(1) = {ax1_lim1, ax2_lim1, 0, cl};
    Point(2) = {ax1_lim2, ax2_lim1, 0, cl};
    Point(3) = {ax1_lim2, ax2_lim2, 0, cl};
    Point(4) = {ax1_lim1, ax2_lim2, 0, cl};
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};
    Line Loop(1) = {1, 2, 3, 4};
    Plane Surface(2) = {1};
    Rotate {{rot_ax1, rot_ax2, rot_ax3}, {0, 0, 0}, rot_ang_rad} { Surface{2}; }
    Mesh.Algorithm = 2;
    """
    import sys

    if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
        pass
        # geometry = (f"ax1_lim1 = {x_axis_lims[0]};\nax1_lim2 = {x_axis_lims[1]};\n"
        #             + "ax2_lim1 = {y_axis_lims[0]};\nax2_lim2 = {y_axis_lims[1]};\n"
        #             + "rot_ax1 = {rotation_axis[0]};\nrot_ax2 = {rotation_axis[1]};\n"
        #             + "rot_ax3 = {rotation_axis[2]};\nrot_ang_rad = {rotation_angle};\ncl = {element_size};\n"
        #             + stub)
    else:
        geometry = (
            "ax1_lim1 = "
            + str(x_axis_lims[0])
            + ";\n"
            + "ax1_lim2 = "
            + str(x_axis_lims[1])
            + ";\n"
            + "ax2_lim1 = "
            + str(y_axis_lims[0])
            + ";\n"
            + "ax2_lim2 = "
            + str(y_axis_lims[1])
            + ";\n"
            + "rot_ax1 = "
            + str(rotation_axis[0])
            + ";\n"
            + "rot_ax2 = "
            + str(rotation_axis[1])
            + ";\n"
            + "rot_ax3 = "
            + str(rotation_axis[2])
            + ";\n"
            + "rot_ang_rad = "
            + rotation_angle
            + ";\n"
            + "cl = "
            + str(element_size)
            + ";\n"
            + stub
        )
    return generate_grid_from_geo_string(geometry)


def create_grid_points(resolution, plane_axes, plane_offset, bounding_box, mode):

    ax1_min, ax1_max, ax2_min, ax2_max = bounding_box
    if mode.lower() == "2d":
        plot_grid = np.mgrid[
            ax1_min : ax1_max : resolution[0] * 1j,
            ax2_min : ax2_max : resolution[1] * 1j,
        ]
        points_tmp = [np.ones(plot_grid[0].size) * plane_offset] * 3
        points_tmp[plane_axes[0]] = plot_grid[0].ravel()
        points_tmp[plane_axes[1]] = plot_grid[1].ravel()
        points = np.vstack((points_tmp,))
        plane = None

    elif mode.lower() == "3d":
        if 2 not in plane_axes:
            axis1_lims = bounding_box[0:2]
            axis2_lims = bounding_box[2:]
            rotation_axis = [0, 0, 1]
            rotation_angle = "2*Pi"
        elif 1 not in plane_axes:
            axis1_lims = bounding_box[0:2]
            axis2_lims = bounding_box[2:]
            rotation_axis = [1, 0, 0]
            rotation_angle = "Pi/2"
        elif 0 not in plane_axes:
            axis1_lims = bounding_box[2:]
            axis2_lims = bounding_box[0:2]
            rotation_axis = [0, 1, 0]
            rotation_angle = "-Pi/2"

        elem_len = np.min(
            [
                (axis1_lims[1] - axis1_lims[0]) / resolution[0],
                (axis2_lims[1] - axis2_lims[0]) / resolution[1],
            ]
        )

        plane = plane_grid(
            axis1_lims, axis2_lims, rotation_axis, rotation_angle, elem_len
        )
        points = plane.leaf_view.vertices

    return (points, plane)
