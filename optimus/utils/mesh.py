"""Get mesh statistics and scale mesh elements."""

import bempp.api as _bempp
import numpy as _np
from ..geometry.common import Geometry as _Geometry
from .conversions import convert_to_float as _convert_to_float


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
        _np.sqrt(
            (
                (
                    element.geometry.corners
                    - _np.roll(element.geometry.corners, 1, axis=1)
                )
                ** 2
            ).sum(axis=0)
        )
        for element in elements
    ]
    elements_min = _np.min(element_size)
    elements_max = _np.max(element_size)
    elements_avg = _np.mean(element_size)
    elements_med = _np.median(element_size)
    elements_std = _np.std(element_size)
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
        total_number_of_nodes = _np.sum(number_of_nodes)

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
    scaled_grid = _bempp.grid_from_element_data(vertices, elements)
    return _Geometry(scaled_grid, label=geometry.label + "_scaled")
