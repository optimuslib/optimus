"""Postprocessing functions for the topology of the geometry."""

import numpy as _np
import time as _time


def calculate_bounding_box(domains_grids, plane_axes):
    """
    Calculate the bounding box for a set of grids.

    Parameters
    ----------
    domains_grids : list[bempp.api.Grid]
        The grids of the subdomains.
    plane_axes : tuple[int]
        The two indices of the boundary plane.
        Possible values are 0,1,2 for x,y,z axes, respectively.

    Returns
    -------
    bounding_box: list[float]
        Bounding box specifying the visualisation section on the plane_axis
        as a list: axis1_min, axis1_max, axis2_min, axis2_max.
    """

    if not isinstance(domains_grids, list):
        domains_grids = list(domains_grids)

    ax1_min, ax1_max, ax2_min, ax2_max = 0, 0, 0, 0
    for grid in domains_grids:
        ax1_min = _np.min([ax1_min, grid.bounding_box[0][plane_axes[0]].min()])
        ax1_max = _np.max([ax1_max, grid.bounding_box[1][plane_axes[0]].max()])
        ax2_min = _np.min([ax2_min, grid.bounding_box[0][plane_axes[1]].min()])
        ax2_max = _np.max([ax2_max, grid.bounding_box[1][plane_axes[1]].max()])
    bounding_box = [ax1_min, ax1_max, ax2_min, ax2_max]

    return bounding_box


def find_int_ext_points(domains_grids, points, verbose):
    """
    Identify the interior and exterior points w.r.t. each grid.

    Parameters
    ----------
    domains_grids : list[bempp.api.Grid, None]
        The grids of the subdomains.
    points : numpy.ndarray
        The field points. The size of the array should be (3,N).
    verbose : boolean
        Display the logs.

    Returns
    -------
    points_interior : list[numpy.ndarray, None]
        A list of numpy arrays of size (3, N_interior), where
        element i of the list is an array of coordinates of the
        interior points for domain i, i=1,...,no_subdomains.
        The element is None if the input grid is None (e.g. inactivated interfaces).
        The element has size (3, 0) if there are no points interior to the grid.
    points_exterior : numpy.ndarray
        An array of size (3, N_exterior) with visualisation points in the exterior domain
        The element has size (3, 0) if there are no points exterior to the grid.
    points_boundary : list[numpy.ndarray, None]
        A list of arrays of size (3,N) with visualisation points at, or close to,
        grid boundaries.
        The element is None if the input grid is None (e.g. inactivated interfaces).
        The element has size (3, 0) if there are no points on the boundary to the grid.
    index_interior : list[numpy.ndarray, None]
        A list of arrays of size (N_points,) with boolean values,
        identifying the interior points for each domain.
        The element is None if the input grid is None (e.g. inactivated interfaces).
    index_exterior : numpy.ndarray
        An array of size (N_points,) with boolean values,
        identifying the exterior points.
    index_boundary : list[numpy.ndarray, None]
        A list of arrays of size (N_points,) with boolean values,
        identifying the boundary points.
        The element is None if the input grid is None (e.g. inactivated interfaces).
    """

    from .solid_angle_method import exterior_interior_points_eval
    from optimus import global_parameters

    postprocess_params = global_parameters.postprocessing

    points_interior = []
    points_boundary = []
    idx_interior = []
    idx_boundary = []
    idx_exterior = _np.full(points.shape[1], True)

    if verbose:
        start_time_int_ext = _time.time()
        print(
            "\n Identifying the exterior and interior points Started at: ",
            _time.strftime("%a, %d %b %Y %H:%M:%S", _time.localtime()),
        )
    else:
        start_time_int_ext = None

    for grid in domains_grids:
        if grid is None:
            points_interior.append(None)
            idx_interior.append(None)
            if postprocess_params.solid_angle_tolerance:
                points_boundary.append(None)
                idx_boundary.append(None)
        else:
            # noinspection PyTypeChecker
            (
                points_interior_temp,
                points_exterior_temp,
                points_boundary_temp,
                idx_interior_temp,
                idx_exterior_temp,
                idx_boundary_temp,
            ) = exterior_interior_points_eval(
                grid=grid,
                points=points,
                solid_angle_tolerance=postprocess_params.solid_angle_tolerance,
                verbose=verbose,
            )
            if len(points_interior_temp) != 1:
                raise ValueError(
                    "The domain grid needs to have a single geometrical tag."
                )
            points_interior.append(points_interior_temp[0])
            idx_interior.append(idx_interior_temp[0])
            idx_exterior[idx_exterior_temp == False] = False
            if postprocess_params.solid_angle_tolerance:
                points_boundary.append(points_boundary_temp[0])
                idx_boundary.append(idx_boundary_temp[0])

    if verbose:
        end_time_int_ext = _time.time()
        print(
            "\n Identifying the exterior and interior points "
            "Finished... Duration in secs: ",
            end_time_int_ext - start_time_int_ext,
        )

    points_exterior = points[:, idx_exterior]
    return (
        points_interior,
        points_exterior,
        points_boundary,
        idx_interior,
        idx_exterior,
        idx_boundary,
    )


def domain_edge(grids, plane_axes, plane_offset):
    """
    Determine the points at the edges of the domains by computing the intersection of
    the grid triangular elements with planes of constant x, y or z. The intersection
    points are then sorted by proximity to one another.

    Parameters
    ----------
    grids : list[bempp.api.Grid]
        A model object which has solution attributes already computed.
    plane_axes : list[int]
        The axes of the plane.
    plane_offset : float
        Offset of the visualisation plane defined along the third axis.

    Returns
    -------
    domains_edge_points : list[numpy.ndarray]
        list of numpy arrays of coordinates of points on the edges
    """

    import warnings
    from itertools import combinations

    warnings.filterwarnings("ignore")

    comb = _np.array(list(combinations([0, 1, 2], 2)))
    axis_0 = plane_axes[0]
    axis_1 = plane_axes[1]
    axes = (0, 1, 2)
    axis_2 = list(set(axes) - set(plane_axes))

    domains_edge_points = list()

    # Find points at which the triangular elements intersect the plane of constant x, y
    # or z, for each subdomain.
    for grid in grids:
        vertices = grid.leaf_view.vertices
        elements = grid.leaf_view.elements

        axis_0_patch = vertices[axis_0, elements]
        axis_1_patch = vertices[axis_1, elements]
        axis_2_patch = vertices[axis_2, elements]

        axis_0_intersect = list()
        axis_1_intersect = list()

        for i in range(comb.shape[0]):
            plane_crossing_condition = (axis_2_patch[comb[i, 1], :] - plane_offset) * (
                axis_2_patch[comb[i, 0], :] - plane_offset
            )
            idx = plane_crossing_condition <= 0
            denominator = axis_2_patch[comb[i, 1], idx] - axis_2_patch[comb[i, 0], idx]

            line_param = (plane_offset - axis_2_patch[comb[i, 0], idx]) / denominator
            axis_0_values = axis_0_patch[comb[i, 0], idx] + line_param * (
                axis_0_patch[comb[i, 1], idx] - axis_0_patch[comb[i, 0], idx]
            )
            axis_1_values = axis_1_patch[comb[i, 0], idx] + line_param * (
                axis_1_patch[comb[i, 1], idx] - axis_1_patch[comb[i, 0], idx]
            )
            if len(axis_0_values):
                axis_0_intersect.append(axis_0_values[~_np.isnan(axis_0_values)])
                axis_1_intersect.append(axis_1_values[~_np.isnan(axis_1_values)])

        if axis_0_intersect and axis_1_intersect:
            axis_0_edge = _np.concatenate(axis_0_intersect)
            axis_1_edge = _np.concatenate(axis_1_intersect)

            # Sort above intersection points by proximity to one another.
            axis_0_axis_1_vstack = _np.vstack((axis_0_edge, axis_1_edge))
            axis_0_axis_1_unique = _np.unique(
                _np.round(axis_0_axis_1_vstack, decimals=16), axis=1
            )
            axis_0_unique = axis_0_axis_1_unique[0, :]
            axis_1_unique = axis_0_axis_1_unique[1, :]
            number_of_edge_points = len(axis_0_unique) + 1
            axis_0_edge_sorted = _np.zeros(number_of_edge_points, dtype=float)
            axis_1_edge_sorted = _np.zeros(number_of_edge_points, dtype=float)
            axis_0_edge_sorted[0] = axis_0_unique[0]
            axis_1_edge_sorted[0] = axis_1_unique[0]
            axis_0_unique = _np.delete(axis_0_unique, 0)
            axis_1_unique = _np.delete(axis_1_unique, 0)

            for i in range(number_of_edge_points - 2):
                distance = _np.sqrt(
                    (axis_0_edge_sorted[i] - axis_0_unique) ** 2
                    + (axis_1_edge_sorted[i] - axis_1_unique) ** 2
                )
                idx = distance == _np.min(distance[distance != 0])
                if i < number_of_edge_points - 2:
                    axis_0_edge_sorted[i + 1] = axis_0_unique[idx]
                    axis_1_edge_sorted[i + 1] = axis_1_unique[idx]
                    domains_edge_points.append(
                        _np.array(
                            [
                                [axis_0_edge_sorted[i], axis_0_edge_sorted[i + 1]],
                                [axis_1_edge_sorted[i], axis_1_edge_sorted[i + 1]],
                            ]
                        )
                    )
                axis_0_unique = axis_0_unique[~idx]
                axis_1_unique = axis_1_unique[~idx]

            axis_0_edge_sorted[number_of_edge_points - 1] = axis_0_edge_sorted[0]
            axis_1_edge_sorted[number_of_edge_points - 1] = axis_1_edge_sorted[0]

            domains_edge_points.append(
                _np.array(
                    [
                        [
                            axis_0_edge_sorted[number_of_edge_points - 2],
                            axis_0_edge_sorted[number_of_edge_points - 1],
                        ],
                        [
                            axis_1_edge_sorted[number_of_edge_points - 2],
                            axis_1_edge_sorted[number_of_edge_points - 1],
                        ],
                    ]
                )
            )

    return domains_edge_points


def create_regions(points, index_exterior, indices_interior, indices_boundary):
    """
    Create region labels for the subdomain and boundary points.

    Each point in a (3,N) numpy array needs to be assigned to a
    unique subdomain or interface. The regions are "subdomain i" and
    "interface j". This algorithms takes index arrays as input.

    Parameters
    ----------
    points : numpy.ndarray
        The field points. The size of the array should be (3, N_points).
    index_exterior : numpy.ndarray
        An array of size (N_points,) with boolean values,
        identifying the exterior points.
    indices_interior : list[numpy.ndarray]
        A list of arrays of size (N_points,) with boolean values,
        identifying the interior points for each interior subdomain.
        The element is None for inactivated subdomains.
    indices_boundary : list[numpy.ndarray]
        A list of arrays of size (N_points,) with boolean values,
        identifying the boundary points for each interface.
        The element is None for inactivated interfaces.

    Returns
    -------
    regions : numpy.ndarray
        An array of size (N_points,) with integer values 0,1,2,...,N_regions.
    regions_legend : list[str]
        A list of strings identifying the name of the region.
    """

    n_points = points.shape[1]

    no_regions = _np.zeros(n_points, dtype=int)
    for index in [index_exterior] + indices_interior + indices_boundary:
        if index is not None:
            no_regions += index.astype(int)
    if not _np.all(no_regions == 1):
        raise AssertionError("Each point should be in exactly one region.")

    region_value = -1
    regions = _np.full(n_points, region_value, dtype=int)
    region_legend = []

    region_value += 1
    regions[index_exterior] = region_value
    region_legend.append("exterior")

    for n, index in enumerate(indices_interior):
        if index is not None and index.any():
            region_value += 1
            regions[index] = region_value
            region_legend.append("subdomain " + str(n + 1))

    for n, index in enumerate(indices_boundary):
        if index is not None and index.any():
            region_value += 1
            regions[index] = region_value
            region_legend.append("interface " + str(n + 1))

    if _np.any(regions == -1):
        raise AssertionError("Each point should be in exactly one region.")

    return regions, region_legend
