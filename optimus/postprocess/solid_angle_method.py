"""The solid angle method for exterior-interior point evaluation."""

import numpy as _np
from functools import partial as _partial
import time as _time
import multiprocessing as _mp


def exterior_interior_points_eval(grid, points, solid_angle_tolerance, verbose=False):
    """
    Evaluate whether a field point is interior or exterior to a domain,
    using a solid angle method.

    Parameters
    ----------
    grid : bempp.api.Grid
        surface grid defining a domain
    points : numpy.ndarray
        Field points to be evaluated if they are inside the volume defined by
        the surface grid or not. Array of size (3,N).
    solid_angle_tolerance : float, None
        the tolerance in solid angle method. If set to None/zero boundary points
        are not identified, otherwise they are.
    verbose : boolean
        display the log

    Returns
    -------
    points_interior : list[numpy.ndarray]
        Array of size (3, N_interior) with coordinates of the interior points.
        The list has elements for each geometrical tag in the mesh.
    points_exterior : numpy.ndarray
        Array of size (3, N_exterior) with coordinates of the exterior points.
    points_boundary : list[numpy.ndarray]
        Array of size (3, N_boundary) with coordinates of the points that lie on the
        surface of the domain.
        The list has elements for each geometrical tag in the mesh.
    index_interior : list[numpy.ndarray]
        Array of size (N_points,) with boolean values identifying the interior points.
        The list has elements for each geometrical tag in the mesh.
    index_exterior : numpy.ndarray
        Array of size (N_points,) with boolean values identifying the exterior points.
    index_boundary : list[numpy.ndarray]
        Array of size (N_points,) with boolean values identifying the surface points.
        The list has elements for each geometrical tag in the mesh.
    """

    elements = grid.leaf_view.elements
    vertices = grid.leaf_view.vertices
    number_of_elements = grid.leaf_view.entity_count(0)

    points_interior = []
    points_exterior = []
    points_boundary = []
    index_interior = []
    index_exterior = _np.full(points.shape[1], True, dtype=bool)
    index_boundary = []
    
    elements_x_coordinate = _np.zeros(shape=(3, number_of_elements), dtype=float)
    elements_y_coordinate = _np.zeros(shape=(3, number_of_elements), dtype=float)
    elements_z_coordinate = _np.zeros(shape=(3, number_of_elements), dtype=float)
    # Populate grid vertices matrices
    for k in range(3):
        elements_x_coordinate[k, :] = vertices[0, elements[k, :]]
        elements_y_coordinate[k, :] = vertices[1, elements[k, :]]
        elements_z_coordinate[k, :] = vertices[2, elements[k, :]]
    # Obtain coordinates of triangular elements centroielements_surface_area
    # through barycentric method.
    elements_barycent_x_coordinate = _np.mean(elements_x_coordinate, axis=0)
    elements_barycent_y_coordinate = _np.mean(elements_y_coordinate, axis=0)
    elements_barycent_z_coordinate = _np.mean(elements_z_coordinate, axis=0)

    # Compute matrix of vectors defining each triangular elements
    elements_u_coordinate = _np.array(
        [
            elements_x_coordinate[1, :] - elements_x_coordinate[0, :],
            elements_y_coordinate[1, :] - elements_y_coordinate[0, :],
            elements_z_coordinate[1, :] - elements_z_coordinate[0, :],
        ]
    )
    elements_v_coordinate = _np.array(
        [
            elements_x_coordinate[2, :] - elements_x_coordinate[0, :],
            elements_y_coordinate[2, :] - elements_y_coordinate[0, :],
            elements_z_coordinate[2, :] - elements_z_coordinate[0, :],
        ]
    )
    elements_u_cross_v = _np.cross(
        elements_u_coordinate, elements_v_coordinate, axisa=0, axisb=0, axisc=0
    )
    elements_u_cross_v_norm = _np.linalg.norm(elements_u_cross_v, axis=0)
    # Obtain outward pointing unit normal vectors for each elements
    normals = _np.divide(elements_u_cross_v, elements_u_cross_v_norm)
    # Obtain surface area of each elements
    elements_surface_area = 0.5 * elements_u_cross_v_norm
    
    start_time = _time.time()
    N_workers = _mp.cpu_count()
    parallelised_compute_solid_angle = _partial(
        compute_solid_angle,
        elements_barycent_x_coordinate,
        elements_barycent_y_coordinate,
        elements_barycent_z_coordinate,
        points,
        normals,
        elements_surface_area,
    )
    pool = _mp.Pool(N_workers)
    result = pool.starmap(
        parallelised_compute_solid_angle, zip(_np.arange(0, points.shape[1]))
    )
    pool.close()
    end_time = _time.time() - start_time
    if verbose:
        print("Time to complete solid angle field parallelisation: ", end_time)
    solid_angle = _np.hstack(result)
    if solid_angle_tolerance:
        index_interior_tmp = solid_angle > 0.5 + solid_angle_tolerance
        index_boundary_tmp = (solid_angle > 0.5 - solid_angle_tolerance) & (
            solid_angle < 0.5 + solid_angle_tolerance
        )
        points_boundary.append(points[:, index_boundary_tmp])
        index_boundary.append(index_boundary_tmp)
        index_exterior = index_exterior & (
            (index_interior_tmp == False) & (index_boundary_tmp == False)
        )
    else:
        index_interior_tmp = solid_angle > 0.5
        index_exterior = index_exterior & (index_interior_tmp == False)

    points_interior.append(points[:, index_interior_tmp])
    index_interior.append(index_interior_tmp)

    points_exterior = points[:, index_exterior]

    return (
        points_interior,
        points_exterior,
        points_boundary,
        index_interior,
        index_exterior,
        index_boundary,
    )


def compute_solid_angle(
    elements_barycent_x_coordinate,
    elements_barycent_y_coordinate,
    elements_barycent_z_coordinate,
    points,
    normals,
    elements_surface_area,
    point_index,
):
    """Compute the solid angle value for a triangular element.

    Parameters
    ----------
    elements_barycent_x_coordinate : numpy.ndarray

    elements_barycent_y_coordinate : numpy.ndarray

    elements_barycent_z_coordinate : numpy.ndarray

    points : numpy.ndarray

    normals : numpy.ndarray

    elements_surface_area : numpy.ndarray

    point_index : numpy.ndarray

    Returns
    -------
    solid_angle : numpy.ndarray

    """

    elements_barycen_dist = _np.array(
        [
            elements_barycent_x_coordinate - points[0, point_index],
            elements_barycent_y_coordinate - points[1, point_index],
            elements_barycent_z_coordinate - points[2, point_index],
        ]
    )
    elements_barycen_dist_norm = _np.linalg.norm(elements_barycen_dist, axis=0)
    elements_barycen_dist_normalised = _np.divide(
        elements_barycen_dist, elements_barycen_dist_norm
    )
    elements_barycen_dist_projected = _np.sum(
        elements_barycen_dist_normalised * normals, axis=0
    )
    solid_angle = _np.sum(
        elements_barycen_dist_projected
        * elements_surface_area
        / elements_barycen_dist_norm**2
    ) / (4 * _np.pi)
    return solid_angle