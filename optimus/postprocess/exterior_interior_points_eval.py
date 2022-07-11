import numpy as np
from functools import partial
import time as _time
import multiprocessing as mp


def exterior_interior_points_eval(
    grid, xyz_field, solid_angle_tolerance=None, verbose=False
):
    """
    To evaluate whether a field point is within a domain or not using a solid angle method.

    Parameters
    ------------
    grid : bempp grid
        surface grid defining a domain
    xyz_field : numpy array of size 3xN
        field points to be evaluated if they are inside the volume defined by the surface grid or not
    solid_angle_tolerance : real/None
        the tolerance in solid angle method
    verbose : boolean
        to display the log or not

    Returns
    ------------
    xyz_int : numpy array 3xN
        coordinates of the interior points
    xyz_ext : numpy array 3xN
        coordinates of the exterior points
    xyz_boundary : numpy array 3xN
        coordinates of the points lie on the surface of the domain
    n_int : boolean numpy array 1xN
        indices of the interior points
    n_ext : boolean numpy array 1xN
        indices of the exterior points
    n_boundary : boolean numpy array 1xN
        indices of the surface points
    """
    elements = grid.leaf_view.elements
    vertices = grid.leaf_view.vertices
    number_of_elements = grid.leaf_view.entity_count(0)
    elem = list(grid.leaf_view.entity_iterator(0))

    element_property = np.zeros(number_of_elements, dtype=np.int)
    element_groups = np.zeros(shape=(4, number_of_elements), dtype=np.int)
    element_groups[1:4, :] = elements
    for i in range(number_of_elements):
        property_number = elem[i].domain
        element_property[i] = property_number
        element_groups[0, i] = property_number

    element_properties = np.array(list(set(element_property)), dtype=np.int)
    if verbose:
        print("Element groups are:")
        print(element_properties)

    xyz_int = []
    xyz_ext = []
    xyz_boundary = []
    n_int = []
    n_ext = np.full(xyz_field.shape[1], True, dtype=bool)
    n_boundary = []

    for i in range(element_properties.size):

        elements_trunc = elements[:, element_groups[0, :] == element_properties[i]]
        num_elem = elements_trunc.shape[1]

        xmesh = np.zeros(shape=(3, num_elem), dtype=float)
        ymesh = np.zeros(shape=(3, num_elem), dtype=float)
        zmesh = np.zeros(shape=(3, num_elem), dtype=float)
        # Populate grid vertices matrices
        for k in range(3):
            xmesh[k, :] = vertices[0, elements_trunc[k, :]]
            ymesh[k, :] = vertices[1, elements_trunc[k, :]]
            zmesh[k, :] = vertices[2, elements_trunc[k, :]]
        # Obtain coordinates of triangular patch centroids through barycentric method
        xcen = np.mean(xmesh, axis=0)
        ycen = np.mean(ymesh, axis=0)
        zcen = np.mean(zmesh, axis=0)

        # Preallocate matrix of vectors for triangular patches
        u = np.zeros(shape=(3, num_elem), dtype=float)
        v = np.zeros(shape=(3, num_elem), dtype=float)
        # Compute matrix of vectors defining each triangular patch
        u = np.array(
            [
                xmesh[1, :] - xmesh[0, :],
                ymesh[1, :] - ymesh[0, :],
                zmesh[1, :] - zmesh[0, :],
            ]
        )
        v = np.array(
            [
                xmesh[2, :] - xmesh[0, :],
                ymesh[2, :] - ymesh[0, :],
                zmesh[2, :] - zmesh[0, :],
            ]
        )
        u_cross_v = np.cross(u, v, axisa=0, axisb=0, axisc=0)
        u_cross_v_norm = np.linalg.norm(u_cross_v, axis=0)
        # Obtain outward pointing unit normal vectors for each patch
        normals = np.divide(u_cross_v, u_cross_v_norm)
        # Obtain surface area of each patch
        dS = 0.5 * u_cross_v_norm

        i_val = np.arange(0, xyz_field.shape[1])
        t0 = _time.time()
        N_workers = mp.cpu_count()
        func = partial(Omega_eval, xcen, ycen, zcen, xyz_field, normals, dS)
        pool = mp.Pool(N_workers)
        result = pool.starmap(func, zip(i_val))
        pool.close()
        t1 = _time.time() - t0
        if verbose:
            print("Time to complete solid angle field parallelisation: ", t1)
        Omega = np.hstack(result)
        if solid_angle_tolerance:
            n_int_tmp = Omega > 0.5 + solid_angle_tolerance
            n_boundary_tmp = (Omega > 0.5 - solid_angle_tolerance) & (
                Omega < 0.5 + solid_angle_tolerance
            )
            xyz_boundary.append(xyz_field[:, n_boundary_tmp])
            n_boundary.append(n_boundary_tmp)
            n_ext = n_ext & ((n_int_tmp == False) & (n_boundary_tmp == False))
        else:
            n_int_tmp = Omega > 0.5
            n_ext = n_ext & (n_int_tmp == False)

        xyz_int.append(xyz_field[:, n_int_tmp])
        n_int.append(n_int_tmp)

    xyz_ext = xyz_field[:, n_ext]

    return xyz_int, xyz_ext, xyz_boundary, n_int, n_ext, n_boundary


def Omega_eval(xcen, ycen, zcen, xyz_field, normals, dS, jj):

    r = np.array(
        [xcen - xyz_field[0, jj], ycen - xyz_field[1, jj], zcen - xyz_field[2, jj]]
    )
    r_norm = np.linalg.norm(r, axis=0)
    r_unit = np.divide(r, r_norm)
    r_unit_dot_n = np.sum(r_unit * normals, axis=0)
    Omega = np.sum(r_unit_dot_n * dS / r_norm**2) / (4 * np.pi)
    return Omega
