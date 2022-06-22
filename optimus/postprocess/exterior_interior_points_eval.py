
import numpy as np
import scipy.io as sio


def exterior_interior_points_eval(grid, xyz_field):
    """
    inputs
    ------
    grid : bempp.gridd
    xyz_field : `numpy.ndarray`
       array of shape: (3, n)

    Returns
    -------
    xyz_int : `list`
    xyz_ext : `numpy.ndarray`
       array xyz_field but with ... # FIXME
    n_int: `list`
    n_ext: `numpy.ndarray`  # NOTE Why all are lists and this is a boolean?
      array of booleans
    """
    n_field = xyz_field.shape[1]
    elements = grid.leaf_view.elements
    vertices = grid.leaf_view.vertices
    # number_of_elements = grid.leaf_view.entity_count(0) # Not Used

    # Retrieves elements from different groups # NOTE rename to elements_property? (it contains all elements)
    element_property = list(map(lambda x: x.domain,
                                grid.leaf_view.entity_iterator(0)))
    # Produce list of element properties
    element_properties = np.array(list(set(element_property)), dtype=np.int)
    print('Element groups are:')
    print(element_properties)

    # creates array of element topologies together with property numbers (Not used)
    # element_groups = np.vstack((element_property, elements)) # [4, number_of_elements]

    xyz_int = []
    n_int = []
    n_ext = np.full(n_field, True, dtype=bool)

    for element_property_value in element_properties:

        # Retrieves the block of elements corresponding to the property linked to the index i
        mask = element_property == element_property_value
        elements_trunc = elements[:, mask] # shape: (3, n)
        # Populate grid vertices matrices
        # NOTE elements_trunc is a 3D array, the indices of it are used to extract the value of
        #      the vertices.
        meshes = vertices[:, elements_trunc]
        # Obtain coordinates of triangular patch centroids through barycentric method
        xyz_center = meshes.mean(axis=1)

        normals, dS = get_normals_areas_from_patches(meshes)

        Omega = compute_solid_angle(xyz_center, normals, dS, xyz_field)

        # Depending on value of solid angle, determine whether field points lie
        # inside or outside the volume associated with mesh property
        # NOTE 0.5 is the theoretical value that should be used. In practice, due to the straight-edge elements,
        # some points which should be "inside" can be predicted as being "outside". 0.55 would be a heuristic
        # values which improved upon this on some test cases.
        n_int_tmp = np.array(Omega) > 0.5
        xyz_int.append(xyz_field[:, n_int_tmp])
        n_ext = n_ext & (n_int_tmp == False)
        n_int.append(n_int_tmp)

    xyz_ext = xyz_field[:, n_ext]

    return xyz_int, xyz_ext, n_int, n_ext

def get_normals_areas_from_patches(meshes):
    """

    inputs
    ------
    meshes: `numpy.ndarray`
       An array of shape (3, 3, n) containing the x,y,z vertices

    Returns
    -------
    normals : `numpy.ndarray`
    dS : `numpy.ndarray`
       Surface area for each element

    """

    # Compute matrix of vectors defining each triangular patch
    u = meshes[:, 1, :] - meshes[:, 0, :]
    v = meshes[:, 2, :] - meshes[:, 0, :]

    # Obtained cross product of u and v vectors
    # NOTE We need the three axisi=0 to multiply properly each column and keep the results as (3, n)
    u_cross_v = np.cross(u, v, axisa=0, axisb=0, axisc=0)
    # Obtain magnitude of above cross product
    u_cross_v_norm = np.linalg.norm(u_cross_v, axis=0)
    # Obtain outward pointing unit normal vectors for each patch
    normals = np.divide(u_cross_v, u_cross_v_norm)
    # Obtain surface area of each patch
    dS = 0.5 * u_cross_v_norm

    return normals, dS

def compute_solid_angle(surface_centers, surface_normals, surface_areas, field):
    """
    Calculate the solid angle as in:

    ..math: \Omega =  \frac{1}{4\pi} \frac{\hat{r} \cdot \hat{n} dS }{|r|^{2}}

    inputs
    ------
    surface_centers: `numpy.ndarray`
    surface_normals: `numpy.ndarray`
    surface_areas: `numpy.ndarray`
    field: `numpy.ndarray`

    """
    # solid angle array
    r = surface_centers[:,:, np.newaxis] - field[:, np.newaxis, :]
    r_norm = np.linalg.norm(r, axis=0)
    r_unit = np.divide(r, r_norm[np.newaxis, :, :])
    r_unit_dot_n = np.sum(r_unit * surface_normals[:, :, np.newaxis], axis=0)
    Omega = np.sum(r_unit_dot_n * surface_areas[:, np.newaxis] / r_norm**2, axis=0) / (4 * np.pi)

    return Omega
