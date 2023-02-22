from unicodedata import decimal
import numpy as _np
import bempp.api as _bempp
import time as _time
from ..utils.linalg import normalize_vector as _normalize_vector


class PostProcess:
    def __init__(self, model, verbose=False):
        """Create an optimus postprocess object with the specified parameters.

        Parameters
        ----------
        model: optimus.Model
            The model object must include the solution vectors, i.e.,
            the solve() function must have been executed.
        verbose: boolean
            Display the logs.

        """

        self.verbose = verbose
        self.model = model
        self.domains_grids = [
            model.geometry[n_sub].grid for n_sub in range(model.n_subdomains)
        ]

    def create_computational_grid(self, **kwargs):
        """Create the grid on which to calculate the pressure field.
        Needs to be overridden by specific source type.

        Parameters
        ----------
        kwargs : dict
            Options to be specified for different types of postprocessing.

        """

        raise NotImplementedError

    def compute_fields(self):
        """Calculate the pressure field in the specified locations.Needs to be overridden by specific source type."""

        raise NotImplementedError

    def print_parameters(self):
        """Display parameters used for visualisation."""

        print("\n", 70 * "*")
        if hasattr(self, "points"):
            print("\n number of visualisation points: ", self.points.shape[1])
        if hasattr(self, "resolution") and hasattr(self, "bounding_box"):
            print("\n resolution in number of points: ", self.resolution)
            print(
                "\n resolution in ppi (diagonal ppi): %d "
                % ppi_calculator(self.bounding_box, self.resolution)
            )
        if hasattr(self, "plane_axes") and hasattr(self, "plane_offset"):
            print("\n 2D plane axes: ", self.plane_axes)
            print("\n the offset of 2D plane along the 3rd axis: ", self.plane_offset)
        if hasattr(self, "bounding_box"):
            print(
                "\n bounding box (2D frame) points: ",
                [self.bounding_box[0], self.bounding_box[2]],
                [self.bounding_box[1], self.bounding_box[3]],
            )
        print("\n", 70 * "*")


def calculate_bounding_box(domains_grids, plane_axes):
    """Calculate the bounding box for a set of grids.

    Parameters
    ----------
    domains_grids : list optimus.Grid
        The grids of the subdomains.
    plane_axes : tuple int
        The two indices of the boundary plane.
        Possible values are 0,1,2 for x,y,z axes, respectively.

    Returns
    -------
    bounding_box: list float
        Bounding box specifying the visualisation section on the plane_axis as a list: axis1_min, axis1_max, axis2_min, axis2_max
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
    """Identify the interior and exterior points w.r.t. each grid.

    Parameters
    ----------
    domains_grids : list optimus.Grid
        The grids of the subdomains.
    points : numpy.ndarray
        The field points. The size of the array should be (3,N).
    verbose : boolean
        Display the logs.

    Returns
    -------
    points_interior : list numpy.ndarray
        A list of numpy arrays of size (3,N), where
        element i of the list is an array of coordinates of the
        interior points for domain i, i=1,...,no_subdomains.
    points_exterior : numpy.ndarray
        An array of size (3,N) with visualisation points in the exterior domain
    points_boundary : list numpy.ndarray
        A list of arrays of size (3,N) with visualisation points at, or close to,
        grid boundaries.
    index_interior : list numpy.ndarray
        A list of arrays of size (1,N) with boolean values,
        identifying the interior points for each domain.
    index_exterior : numpy.ndarray
        An array of size (1,N) with boolean values,
        identifying the exterior points.
    index_boundary : list numpy.ndarray
        A list of arrays of size (1,N) with boolean values,
        identifying the boundary points.
    """

    from .exterior_interior_points_eval import exterior_interior_points_eval
    from optimus import global_parameters

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
            solid_angle_tolerance=global_parameters.postprocessing.solid_angle_tolerance,
            verbose=verbose,
        )
        points_interior.append(points_interior_temp[0])
        idx_interior.append(idx_interior_temp[0])
        idx_exterior[idx_exterior_temp == False] = False
        if global_parameters.postprocessing.solid_angle_tolerance:
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


def compute_pressure_fields(
    model,
    points,
    points_exterior,
    index_exterior,
    points_interior,
    index_interior,
    points_boundary,
    index_boundary,
    verbose,
):
    """Calculate the scattered and total pressure fields for visualisation.

    Parameters
    ----------
    model : optimus.Model
        A model object which has solution attributes already computed.
    points : numpy.ndarray
        An array of size (3,N) with the visualisation points.
    points_exterior : numpy.ndarray
        An array of size (3,N) with the visualisation points in the exterior domain.
    index_exterior : numpy.ndarray
        An array of size (1,N) with boolean values indentifying the exterior points.
    points_interior : list numpy.ndarray
        A list of arrays of size (3,N), where
        element i of the list is an array of coordinates of the
        interior points for domain i, i=1,...,no_subdomains
    index_interior : list numpy.ndarray
        A list of arrays of size (1,N) with boolean values indentifying
        the interior points.
    points_boundary : list numpy.ndarray
        A list of arrays of size (3,N), where
        element i of the list is an array of coordinates of the
        boundary points for domain i, i=1,...,no_subdomains
    index_boundary : list numpy.ndarray
        A list of boolean arrays of size (1,N),
        identifying the boundary points.
    verbose : bool
        Display the logs.

    Returns
    -------
    total_field : numpy.ndarray
        An array of size (1,N) with complex values of the total pressure field.
    scattered_field : numpy.ndarray
        An array of size (1,N) with complex values of the scatterd pressure field.
    incident_exterior_field : numpy.ndarray
        An array of size (1,N) with complex values of the incident pressure field
        in the exterior domain.

    """

    from ..utils.generic import chunker, bold_ul_text
    from optimus import global_parameters

    if model.formulation == 'analytical':
        return compute_analytical_pressure_fields(
            model,
            points,
            points_exterior,
            index_exterior,
            points_interior,
            index_interior,
            points_boundary,
            index_boundary,
        )

    if global_parameters.postprocessing.assembly_type.lower() in [
        "h-matrix",
        "hmat",
        "h-mat",
        "h_mat",
        "h_matrix",
    ]:
        _bempp.global_parameters.hmat.eps = global_parameters.postprocessing.hmat_eps
        _bempp.global_parameters.hmat.max_rank = (
            global_parameters.postprocessing.hmat_max_rank
        )
        _bempp.global_parameters.hmat.max_block_size = (
            global_parameters.postprocessing.hmat_max_block_size
        )
        _bempp.global_parameters.assembly.potential_operator_assembly_type = "hmat"
    elif global_parameters.postprocessing.assembly_type.lower() == "dense":
        _bempp.global_parameters.assembly.potential_operator_assembly_type = "dense"
    else:
        raise ValueError(
            "Supported operator assembly methods are "
            + bold_ul_text("dense")
            + " and "
            + bold_ul_text("hmat")
        )

    _bempp.global_parameters.quadrature.near.single_order = (
        global_parameters.postprocessing.quadrature_order
    )

    start_time_pot_ops = _time.time()
    if verbose:
        print(
            "\n Calculating the interior and exterior potential operators Started at: ",
            _time.strftime("%a, %d %b %Y %H:%M:%S", _time.localtime()),
        )

    total_field = _np.full(points.shape[1], _np.nan, dtype=complex)
    scattered_field = _np.full(points.shape[1], _np.nan, dtype=complex)
    incident_exterior_field = _np.full(points.shape[1], _np.nan, dtype=complex)

    if index_exterior.any():
        exterior_values = _np.zeros((1, points_exterior.shape[1]), dtype="complex128")
        ext_calc_flag = True
    else:
        exterior_values = None
        ext_calc_flag = False

    if index_boundary:
        bound_calc_flag = True
    else:
        bound_calc_flag = False

    i = 0
    for (solution_pair, space, interior_point, interior_idx, interior_material,) in zip(
        chunker(model.solution, 2),
        model.space,
        points_interior,
        index_interior,
        model.material_interior,
    ):
        if verbose:
            print("Calculating the fields of Domain {0}".format(i + 1))
            print(
                interior_point.shape,
                interior_idx.shape,
                interior_material.compute_wavenumber(model.source.frequency),
                interior_material,
            )

        if interior_idx.any():
            pot_int_sl = _bempp.operators.potential.helmholtz.single_layer(
                space,
                interior_point,
                interior_material.compute_wavenumber(model.source.frequency),
            )
            pot_int_dl = _bempp.operators.potential.helmholtz.double_layer(
                space,
                interior_point,
                interior_material.compute_wavenumber(model.source.frequency),
            )
            rho_ratio = interior_material.density / model.material_exterior.density
            interior_value = (
                pot_int_sl * solution_pair[1] * rho_ratio
                - pot_int_dl * solution_pair[0]
            )
            total_field[interior_idx] = interior_value.ravel()

        if ext_calc_flag:
            pot_ext_sl = _bempp.operators.potential.helmholtz.single_layer(
                space,
                points_exterior,
                model.material_exterior.compute_wavenumber(model.source.frequency),
            )
            pot_ext_dl = _bempp.operators.potential.helmholtz.double_layer(
                space,
                points_exterior,
                model.material_exterior.compute_wavenumber(model.source.frequency),
            )
            exterior_values += (
                -pot_ext_sl * solution_pair[1] + pot_ext_dl * solution_pair[0]
            )

        i += 1

        if verbose:
            end_time_pot_ops = _time.time()
            print(
                "\n Calculating the interior and exterior potential operators "
                "Finished... Duration in secs: ",
                end_time_pot_ops - start_time_pot_ops,
            )

    if ext_calc_flag:
        start_time_pinc = _time.time()
        if verbose:
            print(
                "\n Calculating the incident field Started at: ",
                _time.strftime("%a, %d %b %Y %H:%M:%S", _time.localtime()),
            )

        incident_exterior = model.source.pressure_field(
            model.material_exterior, points_exterior
        )

        end_time_pinc = _time.time()
        if verbose:
            print(
                "\n Calculating the incident field Finished... Duration in secs: ",
                end_time_pinc - start_time_pinc,
            )
        incident_exterior_field[index_exterior] = incident_exterior.ravel()
        scattered_field[index_exterior] = exterior_values.ravel()
        total_field[index_exterior] = (
            scattered_field[index_exterior] + incident_exterior_field[index_exterior]
        )

    if bound_calc_flag:
        for subdomain_number in range(model.n_subdomains):

            grid = model.geometry[subdomain_number].grid
            dirichlet_solution = model.solution[2 * subdomain_number].coefficients
            subdomain_boundary_points = points_boundary[subdomain_number]

            total_field[index_boundary[subdomain_number]] = compute_pressure_boundary(
                grid, subdomain_boundary_points, dirichlet_solution
            )

    return total_field, scattered_field, incident_exterior_field

def compute_analytical_pressure_fields(
    model,
    points,
    points_exterior,
    index_exterior,
    points_interior,
    index_interior,
    points_boundary,
    index_boundary,
):
    """Calculate the scattered and total pressure fields for visualisation
    in the analytical model.

    Parameters
    ----------
    model : optimus.Model
        A model object which has solution attributes already computed.
    points : numpy.ndarray
        An array of size (3,N) with the visualisation points.
    points_exterior : numpy.ndarray
        An array of size (3,N) with the visualisation points in the exterior domain.
    index_exterior : numpy.ndarray
        An array of size (1,N) with boolean values indentifying the exterior points.
    points_interior : list numpy.ndarray
        A list of arrays of size (3,N), where
        element i of the list is an array of coordinates of the
        interior points for domain i, i=1,...,no_subdomains
    index_interior : list numpy.ndarray
        A list of arrays of size (1,N) with boolean values indentifying
        the interior points.
    points_boundary : list numpy.ndarray
        A list of arrays of size (3,N), where
        element i of the list is an array of coordinates of the
        boundary points for domain i, i=1,...,no_subdomains
    index_boundary : list numpy.ndarray
        A list of boolean arrays of size (1,N),
        identifying the boundary points.

    Returns
    -------
    total_field : numpy.ndarray
        An array of size (1,N) with complex values of the total pressure field.
    scattered_field : numpy.ndarray
        An array of size (1,N) with complex values of the scatterd pressure field.
    incident_exterior_field : numpy.ndarray
        An array of size (1,N) with complex values of the incident pressure field
        in the exterior domain.
    """
    from scipy.special import sph_jn, sph_yn, eval_legendre

    total_field = _np.full(points.shape[1], _np.nan, dtype=complex)
    scattered_field = _np.full(points.shape[1], _np.nan, dtype=complex)
    incident_exterior_field = _np.full(points.shape[1], _np.nan, dtype=complex)

    k_ext = model.material_exterior.compute_wavenumber(model.source.frequency)
    k_int = model.material_interior.compute_wavenumber(model.source.frequency)
    
    rho_ext = model.material_exterior.density
    rho_int = model.material_interior.density
    
    rho = rho_int / rho_ext
    k = k_ext / k_int
    n_iter = model.interior_coefficients.size

    #
    # Interior
    #
    pi = points_interior[0]
    ii = index_interior[0]
    if ii.any():
        radial_space = _np.linalg.norm(pi, axis=0)
        directional_space = _np.dot(model.source.direction_vector, pi)
        directional_space /= radial_space

        jn, djn = _np.array(
            list(zip(*[sph_jn(n_iter - 1, k_int * r) for r in radial_space]))
        )

        legendre = _np.array(
            [eval_legendre(n, directional_space) for n in range(n_iter)]
        )

        total_field[ii] = _np.dot(
            model.interior_coefficients, jn.T * legendre
        )

    #
    # Exterior
    #
    pe = points_exterior
    ie = index_exterior
    if ie.any():
        radial_space = _np.linalg.norm(pe, axis=0)
        directional_space = _np.dot(model.source.direction_vector, pe)
        directional_space /= radial_space

        jn, djn = _np.array(
            list(zip(*[sph_jn(n_iter - 1, k_ext * r) for r in radial_space]))
        )
        yn, dyn = _np.array(
            list(zip(*[sph_yn(n_iter - 1, k_ext * r) for r in radial_space]))
        )
        h1n, dh1n = jn.T + 1j * yn.T, djn.T + 1j * dyn.T

        legendre = _np.array(
            [eval_legendre(n, directional_space) for n in range(n_iter)]
        )
        
        scattered_field[ie] = _np.dot(
            model.scattered_coefficients, h1n * legendre
        )

        incident_exterior_field[ie] = _np.dot(
            _np.array([(2*n + 1) * 1j**n for n in range(n_iter)])
            ,
            jn.T * legendre
        )

        total_field[ie] = scattered_field[ie] + incident_exterior_field[ie]

    #
    # Boundary
    #
    pb = points_boundary[0]
    ib = index_boundary[0]
    if ib.any():
        # We use the interior field to compute the boundary points
        radial_space = _np.linalg.norm(pb, axis=0)
        directional_space = _np.dot(model.source.direction_vector, pb)
        directional_space /= radial_space

        jn, djn = _np.array(
            list(zip(*[sph_jn(n_iter - 1, k_int * r) for r in radial_space]))
        )

        legendre = _np.array(
            [eval_legendre(n, directional_space) for n in range(n_iter)]
        )

        total_field[ib] = _np.dot(model.interior_coefficients, jn.T * legendre)

    return total_field, scattered_field, incident_exterior_field

def compute_pressure_boundary(grid, boundary_points, dirichlet_solution):
    """Calculate pressure for points near or at the boundary of a domain. When the solid
    angle associated with a boundary vertex is below 0.1, it is assumed to lie on the
    boundary.

    Parameters
    ----------
    grid : bempp.api.Grid
        The surface mesh of bempp.
    boundary_points : numpy.ndarray
        An array of size (3,N) with the coordinates of vertices
        on the domain boundary.
    dirichlet_solution : numpy.ndarray
        An array of size (N,) with the Dirichlet component of the
        solution vector on the boundary.

    Returns
    -------
    total_boundary_pressure : numpy.ndarray
        An array of size (N,) with complex values of the pressure field.

    """

    vertices = grid.leaf_view.vertices
    elements = grid.leaf_view.elements
    centroids = _np.mean(vertices[:, elements], axis=1)

    # Initialise arrays with None-values for the element indices
    n = boundary_points.shape[1]
    element_index = _np.repeat(None, n)

    # Loop over all centroids and find the elements within which boundary points lie
    for i in range(n):
        eucl_norm = _np.linalg.norm(
            centroids - _np.atleast_2d(boundary_points[:, i]).transpose(), axis=0
        )

        comp = _np.where(eucl_norm == _np.min(eucl_norm))[0]

        if comp.size != 0:
            element_index[i] = comp[0]

    space = _bempp.function_space(grid, "P", 1)
    grid_function = _bempp.GridFunction(space, coefficients=dirichlet_solution)
    local_coords = _np.zeros((2, n), dtype=float)
    total_boundary_pressure = _np.zeros(n, dtype="complex128")

    # Loop over elements within which near points lie
    for i in range(n):

        # Obtain vertices of element
        vertices_elem = vertices[:, elements[:, element_index[i]]].transpose()

        # Translate element so that first vertex is global origin
        vertices_translated = vertices_elem - vertices_elem[0, :]
        boundary_point_translated = boundary_points[:, i] - vertices_elem[0, :]

        # Compute element normal
        vector_a = vertices_translated[1, :] - vertices_translated[0, :]
        vector_b = vertices_translated[2, :] - vertices_translated[0, :]
        vector_a_cross_vector_b = _np.cross(vector_a, vector_b)
        element_normal = _normalize_vector(vector_a_cross_vector_b)

        # Obtain first rotation matrix for coordinate transformation
        h = _np.sqrt(element_normal[0] ** 2 + element_normal[1] ** 2)
        if h != 0:
            r_z = _np.array(
                [
                    [element_normal[0] / h, element_normal[1] / h, 0],
                    [-element_normal[1] / h, element_normal[0] / h, 0],
                    [0, 0, 1],
                ]
            )
        else:
            r_z = _np.identity(3, dtype=float)

        # Obtain rotated element normal
        element_normal_rotated = _np.matmul(r_z, element_normal)

        # Obtain second rotation matrix for coordinate transformation
        r_y = _np.array(
            [
                [element_normal_rotated[2], 0, -element_normal_rotated[0]],
                [0, 1, 0],
                [element_normal_rotated[0], 0, element_normal_rotated[2]],
            ]
        )

        # Obtain total rotation matrix
        r_y_mult_r_z = _np.matmul(r_y, r_z)
        vertices_0_transformed = _np.matmul(r_y_mult_r_z, vertices_translated[0, :])
        vertices_1_transformed = _np.matmul(r_y_mult_r_z, vertices_translated[1, :])
        vertices_2_transformed = _np.matmul(r_y_mult_r_z, vertices_translated[2, :])
        boundary_point_transformed = _np.matmul(r_y_mult_r_z, boundary_point_translated)

        # Extract vertex coordinates in rotated coordinate system in x-y plane
        x = boundary_point_transformed[0]
        y = boundary_point_transformed[1]
        x0 = vertices_0_transformed[0]
        y0 = vertices_0_transformed[1]
        x1 = vertices_1_transformed[0]
        y1 = vertices_1_transformed[1]
        x2 = vertices_2_transformed[0]
        y2 = vertices_2_transformed[1]

        # Obtain local coordinates in orthonormal system for element
        transformation_matrix = _np.array([[x1 - x0, x2 - x0], [y1 - y0, y2 - y0]])
        transformation_matrix_inv = _np.linalg.inv(transformation_matrix)
        rhs = _np.vstack((x - x0, y - y0))
        local_coords[:, i] = _np.matmul(transformation_matrix_inv, rhs).transpose()

        # Required format for element and local coordinates for GridFunction.evaluate
        elem = list(grid.leaf_view.entity_iterator(0))[element_index[i]]
        coord = _np.array([[local_coords[0, i]], [local_coords[1, i]]])

        # Calculate pressure phase and magnitude at near point
        total_boundary_pressure[i] = grid_function.evaluate(elem, coord)

    return total_boundary_pressure


def ppi_calculator(bounding_box, resolution):
    """To convert resolution to diagonal ppi

    Parameters
    ----------
    bounding_box : list float
        list of min and max of the 2D plane
    resolution : list float
        list of number of points along each direction

    Returns
    -------
    resolution : float
        resolution in ppi

    """
    diagonal_length_meter = _np.sqrt(
        (bounding_box[1] - bounding_box[0]) ** 2
        + (bounding_box[3] - bounding_box[2]) ** 2
    )
    diagonal_length_inches = diagonal_length_meter * 39.37
    diagonal_points = _np.sqrt(resolution[0] ** 2 + resolution[1] ** 2)

    return diagonal_points / diagonal_length_inches


def domain_edge(model, plane_axes, plane_offset):
    """Determine the points at the edges of the domains by computing the intersection of
    the grid triangular elements with planes of constant x, y or z. The intersection
    points are then sorted by proximity to one another.

    Parameters
    ----------
    model : optimus.Model
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
    for subdomain_number in range(model.n_subdomains):
        vertices = model.geometry[subdomain_number].grid.leaf_view.vertices
        elements = model.geometry[subdomain_number].grid.leaf_view.elements

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


def array_to_imshow(field_array):
    """Convert a two-dimensional array to a format for imshow plots.

    Parameters
    ----------
    field_array : numpy.ndarray
        The two-dimensional array with grid values.

    Returns
    -------
    field_imshow : numpy.ndarray
        The two-dimensional array for imshow plots.

    """
    return _np.flipud(field_array.T)
