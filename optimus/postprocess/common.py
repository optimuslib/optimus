"""Common functionality for postprocessing and visualising results."""

import numpy as _np
import bempp.api as _bempp
import time as _time


class PostProcess:
    def __init__(self, model, verbose=False):
        """
        Create an optimus postprocess object with the specified parameters.

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
        """
        Create the grid on which to calculate the pressure field.
        Needs to be overridden by specific source type.

        Parameters
        ---------
        kwargs : dict
            Options to be specified for different types of postprocessing.
        """
        raise NotImplementedError

    def compute_fields(self):
        """
        Calculate the pressure field in the specified locations.
        Needs to be overridden by specific source type.
        """
        raise NotImplementedError

    def print_parameters(self):
        """
        Display parameters used for visualisation.
        """
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
    """
    Calculate the bounding box for a set of grids.

    Parameters
    ---------
    domains_grids : list[optimus.Grid]
        The grids of the subdomains.
    plane_axes : tuple[int]
        The two indices of the boundary plane.
        Possible values are 0,1,2 for x,y,z axes, respectively.

    Returns
    ----------
    bounding_box: list[float]
        Bounding box specifying the visualisation section on
        the plane_axis: [axis1_min, axis1_max, axis2_min, axis2_max]
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
    ---------
    domains_grids : list[optimus.Grid]
        The grids of the subdomains.
    points : numpy.ndarray
        The field points.
        The size of the array should be (3,N).
    verbose : boolean
        Display the logs.

    Returns
    -----------
    points_interior : list[np.ndarray]
        A list of numpy arrays of size (3,N), where
        element i of the list is an array of coordinates of the
        interior points for domain i, i=1,...,no_subdomains.
    points_exterior : numpy.ndarray
        An array of size (3,N) with visualisation points in the exterior domain
    index_interior : list[numpy.ndarray]
        A list of arrays of size (1,N) with boolean values,
        identifying the interior points for each domain.
    index_exterior : numpy.ndarray
        An array of size (1,N) with boolean values,
        identifying the exterior points.
    """

    from .exterior_interior_points_eval import exterior_interior_points_eval
    from optimus import global_parameters

    points_interior = []
    idx_interior = []
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
        idx_interior,
        idx_exterior,
    )


def compute_pressure_fields(
    model,
    points,
    points_exterior,
    index_exterior,
    points_interior,
    index_interior,
    verbose,
):
    """
    Calculate the scattered and total pressure fields for visualisation.

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
    points_interior : list[numpy.ndarray]
        A list of arrays of size (3,N), where
        element i of the list is an array of coordinates of the
        interior points for domain i, i=1,...,no_subdomains
    index_interior : list[numpy.ndarray]
        A list of arrays of size (1,N) with boolean values indentifying
        the interior points.
    verbose : bool
        Display the logs.

    Returns
    -----------
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

    return total_field, scattered_field, incident_exterior_field


def ppi_calculator(bounding_box, resolution):
    """
    To convert resolution to diagonal ppi

    Parameters
    -----------
    bounding_box : list[float]
        list of min and max of the 2D plane
    resolution : list[float]
        list of number of points along each direction

    Returns
    -----------
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


def domain_edge(points_interior, plane_axes, alpha=0.001, only_outer=True):
    """
    Determine the points on the edges of the domains using the Concave Hull method.

    Parameters
    ----------
    points_interior : list[numpy.ndarray]
        List of arrays of size (3,N) with the interior points for each domain.
    plane_axes : list[int]
        The axes of the plane.
    alpha : float
        The threshold parameter in the Concave Hell method.
    only_outer : boolean
        Specify if we keep only the points on the outer border or also inner edges.

    Returns
    -----------
    domains_edge_points : list[numpy.ndarray]
        list of numpy arrays of coordinates of points on the edges
    """

    from .concave_hull import concave_hull as _concave_hull

    domains_edge_points = []
    for k in range(len(points_interior)):
        if points_interior[k].any():
            points_int_planar = points_interior[k][plane_axes, :]
            edges = _concave_hull(points_int_planar.T, alpha, only_outer)
            for i, j in edges:
                domains_edge_points.append(
                    _np.vstack(
                        [points_int_planar[0, [i, j]], points_int_planar[1, [i, j]]]
                    )
                )
    return domains_edge_points


def array_to_imshow(field_array):
    """
    Convert a two-dimensional array to a format for imshow plots.

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
