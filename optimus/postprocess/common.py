import numpy as np
import bempp.api as bempp
import time


class PostProcess:
    """
    Create an optimus postprocess object with the specified parameters.

    Input argument
    ----------
        model: optimus model object
            this model object must include the solution vectors (the solve() function must be executed).
        verbose: boolean
            to display the logs or not.
    """

    def __init__(self, model, verbose=False):
        self.verbose = verbose
        self.model = model
        self.domains_grids = [
            model.geometry[n_sub].grid for n_sub in range(model.n_subdomains)
        ]

    def create_computational_grid(self, **kwargs):
        """
        Calculate the pressure field in the specified locations.
        Needs to be overridden by specific source type.

        Input argument
        ---------
            **kwargs: to be specified for different types of postprocessing.
        """
        raise NotImplementedError

    def compute_fields(self):
        """
        Calculate the pressure field in the specified locations.
        Needs to be overridden by specific source type.
        """
        raise NotImplementedError

    def print_parameters(self):
        print("\n", 70 * "*")
        if hasattr(self, "resolution") and hasattr(self, "bounding_box"):
            print("\n resolution in number of points: ", self.resolution)
            print(
                "\n resolution in ppi (diagonal ppi): %d "
                % ppi_calculator(self.bounding_box, self.resolution)
            )
        if hasattr(self, "plane_axes"):
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

    Input argument
    ---------
        domains_grids: a list of optimus grids
            the bounding box of the union of grids is determined.
        plane_axes : a list/tuple of two int numbers
            The boundary plane. possible values are 0,1,2 for x,y,z axes, respectively
    """
    if not isinstance(domains_grids, list):
        domains_grids = list(domains_grids)

    ax1_min, ax1_max, ax2_min, ax2_max = 0, 0, 0, 0
    for grid in domains_grids:
        ax1_min = np.min([ax1_min, grid.bounding_box[0][plane_axes[0]].min()])
        ax1_max = np.max([ax1_max, grid.bounding_box[1][plane_axes[0]].max()])
        ax2_min = np.min([ax2_min, grid.bounding_box[0][plane_axes[1]].min()])
        ax2_max = np.max([ax2_max, grid.bounding_box[1][plane_axes[1]].max()])
    bounding_box = [ax1_min, ax1_max, ax2_min, ax2_max]
    return bounding_box


def find_int_ext_points(domain_grids, points, verbose):
    """
    Identify the interior and exterior points w.r.t each grid.

    Input argument
    ---------
        domains_grids: a list of optimus grids
            the bounding box of the union of grids is determined.
        points : a numpy array of size 3xN
            The field points.
    """
    from .exterior_interior_points_eval import exterior_interior_points_eval

    points_interior = []
    idx_interior = []
    idx_exterior = np.full((points.shape)[1], True, dtype=bool)
    if verbose:
        TS_INT_EXT = time.time()
        print(
            "\n Identifying the exterior and interior points Started at: ",
            time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
        )
    for grid in domain_grids:
        (
            POINTS_INTERIOR_TMP,
            POINTS_EXTERIOR_TMP,
            POINTS_BOUNDARY_TMP,
            IDX_INTERIOR_TMP,
            IDX_EXTERIOR_TMP,
            IDX_BOUNDARY_TMP,
        ) = exterior_interior_points_eval(grid=grid, xyz_field=points, verbose=verbose)
        points_interior.append(POINTS_INTERIOR_TMP[0])
        idx_interior.append(IDX_INTERIOR_TMP[0])
        idx_exterior[IDX_EXTERIOR_TMP == False] = False

    if verbose:
        TE_INT_EXT = time.time()
        print(
            "\n Identifying the exterior and interior points Finished... Duration in secs: ",
            TE_INT_EXT - TS_INT_EXT,
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
    from ..utils.generic import chunker, bold_ul_text
    from optimus import global_parameters

    if global_parameters.postprocessing.assembly_type.lower() in [
        "h-matrix",
        "hmat",
        "h-mat",
        "h_mat",
        "h_matrix",
    ]:
        bempp.global_parameters.hmat.eps = global_parameters.postprocessing.hmat_eps
        bempp.global_parameters.hmat.max_rank = (
            global_parameters.postprocessing.hmat_max_rank
        )
        bempp.global_parameters.hmat.max_block_size = (
            global_parameters.postprocessing.hmat_max_block_size
        )
    elif global_parameters.postprocessing.assembly_type.lower() == "dense":
        bempp.global_parameters.assembly.potential_operator_assembly_type = "dense"
    else:
        raise ValueError(
            "Supported operator assembly methods are "
            + bold_ul_text("dense")
            + " and "
            + bold_ul_text("h-matrix.")
        )

    TS_POT_OPS_FIELD = time.time()
    if verbose:
        print(
            "\n Calculating the interior and exterior potential operators Started at: ",
            time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
        )

    total_field = np.empty((1, points.shape[1]), dtype="complex128").ravel()
    scattered_field = np.empty((1, points.shape[1]), dtype="complex128").ravel()
    scattered_field[:] = np.nan
    incident_exterior_field = np.empty((1, points.shape[1]), dtype="complex128").ravel()
    incident_exterior_field[:] = np.nan

    if index_exterior.any():
        exterior_values = np.zeros((1, points_exterior.shape[1]), dtype="complex128")
        ext_calc_flag = True
    else:
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

        if (interior_idx).any():
            pot_int_sl = bempp.operators.potential.helmholtz.single_layer(
                space,
                interior_point,
                interior_material.compute_wavenumber(model.source.frequency),
            )
            pot_int_dl = bempp.operators.potential.helmholtz.double_layer(
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
            pot_ext_sl = bempp.operators.potential.helmholtz.single_layer(
                space,
                points_exterior,
                model.material_exterior.compute_wavenumber(model.source.frequency),
            )
            pot_ext_dl = bempp.operators.potential.helmholtz.double_layer(
                space,
                points_exterior,
                model.material_exterior.compute_wavenumber(model.source.frequency),
            )
            exterior_values += (
                -pot_ext_sl * solution_pair[1] + pot_ext_dl * solution_pair[0]
            )

        i += 1

        if verbose:
            TE_POT_OPS_FIELD = time.time()
            print(
                "\n Calculating the interior and exterior potential operators Finished... Duration in secs: ",
                TE_POT_OPS_FIELD - TS_POT_OPS_FIELD,
            )

    if ext_calc_flag:
        TS_INC_FIELD = time.time()
        if verbose:
            print(
                "\n Calculating the incident field Started at: ",
                time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            )

        incident_exterior = model.source.pressure_field(
            model.material_exterior, points_exterior
        )

        TE_INC_FIELD = time.time()
        if verbose:
            print(
                "\n Calculating the incident field Finished... Duration in secs: ",
                TE_INC_FIELD - TS_INC_FIELD,
            )
        incident_exterior_field[index_exterior] = incident_exterior.ravel()
        scattered_field[index_exterior] = exterior_values.ravel()
        total_field[index_exterior] = (
            scattered_field[index_exterior] + incident_exterior_field[index_exterior]
        )

    return total_field, scattered_field, incident_exterior_field


def ppi_calculator(bounding_box, resolution):
    diagonal_length_meter = np.sqrt(
        (bounding_box[1] - bounding_box[0]) ** 2
        + (bounding_box[3] - bounding_box[2]) ** 2
    )
    diagonal_length_inches = diagonal_length_meter * 39.37
    diagonal_points = np.sqrt(resolution[0] ** 2 + resolution[1] ** 2)
    return diagonal_points / diagonal_length_inches


def domain_edge(points_interior, plane_axes, alpha=0.001, only_outer=True):
    """This function determines the points on the edges of the domains using the Concave Hull method.
    alpha: the threshhold value.
    only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    """

    from .concave_hull import concave_hull as _concave_hull

    domains_edge_points = []
    for k in range(len(points_interior)):
        if points_interior[k].any():
            points_int_planar = points_interior[k][plane_axes, :]
            edges = _concave_hull(points_int_planar.T, alpha, only_outer)
            for i, j in edges:
                domains_edge_points.append(
                    np.vstack(
                        [points_int_planar[0, [i, j]], points_int_planar[1, [i, j]]]
                    )
                )
    return domains_edge_points
