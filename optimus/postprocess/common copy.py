import numpy as np
import bempp.api as bempp
import time
from .post_process_default_parameters import DefaultPostProcessParameters as DP
import sys


class PostProcess:
    """Sets default values for post_process parameters"""

    def __init__(self):

        self.resolution = DP.RESOLUTION
        self.plane_axes = DP.PLANE_AXES
        self.plane_offset = DP.PLANE_OFFSET
        self.bounding_box = DP.BOUNDING_BOX
        self.points = DP.POINTS
        self.index_exterior = DP.INDEX_EXTERIOR
        self.index_interior = DP.INDEX_INTERIOR
        self.obj_changed = DP.OBJ_CHANGED
        self.repeated_run_flag = DP.REPEATED_RUN_FLAG
        self.verbose = DP.VERBOSE

    def update_changed_flag(self, attr_name, value):
        if hasattr(self, attr_name) and getattr(self, attr_name) != value:
            self.obj_changed = True
        else:
            self.obj_changed = False

    @property
    def resolution(self):
        """Sets the resolution of the visualisation grid. It is a list of two elements:
        [number of points along axis 1, number of points along axis 2]"""
        return self.__resolution

    @resolution.setter
    def resolution(self, value):
        self.update_changed_flag("resolution", value)
        self.__resolution = value

    @property
    def plane_offset(self):
        """Defines the offset of the visualisation plane along axis 3. It is a numeric value with the same unit as bounding box values."""
        return self.__plane_offset

    @plane_offset.setter
    def plane_offset(self, value):
        self.update_changed_flag("plane_offset", value)
        self.__plane_offset = value

    @property
    def plane_axes(self):
        """Defines the visualisation plane by two axes. It is a list of two values from 0,1,2 refering to X,Y,Z axes, respectively."""
        return self.__plane_axes

    @plane_axes.setter
    def plane_axes(self, value):
        self.update_changed_flag("plane_axes", value)
        self.__plane_axes = value

    @property
    def bounding_box(self):
        """Defines the bounding box for visualisation. It is a list of 4 values: [axis1_min, axis1_max, axis2_min, axis2_max]"""
        return self.__bounding_box

    @bounding_box.setter
    def bounding_box(self, value):
        self.update_changed_flag("bounding_box", value)
        self.__bounding_box = value

    @property
    def points(self):
        """Defines the points (3D grid) for visualisation."""
        return self.__points

    @points.setter
    def points(self, value):
        self.__points = value

    def chunker(self, seq, size):
        """
        Imported from https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
        """
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    # def solution_to_gridfunction(self, model, gmres_solution):

    #     number_of_scatterers = int(model.operator_blocks.ndims[0] / 2)

    #     solution_gridfunction = []
    #     gmres_solution_index = [None, None, 0]
    #     domain_spaces_index = [0, 1]

    #     # if "permut" in model.meta_data:
    #     #     permute = True
    #     # else:
    #     #     permute = False

    #     # if permute:
    #     #     domain_spaces_index.reverse()

    #     # Calculate the fields from the surface potential with the potential operators
    #     for spaces in self.chunker(model.operator_blocks.domain_spaces, 2):
    #         gmres_solution_index[0] = gmres_solution_index[2]
    #         gmres_solution_index[1] = (
    #             gmres_solution_index[0]
    #             + spaces[domain_spaces_index[0]].global_dof_count
    #         )
    #         gmres_solution_index[2] = (
    #             gmres_solution_index[1]
    #             + spaces[domain_spaces_index[1]].global_dof_count
    #         )

    #         scatterer_gridfunction = [
    #             bempp.GridFunction(
    #                 spaces[domain_spaces_index[0]],
    #                 coefficients=gmres_solution[
    #                     gmres_solution_index[0] : gmres_solution_index[1]
    #                 ],
    #             ),
    #             bempp.GridFunction(
    #                 spaces[domain_spaces_index[1]],
    #                 coefficients=gmres_solution[
    #                     gmres_solution_index[1] : gmres_solution_index[2]
    #                 ],
    #             ),
    #         ]

    #         # if permute:
    #         #     scatterer_gridfunction.reverse()

    #         solution_gridfunction += scatterer_gridfunction

    #     return solution_gridfunction

    def msh_from_string(self, geo_string):
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

    def generate_grid_from_geo_string(self, geo_string):
        """Helper routine that implements the grid generation"""
        import os

        msh_name = self.msh_from_string(geo_string)
        grid = bempp.import_grid(msh_name)
        os.remove(msh_name)
        return grid

    def plane_grid(
        self,
        x_axis_lims=DP.PLANE_GRID_X_AXIS_LIMS,
        y_axis_lims=DP.PLANE_GRID_Y_AXIS_LIMS,
        rotation_axis=DP.PLANE_GRID_ROTATION_AXIS,
        rotation_angle=DP.PLANE_GRID_ROTATION_ANGLE,
        h=DP.PLANE_GRID_H,
    ):
        """
        Return a 2D square shaped plane.

        x_axis_lims : list of two float numbers
            The bounding values along the x-axis of plane.
        y_axis_lims : list of two float numbers
            The bounding values along the y-axis of plane.
        rotation_axis : a list of size three populated with 0 or 1,
            It defines the axis of rotation so to construct the desired plane from an x-y plane.
        h : float
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

        if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
            pass
            # geometry = (f"ax1_lim1 = {x_axis_lims[0]};\nax1_lim2 = {x_axis_lims[1]};\n"
            #             + "ax2_lim1 = {y_axis_lims[0]};\nax2_lim2 = {y_axis_lims[1]};\n"
            #             + "rot_ax1 = {rotation_axis[0]};\nrot_ax2 = {rotation_axis[1]};\n"
            #             + "rot_ax3 = {rotation_axis[2]};\nrot_ang_rad = {rotation_angle};\ncl = {h};\n"
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
                + str(h)
                + ";\n"
                + stub
            )
        return self.generate_grid_from_geo_string(geometry)

    def create_grid(self, model, mode=DP.CREATE_GRID_MODE, user_grid=False):

        from .exterior_interior_points_eval import exterior_interior_points_eval

        domain_grids = [
            model.geometry[n_sub].grid for n_sub in range(model.n_subdomains)
        ]

        if not self.bounding_box:
            ax1_min, ax1_max, ax2_min, ax2_max = 0, 0, 0, 0
            for grid in domain_grids:
                ax1_min = np.min(
                    [ax1_min, grid.bounding_box[0][self.plane_axes[0]].min()]
                )
                ax1_max = np.max(
                    [ax1_max, grid.bounding_box[1][self.plane_axes[0]].max()]
                )
                ax2_min = np.min(
                    [ax2_min, grid.bounding_box[0][self.plane_axes[1]].min()]
                )
                ax2_max = np.max(
                    [ax2_max, grid.bounding_box[1][self.plane_axes[1]].max()]
                )

            self.bounding_box = [5.0 * x for x in [ax1_min, ax1_max, ax2_min, ax2_max]]

        ax1_min, ax1_max, ax2_min, ax2_max = self.bounding_box

        # if (not user_grid and self.obj_changed) or (not user_grid and not self.points.any()):

        if not user_grid and (self.obj_changed or not self.points.any()):
            self.obj_changed = False

            if mode == "2D":
                plot_grid = np.mgrid[
                    ax1_min : ax1_max : self.resolution[0] * 1j,
                    ax2_min : ax2_max : self.resolution[1] * 1j,
                ]

                points_tmp = [np.ones(plot_grid[0].size) * self.plane_offset] * 3
                points_tmp[self.plane_axes[0]] = plot_grid[0].ravel()
                points_tmp[self.plane_axes[1]] = plot_grid[1].ravel()

                points = np.vstack((points_tmp,))

            elif mode == "3D":
                if 2 not in self.plane_axes:
                    axis1_lims = self.bounding_box[0:2]
                    axis2_lims = self.bounding_box[2:]
                    rotation_axis = [0, 0, 1]
                    rotation_angle = "2*Pi"
                elif 1 not in self.plane_axes:
                    axis1_lims = self.bounding_box[0:2]
                    axis2_lims = self.bounding_box[2:]
                    rotation_axis = [1, 0, 0]
                    rotation_angle = "Pi/2"
                elif 0 not in self.plane_axes:
                    axis1_lims = self.bounding_box[2:]
                    axis2_lims = self.bounding_box[0:2]
                    rotation_axis = [0, 1, 0]
                    rotation_angle = "-Pi/2"

                elem_len = np.min(
                    [
                        (axis1_lims[1] - axis1_lims[0]) / self.resolution[0],
                        (axis2_lims[1] - axis2_lims[0]) / self.resolution[1],
                    ]
                )

                plane_grid = self.plane_grid(
                    axis1_lims, axis2_lims, rotation_axis, rotation_angle, elem_len
                )
                points = plane_grid.leaf_view.vertices

            points_interior = []
            idx_interior = []
            idx_exterior = np.full((points.shape)[1], True, dtype=bool)

            if self.verbose:
                TS_INT_EXT = time.time()
                print(
                    "\n Identifying the exterior and interior points Started at: ",
                    time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
                )

            for grid in domain_grids:
                (
                    POINTS_INTERIOR_TMP,
                    POINTS_EXTERIOR_TMP,
                    IDX_INTERIOR_TMP,
                    IDX_EXTERIOR_TMP,
                ) = exterior_interior_points_eval(grid, points)
                points_interior.append(POINTS_INTERIOR_TMP[0])
                idx_interior.append(IDX_INTERIOR_TMP[0])
                idx_exterior[IDX_EXTERIOR_TMP == False] = False

                # for x, y, z, R0 in zip(
                #     system_setup.x_center,
                #     system_setup.y_center,
                #     system_setup.z_center,
                #     system_setup.R0_list,
                # ):

                #     xyz_center = np.tile(np.vstack((x, y, z)), (1, points.shape[1]))
                #     idx_exterior_tmp = np.linalg.norm(points - xyz_center) > R0
                #     idx_exterior[(idx_exterior_tmp == False)] = False
                #     idx_interior.append(np.linalg.norm(points - xyz_center) <= R0)
                #     points_interior.append(points[:, idx_interior[-1]])

            if self.verbose:
                TE_INT_EXT = time.time()
                print(
                    "\n Identifying the exterior and interior points Finished... Duration in secs: ",
                    TE_INT_EXT - TS_INT_EXT,
                )

            points_exterior = points[:, idx_exterior]
            self.points_interior = points_interior
            self.points_exterior = points_exterior
            self.points = points
            self.index_exterior = idx_exterior
            self.index_interior = idx_interior

        elif user_grid:
            self.obj_changed = False
            print("\n Using the visualisation grid provided by the user")

            assert isinstance(self.points, np.ndarray), "points must be a Numpy array."
            assert isinstance(
                self.index_interior, list
            ), "index_interior must be a list."
            assert isinstance(
                self.index_exterior, list
            ), "index_exterior must be a list."

            idx_exterior = self.index_exterior[0]
            idx_interior = self.index_interior
            points_exterior = self.points[:, idx_exterior]
            points_interior = []
            for id in idx_interior:
                points_interior.append(self.points[:, id])

            self.points_interior = points_interior
            self.points_exterior = points_exterior

        return (
            self.points,
            self.points_interior,
            self.points_exterior,
            self.index_interior,
            self.index_exterior,
        )

    def visualiser(
        self,
        model,
        h_mat="dense",
        mode="2D",
        user_grid=False,
    ):
        if h_mat == "dense":
            bempp.global_parameters.assembly.potential_operator_assembly_type = "dense"
        else:
            bempp.global_parameters.hmat.eps = DP.BEMPP_HMAT_EPS
            bempp.global_parameters.hmat.max_rank = DP.BEMPP_HMAT_MAX_RANK
            bempp.global_parameters.hmat.max_block_size = DP.BEMPP_HMAT_MAX_BLOC_SIZE

        (
            points,
            points_interior,
            points_exterior,
            idx_interior,
            idx_exterior,
        ) = self.create_grid(model, mode=mode, user_grid=user_grid)

        if np.array_equal(self.repeated_run_flag, points_exterior):
            return

        self.repeated_run_flag = points_exterior

        if self.verbose:
            TS_INC_FIELD = time.time()
            print(
                "\n Calculating the incident field Started at: ",
                time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            )

        incident_exterior = model.source.pressure_field(
            model.material_exterior, points_exterior
        )

        if self.verbose:
            TE_INC_FIELD = time.time()
            print(
                "\n Calculating the incident field Finished... Duration in secs: ",
                TE_INC_FIELD - TS_INC_FIELD,
            )

        total_field = np.empty((1, points.shape[1]), dtype="complex128").ravel()
        exterior_values = np.zeros((1, points_exterior.shape[1]), dtype="complex128")

        if self.verbose:
            TS_POT_OPS_FIELD = time.time()
            print(
                "\n Calculating the interior and exterior potential operators Started at: ",
                time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            )
        i = 0
        for (
            solution_pair,
            space,
            interior_point,
            interior_id,
            interior_material,
        ) in zip(
            self.chunker(model.solution, 2),
            model.space,
            points_interior,
            idx_interior,
            model.material_interior,
        ):
            if self.verbose:
                print("Calculating the fields of Domain {0}".format(i + 1))
                print(
                    interior_point.shape,
                    interior_id.shape,
                    interior_material.compute_wavenumber(model.source.frequency),
                    interior_material,
                )

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

            rho_ratio = interior_material.density / model.material_exterior.density

            exterior_values += (
                -pot_ext_sl * solution_pair[1] + pot_ext_dl * solution_pair[0]
            )
            interior_value = (
                pot_int_sl * solution_pair[1] * rho_ratio
                - pot_int_dl * solution_pair[0]
            )
            total_field[interior_id] = interior_value.ravel()

            i += 1

        if self.verbose:
            TE_POT_OPS_FIELD = time.time()
            print(
                "\n Calculating the interior and exterior potential operators Finished... Duration in secs: ",
                TE_POT_OPS_FIELD - TS_POT_OPS_FIELD,
            )

        # First compute the scattered field
        scattered_field = np.empty((1, points.shape[1]), dtype="complex128").ravel()
        scattered_field[:] = np.nan
        scattered_field[idx_exterior] = exterior_values.ravel()

        incident_exterior_field = np.empty(
            (1, points.shape[1]), dtype="complex128"
        ).ravel()
        incident_exterior_field[:] = np.nan
        incident_exterior_field[idx_exterior] = incident_exterior.ravel()

        total_field[idx_exterior] = (
            scattered_field[idx_exterior] + incident_exterior_field[idx_exterior]
        )

        if mode == "2D":
            self.scattered_field = scattered_field.reshape(self.resolution, order="F")
            self.total_field = total_field.reshape(self.resolution, order="F")
            self.incident_field = incident_exterior_field.reshape(
                self.resolution, order="F"
            )
            self.l2_norm_total_field_mpa = np.linalg.norm(total_field)

        # domain_grids = [
        #     model.geometry[n_sub].grid for n_sub in range(model.n_subdomains)
        # ]

        elif mode == "3D":
            domain_grids.append(plane_grid)
            grids_union_all = bempp.shapes.union(domain_grids)
            space_union_all = bempp.function_space(grids_union_all, "P", 1)
            domain_solutions_all = [
                model.solution[2 * i].coefficients for i in range(model.n_subdomains)
            ]
            domain_solutions_all.append(total_field)
            plot3D_ptot_all = bempp.GridFunction(
                space_union_all,
                coefficients=np.concatenate(
                    [domain_solutions_all[i] for i in range(model.n_subdomains + 1)]
                ),
            )
            plot3D_ptot_abs_all = bempp.GridFunction(
                space_union_all,
                coefficients=np.concatenate(
                    [
                        np.abs(domain_solutions_all[i])
                        for i in range(model.n_subdomains + 1)
                    ]
                ),
            )

            bempp.export(file_name=DP.PTOT_ALL_FILENAME, grid_function=plot3D_ptot_all)
            bempp.export(
                file_name=DP.PTOT_ABS_ALL_FILENAME, grid_function=plot3D_ptot_abs_all
            )

            self.total_field3D_gridfunc = plot3D_ptot_all
            self.grids3D = grids_union_all
            self.plane_grid = plane_grid
            self.total_field = total_field
            self.scattered_field = scattered_field
            self.incident_field = incident_exterior_field

    def ppi_calculator(self):
        diagonal_length_meter = np.sqrt(
            (self.bounding_box[1] - self.bounding_box[0]) ** 2
            + (self.bounding_box[3] - self.bounding_box[2]) ** 2
        )
        diagonal_length_inches = diagonal_length_meter * 39.37
        diagonal_points = np.sqrt(self.resolution[0] ** 2 + self.resolution[1] ** 2)
        return diagonal_points / diagonal_length_inches

    def print_parameters(self):
        print("\n", 70 * "*")
        print("\n resolution in number of points: ", self.resolution)
        print("\n resolution in ppi (diagonal ppi): %d " % self.ppi_calculator())
        print("\n 2D plane axes: ", self.plane_axes)
        print(
            "\n bounding box (2D frame) points: ",
            [self.bounding_box[0], self.bounding_box[2]],
            [self.bounding_box[1], self.bounding_box[3]],
        )
        print("\n the offset of 2D plane along the 3rd axis: ", self.plane_offset)
        print("\n", 70 * "*")

        return {
            "resolution_in_points": self.resolution,
            "resolution_in__diagonal_ppi": self.ppi_calculator(),
            "2D_plane_axes": self.plane_axes,
            "bounding_box": self.bounding_box,
            "plane_offset": self.plane_offset,
        }

    def domain_edges(self, alpha=0.005, only_outer=True):
        """This function determines the points on the edges of thedomains using the Concave Hull method.
        alpha: the threshhold value.
        only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
        """

        from .concave_hull import concave_hull as _concave_hull

        self.edge_points = []
        for k in range(self.number_of_scatterers):
            points_int = self.points_interior[k][self.plane_axes, :]
            edges = _concave_hull(points_int.T, alpha, only_outer)
            for i, j in edges:
                self.edge_points.append(
                    np.vstack([points_int[0, [i, j]], points_int[1, [i, j]]])
                )

        return self.edge_points
