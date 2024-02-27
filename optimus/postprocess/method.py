"""Postprocess methods."""
from .common import PostProcess as _PostProcess
import numpy as _np


class VisualisePlane(_PostProcess):
    def __init__(self, model, verbose=False):
        """
        Create a PostProcess object to visualise to field on a 2D planar grid.

        Parameters
        ----------
        model : optimus.model.common.Model
            An optimus model object that includes the solution fields on the boundaries.
        verbose : boolean
            Display the log information.
        """

        super().__init__(model, verbose)

        # Parameters of the 2D planar visualisation grid
        self.resolution = None
        self.plane_axes = None
        self.plane_offset = None
        self.bounding_box = None

        # Points of the 2D planar visualisation grid
        self.points = None
        self.plane = None
        self.domains_edges = None

        # Fields in the 2D planar visualisation grid
        self.total_field_imshow = None
        self.scattered_field_imshow = None
        self.incident_field_imshow = None
        self.l2_norm_total_field_mpa = None

        return

    def create_computational_grid(
        self,
        resolution=(141, 141),
        plane_axes=(0, 1),
        plane_offset=0.0,
        bounding_box=None,
    ):
        """
        Create a planar grid for visualisation of the pressure fields.

        Parameters
        ----------
        resolution : list[int], tuple[int]
            Number of points along the two axes.
        plane_axes : list[int], tuple[int]
            The indices of the axes for the visualisation plane.
            Possible values are 0,1,2 denoting the x,y,z axes, respectively.
            Default: (0, 1).
        plane_offset : float
            Offset of the visualisation plane defined along the third axis.
            Default: 0.
        bounding_box : None, list[float], tuple[float]
            Bounding box specifying the visualisation section along
            the plane's axes: [axis1_min, axis1_max, axis2_min, axis2_max]
            If not specified, the bounding box is calculated from the geometry.
        """
        from .topology import calculate_bounding_box, domain_edge
        from ..utils.mesh import create_grid_points

        self.resolution = resolution
        self.plane_axes = plane_axes
        self.plane_offset = plane_offset

        if bounding_box is not None:
            self.bounding_box = bounding_box
        else:
            self.bounding_box = calculate_bounding_box(self.domains_grids, plane_axes)

        self.points, self.plane = create_grid_points(
            self.resolution,
            self.plane_axes,
            self.plane_offset,
            self.bounding_box,
            mode="numpy",
        )

        self.field.segmentation(self.points)

        self.domains_edges = domain_edge(
            self.domains_grids,
            self.plane_axes,
            self.plane_offset,
        )

        return

    def compute_fields(self):
        """Calculate the scattered and total pressure fields in the planar grid."""

        from .common import array_to_imshow

        self.field.compute_fields()

        self.l2_norm_total_field_mpa = _np.linalg.norm(self.field.total_field)

        self.scattered_field_imshow = array_to_imshow(
            self.field.scattered_field.reshape(self.resolution)
        )
        self.total_field_imshow = array_to_imshow(
            self.field.total_field.reshape(self.resolution)
        )
        self.incident_field_imshow = array_to_imshow(
            self.field.incident_field.reshape(self.resolution)
        )

        return


class VisualiseCloudPoints(_PostProcess):
    def __init__(self, model, verbose=False):
        """Create a PostProcess optimus object where the visualisation grid is
        a cloud of 3D points.

        Parameters
        ----------
        model : optimus.Model
            An optimus model object that includes the solution fields on the boundaries.
        verbose : boolean
            Display the log information.

        """
        super().__init__(model, verbose)

        self.points = None
        self.l2_norm_total_field_mpa = None
        return

    def create_computational_grid(self, points):
        """Create a point cloud to compute the pressure fields.

        Parameters
        ----------
        points: numpy.ndarray
            Array of size (3,N) with points on which to calculate the pressure field.

        """

        from ..utils.conversions import convert_to_3n_array

        self.points = convert_to_3n_array(points, "visualisation points")

        self.field.segmentation(self.points)

    def compute_fields(self):
        """Calculate the scattered and total pressure fields in the postprocess grid."""

        self.field.compute_fields()

        self.l2_norm_total_field_mpa = _np.linalg.norm(self.field.total_field)
        return

    def display_field(self, size=0.2):
        """Display the magnitude of the field in the cloud points.

        Parameters
        ___________
        size : float
            The point size, 0.2 by default.
        """
        import k3d

        plot = k3d.plot()
        plot += k3d.factory.points(
            self.points, attribute=abs(self.field.total_field), point_size=size
        )

        plot.display()
        return


class VisualisePlaneAndBoundary(_PostProcess):
    def __init__(self, model, verbose=False):
        """
        Create a PostProcess object to visualise to field in a union of
        a plane and surface meshes of the domains with Gmsh.

        Parameters
        ----------
        model : optimus.model.common.ExteriorModel
            An optimus model object that includes the solution fields on the boundaries.
        verbose : boolean
            Display the log information.
        """

        super().__init__(model, verbose)

        # Parameters of the 2D planar visualisation grid
        self.resolution = None
        self.plane_axes = None
        self.plane_offset = None
        self.bounding_box = None

        # Points of the 2D planar visualisation grid
        self.points = None
        self.plane = None
        self.points_interior = None
        self.points_exterior = None
        self.points_boundary = None
        self.index_interior = None
        self.index_exterior = None
        self.index_boundary = None
        self.domains_edges = None

        # Fields in the 2D planar visualisation grid
        self.total_field = None
        self.scattered_field = None
        self.incident_field = None
        self.l2_norm_total_field_mpa = None

        return

    def create_computational_grid(
        self,
        resolution=(141, 141),
        plane_axes=(0, 1),
        plane_offset=0.0,
        bounding_box=None,
    ):
        """Create a planar grid to compute the pressure fields.

        Parameters
        ----------
        resolution : list[int], tuple[int]
            Number of points along the two axes.
        plane_axes : list[int], tuple[int]
            The indices of the axes for the visualisation plane.
            Possible values are 0,1,2 denoting the x,y,z axes, respectively.
            Default: (0, 1).
        plane_offset : float
            Offset of the visualisation plane defined along the third axis.
            Default: 0.
        bounding_box : list[float], tuple[float]
            Bounding box specifying the visualisation section along
            the plane's axes: [axis1_min, axis1_max, axis2_min, axis2_max]
        """

        from .topology import calculate_bounding_box
        from ..utils.mesh import create_grid_points

        self.resolution = resolution
        self.plane_axes = plane_axes
        self.plane_offset = plane_offset

        if bounding_box:
            self.bounding_box = bounding_box
        else:
            self.bounding_box = calculate_bounding_box(self.domains_grids, plane_axes)

        self.points, self.plane = create_grid_points(
            self.resolution,
            self.plane_axes,
            self.plane_offset,
            self.bounding_box,
            mode="gmsh",
        )

        self.field.segmentation(self.points)

        return

    def compute_fields(self, file_name="planar_and_surface"):
        """
        Calculate the scattered and total pressure fields in the planar grid created.
        Export the field values to gmsh files.

        Parameters
        ----------
        file_name : str
            The name for the output file. The results are saved as GMSH files.
            GMSH should be used for visualisation.
        """

        import bempp.api as _bempp

        if self.model.solution is None:
            raise ValueError(
                "The model does not contain a solution, "
                "so no fields can be computed."
            )

        self.field.compute_fields()

        self.l2_norm_total_field_mpa = _np.linalg.norm(self.total_field)

        self.domains_grids.append(self.plane)
        grids_union_all = _bempp.shapes.union(self.domains_grids)
        space_union_all = _bempp.function_space(grids_union_all, "P", 1)
        domain_solutions_all = [
            self.model.solution[2 * i].coefficients
            for i in range(self.model.n_subdomains)
        ]
        domain_solutions_all.append(self.total_field)
        plot3d_ptot_all = _bempp.GridFunction(
            space_union_all,
            coefficients=_np.concatenate(
                [domain_solutions_all[i] for i in range(self.model.n_subdomains + 1)]
            ),
        )
        plot3d_ptot_abs_all = _bempp.GridFunction(
            space_union_all,
            coefficients=_np.concatenate(
                [
                    _np.abs(domain_solutions_all[i])
                    for i in range(self.model.n_subdomains + 1)
                ]
            ),
        )
        # noinspection PyArgumentList
        _bempp.export(
            file_name=file_name + "_ptot_complex.msh",
            grid_function=plot3d_ptot_all,
        )
        # noinspection PyArgumentList
        _bempp.export(
            file_name=file_name + "_ptot_abs.msh",
            grid_function=plot3d_ptot_abs_all,
        )

        return


class VisualiseTimeDomain(_PostProcess):
    def __init__(self, model, verbose=False):
        """
        Create a interactive time domain visualization of the pressure field.

        Parameters
        ----------
        model : optimus.Model
            An optimus model object that includes the solution fields on the boundaries.
        verbose : bool
            Display the logs.
        """

        super().__init__(model, verbose)

    def create_computational_grid(self, time_length=1, n_samples=100):
        """
        Create uniform time points to visualize the field.

        Parameters
        ----------
        time_length : float
            The length of the time interval, in periods.
        n_samples : int
            The number of samples within the time interval.
        """

        self.time_samples = _np.linspace(
            0, 2 * _np.pi * time_length, n_samples, endpoint=False
        )

    def compute_fields(self, postprocess_plane):
        """Calculate the time-dependent field from the harmonic solution.

        Parameters
         ----------
        postprocess_plane : optimus.postprocess.VisualisePlane
            An optimus optimus postprocess object that includes the total,
            scattered and incident field.
        """
        self.postprocess_plane = postprocess_plane
        harmonic_field = postprocess_plane.total_field_imshow[:, :, _np.newaxis]
        time_interval = self.time_samples[_np.newaxis, _np.newaxis, :]
        self.spacetime_field = _np.real(harmonic_field * _np.exp(-1j * time_interval))

    def display_field(self):
        """Display a time visualisation of the harmonic field."""

        from matplotlib import pylab as plt
        from matplotlib import animation
        from IPython.display import HTML
        from IPython.display import display

        plot_start = 0
        plot_end = _np.size(self.time_samples)
        plot_step = 1

        ims = []

        fig, ax = plt.subplots()

        ax.set_title("Temporal animation")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        for t in range(plot_start, plot_end, plot_step):
            field_plot = self.spacetime_field[:, :, t]
            im = plt.imshow(
                field_plot,
                extent=_np.array(self.postprocess_plane.bounding_box),
                clim=(
                    -_np.max(abs(self.postprocess_plane.total_field_imshow)),
                    _np.max(abs(self.postprocess_plane.total_field_imshow)),
                ),
                cmap="seismic",
            )
            ims.append([im])

        plt.close()

        ani = animation.ArtistAnimation(fig, ims)
        plt.show()
        html_anim = ani.to_jshtml()
        display(HTML(html_anim))


class Visualise3DField(_PostProcess):
    def __init__(self, model, verbose=False):
        """
        Create a three-dimensional interactive visualisation of the mesh.

        Parameters
        ----------
        model : optimus.Model
            An optimus model object that includes the solution fields on the boundaries.
        """

        super().__init__(model, verbose)

        self.planes = []
        self.k3d_planes = []

    def create_computational_grid(self):
        """Preprocess the grids in the model."""
        import meshio
        import k3d
        from k3d.colormaps import matplotlib_color_maps
        import os

        self.model.geometry[0].export_mesh("dummy.msh")
        # For multiple domains, we use the first geometry

        mesh = meshio.read("dummy.msh")
        os.remove("dummy.msh")

        indices = mesh.cells["triangle"]
        vertices = mesh.points
        self.mesh3d = k3d.mesh(
            vertices, indices, wireframe=False, color_map=matplotlib_color_maps.viridis
        )

    def add_VisualisePlane(self, postprocess_plane):
        """Add a plane with pressure field to the 3D visualisation.

        Parameters
        ----------
        postprocess_plane : optimus.postprocess.VisualisePlane
            An optimus optimus postprocess object that includes the total,
            scattered and incident field
        """
        import k3d
        from k3d.colormaps import matplotlib_color_maps

        self.planes.append(postprocess_plane)
        div_y, div_z = postprocess_plane.resolution
        y_0, y_1, z_0, z_1 = postprocess_plane.bounding_box

        increment_y = abs(y_1 - y_0) / (div_y - 1)
        increment_z = abs(z_1 - z_0) / (div_z - 1)

        vertices = _np.zeros((div_y * div_z, 3))

        index = 0
        for i in range(div_y):
            angle_y = y_0 + i * increment_y
            for j in range(div_z):
                angle_z = z_0 + j * increment_z
                vertices[index] = [0, angle_y, angle_z]
                index += 1

        index = _np.zeros((div_y * div_z * 12))
        cont = 0
        for i in range(div_y - 1):
            for j in range(div_z - 1):
                current_index = i * div_z + j
                next_index = current_index + div_z
                index[cont : cont + 3] = [current_index, current_index + 1, next_index]
                index[cont + 3 : cont + 6] = [
                    current_index + 1,
                    next_index,
                    next_index + 1,
                ]
                index[cont + 6 : cont + 9] = [current_index, next_index, next_index + 1]
                index[cont + 9 : cont + 12] = [
                    current_index,
                    next_index + 1,
                    current_index + 1,
                ]
                cont += 12

        td_plane = k3d.mesh(vertices, index, color_map=matplotlib_color_maps.viridis)
        td_plane.attribute = abs((postprocess_plane.field.total_field))
        td_plane.color_range = [
            min(abs(postprocess_plane.field.total_field)),
            max(abs(postprocess_plane.field.total_field)),
        ]

        self.k3d_planes.append(td_plane)

    def compute_fields(self):
        """Calculate the total pressure field in the planar grid created."""

        total_field_dirichlet = self.model.solution[0]

        self.mesh3d.attribute = abs(total_field_dirichlet.coefficients)
        maximum = max(abs(total_field_dirichlet.coefficients))
        minimum = min(abs(total_field_dirichlet.coefficients))

        self.mesh3d.color_range = [minimum, maximum]

    def display_field(self, surface=True, planes=True):
        """
        Display surface and/or planes used for visualisation.

        Parameters
        ----------
        surface : bool
            Display the surface field on the mesh.
        planes : bool
            Display the field in the visualisation planes.
        """

        import k3d

        max_search = _np.zeros(len(self.k3d_planes) + 1)

        max_search[-1] = _np.max(_np.abs(self.mesh3d.color_range))
        for i in range(_np.shape(self.k3d_planes)[0]):
            max_search[i] = _np.max(_np.abs(self.k3d_planes[i].color_range))

        map = [0, _np.max(max_search)]

        for i in range(_np.shape(self.k3d_planes)[0]):
            self.k3d_planes[i].color_range = map

        self.mesh3d.color_range = map

        plot = k3d.plot()
        if surface:
            plot += self.mesh3d

        if planes:
            for plane in self.k3d_planes:
                plot += plane

        plot.display()


class Visualise3DGrid(_PostProcess):
    def __init__(self, geometries, verbose=False):
        """
        Visualise the mesh in three dimensions.

        Parameters
        ----------
        geometries : list[optimus.geometry.Geometry]
            A list of geometries to visualise.
        """
        self.geometries = geometries
        self.mesh3ds = []

    def create_computational_grid(self):
        """Preprocess the grids for the visualisation"""

        import meshio
        import k3d
        import os

        for i in range(len(self.geometries)):
            self.geometries[i - 1].export_mesh("dummy.msh")

            mesh = meshio.read("dummy.msh")
            os.remove("dummy.msh")

            indices = mesh.cells["triangle"]
            vertices = mesh.points
            self.mesh3ds.append(
                k3d.mesh(vertices, indices, wireframe=True, color=0x9494A5)
            )

    def display_field(self):
        """Display surface grids"""

        import k3d

        plot = k3d.plot()
        for i in range(len(self.mesh3ds)):
            plot += self.mesh3ds[i]

        plot.display()
