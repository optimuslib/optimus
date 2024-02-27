"""Common functionality for postprocessing."""

import numpy as _np


class PostProcess:
    def __init__(self, model, verbose=False):
        """
        Base class for field visualisation.

        Parameters
        ----------
        model: optimus.model.common.Model
            The model object must include the solution vectors, i.e.,
            the solve() function must have been executed.
        verbose: boolean
            Display the logs.
        """

        from ..model.common import ExteriorModel
        from ..model.nested import NestedModel
        from ..model.acoustics import Analytical
        from .acoustics import ExteriorField, AnalyticalField, NestedField

        self.verbose = verbose
        self.model = model

        # Retrieve all surface grids from the model. This is a list
        # of Bempp grids, for each active interface in the model.
        if isinstance(model, ExteriorModel):
            self.field = ExteriorField(model, verbose)
            self.domains_grids = [
                model.geometry[n_sub].grid for n_sub in range(model.n_subdomains)
            ]

        elif isinstance(model, Analytical):
            self.field = AnalyticalField(model, verbose)
            self.domains_grids = [model.geometry.grid]

        elif isinstance(model, NestedModel):
            self.field = NestedField(model, verbose)
            self.domains_grids = []
            for interface in model.topology.interface_nodes:
                if interface.geometry is not None:
                    self.domains_grids.append(interface.geometry.grid)

        else:
            raise TypeError("Model type not recognised: " + str(type(model)))

        return

    def create_computational_grid(self, **kwargs):
        """
        Create the grid on which to calculate the pressure field.

        This function needs to be overridden by a specific postprocessing type.
        """

        raise NotImplementedError

    def compute_fields(self):
        """
        Calculate the pressure field in the specified locations.

        This function needs to be overridden by a specific postprocessing type.
        """

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

        return


def ppi_calculator(bounding_box, resolution):
    """
    Convert resolution to diagonal ppi.

    Parameters
    ----------
    bounding_box : list[float]
        list of min and max of the 2D plane
    resolution : list[float]
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
