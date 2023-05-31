"""Common functionality for acoustic sources."""

import bempp.api as _bempp
import numpy as _np

from ..utils.conversions import convert_to_positive_float as _convert_to_positive_float


class Source:
    def __init__(
        self,
        source_type,
        frequency,
    ):
        """
        The source of the acoustic wave.

        Parameters
        ----------
        source_type : str
            The name of the source.
        frequency : float
            The frequency of the wave field.
        """

        self.type = source_type
        self.frequency = _convert_to_positive_float(
            frequency, "frequency", nonnegative=True
        )

    def pressure_field(self, medium, locations):
        """
        Calculate the pressure field in the specified locations. Needs to be overridden by specific source type.
        """

        raise NotImplementedError

    def normal_pressure_gradient(self, medium, locations, normals):
        """
        Calculate the normal gradient of the pressure field in the
        specified locations. Needs to be overridden by specific source type.
        """

        raise NotImplementedError

    def pressure_field_and_normal_gradient(self, medium, locations, normals):
        """
        Calculate the pressure field and the normal gradient of the pressure
        field in the specified locations.
        """

        return (
            self.pressure_field(medium, locations),
            self.normal_pressure_gradient(medium, locations, normals),
        )

    def calc_surface_traces(
        self,
        medium,
        space_dirichlet=None,
        space_neumann=None,
        dirichlet_trace=True,
        neumann_trace=True,
    ):
        """
        Calculate the surface traces of the source field on the mesh. Needs to be overridden by specific source type.
        """

        raise NotImplementedError

    def _calc_surface_traces_from_function(
        self,
        medium,
        space_dirichlet=None,
        space_neumann=None,
        dirichlet_trace=True,
        neumann_trace=True,
    ):
        """
        Calculate the surface traces of the source field from explicit
        functions for the pressure and its normal gradient.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        space_dirichlet, space_neumann : bempp.api.FunctionSpace
            The discrete spaces on the surface grid.
        dirichlet_trace, neumann_trace : bool
            Calculate the Dirichlet or Neumann trace of the field.

        Returns
        -------
        trace : bempp.api.GridFunction
            The surface traces.
        """

        if dirichlet_trace:
            if space_dirichlet is None:
                raise ValueError(
                    "The Dirichlet space needs to be specified to calculate "
                    "the Dirichlet trace."
                )

            def dirichlet_fun(x, n, domain_index, result):
                result[0] = self.pressure_field(medium, x)

            trace_dirichlet = _bempp.GridFunction(space_dirichlet, fun=dirichlet_fun)
        else:
            trace_dirichlet = None

        if neumann_trace:
            if space_neumann is None:
                raise ValueError(
                    "The Neumann space needs to be specified to calculate "
                    "the Neumann trace."
                )

            def neumann_fun(x, n, domain_index, result):
                result[0] = self.normal_pressure_gradient(medium, x, n)

            trace_neumann = _bempp.GridFunction(space_neumann, fun=neumann_fun)
        else:
            trace_neumann = None

        if dirichlet_trace and neumann_trace:
            return trace_dirichlet, trace_neumann
        elif dirichlet_trace:
            return trace_dirichlet
        elif neumann_trace:
            return trace_neumann

    def _calc_surface_traces_from_coefficients(
        self,
        medium,
        space_dirichlet=None,
        space_neumann=None,
        dirichlet_trace=True,
        neumann_trace=True,
    ):
        """Calculate the surface traces of the source field from values of the
        pressure and its normal gradient in the degrees of freedom.

        Parameters
        ----------
        medium : optimus.material.Material
            The propagating medium.
        space_dirichlet, space_neumann : bempp.api.FunctionSpace
            The discrete spaces on the surface grid.
        dirichlet_trace, neumann_trace : bool
            Calculate the Dirichlet or Neumann trace of the field.

        Returns
        -------
        trace : bempp.api.GridFunction
            The surface traces.
        """

        if dirichlet_trace:
            if space_dirichlet is None:
                raise ValueError(
                    "The Dirichlet space needs to be specified to calculate "
                    "the Dirichlet trace."
                )
            points_dirichlet = space_dirichlet.global_dof_interpolation_points
        else:
            points_dirichlet = None

        if neumann_trace:
            if space_neumann is None:
                raise ValueError(
                    "The Neumann space needs to be specified to calculate "
                    "the Neumann trace."
                )
            points_neumann = space_neumann.global_dof_interpolation_points
            normals_neumann = space_neumann.global_dof_normals
        else:
            points_neumann = None
            normals_neumann = None

        if dirichlet_trace and neumann_trace:

            if _np.all(points_dirichlet == points_neumann):
                (field_val, gradient_val,) = self.pressure_field_and_normal_gradient(
                    medium, points_dirichlet, normals_neumann
                )
            else:
                field_val = self.pressure_field(medium, points_dirichlet)
                gradient_val = self.normal_pressure_gradient(
                    medium, points_neumann, normals_neumann
                )
            trace_dirichlet = _bempp.GridFunction(
                space_dirichlet, coefficients=field_val
            )
            trace_neumann = _bempp.GridFunction(
                space_neumann, coefficients=gradient_val
            )
            return trace_dirichlet, trace_neumann

        elif dirichlet_trace:

            field_val = self.pressure_field(medium, points_dirichlet)
            trace_dirichlet = _bempp.GridFunction(
                space_dirichlet, coefficients=field_val
            )
            return trace_dirichlet

        elif neumann_trace:

            gradient_val = self.normal_pressure_gradient(
                medium, points_neumann, normals_neumann
            )
            trace_neumann = _bempp.GridFunction(
                space_neumann, coefficients=gradient_val
            )
            return trace_neumann
