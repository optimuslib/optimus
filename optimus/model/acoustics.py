"""Acoustic models."""

import bempp.api as _bempp
import numpy as _np
from .common import Model as _Model


def create_default_model(source, geometry, exterior, interior):
    """
    Create an acoustics model with default settings.

    Parameters
    ----------
    source : optimus.source
        The Optimus representation of a source field.
    geometry : optimus.geometry
        The Optimus representation of the geometry, with the grid of
         the scatterers.
    exterior : optimus.material
        The Optimus representation of the material for the unbounded
         exterior region.
    interior : optimus.material
        The Optimus representation of the material for the bounded
         scatterers.
    """

    model = Pmchwt(
        source=source,
        geometry=geometry,
        material_exterior=exterior,
        material_interior=interior,
        preconditioner="mass",
    )

    return model


class Pmchwt(_Model):
    def __init__(
        self,
        source,
        geometry,
        material_exterior,
        material_interior,
        preconditioner,
        parameters=None,
    ):
        """
        Create a model based on the PMCHWT formulation.
        """
        super().__init__(
            source,
            geometry,
            material_exterior,
            material_interior,
            "pmchwt",
            preconditioner,
        )

        self.parameters = parameters
        self.solution = None

    def solve(self):
        """
        Solve the PMCHWT model.
        """

        from .common import _vector_to_gridfunction

        self._create_function_spaces()
        self._create_continuous_operator()
        self._create_preconditioner()
        self._create_rhs_vector()
        self._create_discrete_system()
        self._solve_linear_system()

        self.solution = _vector_to_gridfunction(
            self.solution_vector, self.surface_spaces
        )

    def _create_function_spaces(self):
        self.space = _bempp.function_space(self.geometry.grid, "P", 1)
        self.surface_spaces = (self.space, self.space)

    def _create_continuous_operator(self):

        freq = self.source.frequency
        k_ext = self.material_exterior.wavenumber(freq)
        k_int = self.material_interior.wavenumber(freq)
        rho_ext = self.material_exterior.density
        rho_int = self.material_interior.density

        sl_ext = _bempp.operators.boundary.helmholtz.single_layer(
            self.space,
            self.space,
            self.space,
            k_ext,
            use_projection_spaces=False,
        )
        sl_int = _bempp.operators.boundary.helmholtz.single_layer(
            self.space,
            self.space,
            self.space,
            k_int,
            use_projection_spaces=False,
        )
        dl_ext = _bempp.operators.boundary.helmholtz.double_layer(
            self.space,
            self.space,
            self.space,
            k_ext,
            use_projection_spaces=False,
        )
        dl_int = _bempp.operators.boundary.helmholtz.double_layer(
            self.space,
            self.space,
            self.space,
            k_int,
            use_projection_spaces=False,
        )
        ad_ext = _bempp.operators.boundary.helmholtz.adjoint_double_layer(
            self.space,
            self.space,
            self.space,
            k_ext,
            use_projection_spaces=False,
        )
        ad_int = _bempp.operators.boundary.helmholtz.adjoint_double_layer(
            self.space,
            self.space,
            self.space,
            k_int,
            use_projection_spaces=False,
        )
        hs_ext = _bempp.operators.boundary.helmholtz.hypersingular(
            self.space,
            self.space,
            self.space,
            k_ext,
            use_projection_spaces=False,
        )
        hs_int = _bempp.operators.boundary.helmholtz.hypersingular(
            self.space,
            self.space,
            self.space,
            k_int,
            use_projection_spaces=False,
        )

        self.continous_operator = _bempp.BlockedOperator(2, 2)
        self.continous_operator[0, 0] = -dl_ext - dl_int
        self.continous_operator[0, 1] = sl_ext + (rho_int / rho_ext) * sl_int
        self.continous_operator[1, 0] = hs_ext + (rho_ext / rho_int) * hs_int
        self.continous_operator[1, 1] = ad_ext + ad_int

    def _create_preconditioner(self):

        if self.preconditioner == "mass":

            id_op = _bempp.operators.boundary.sparse.identity(
                self.space, self.space, self.space
            )
            id_wf = id_op.weak_form()
            mass_matrix = _bempp.InverseSparseDiscreteBoundaryOperator(id_wf)

            self.discrete_preconditioner = _bempp.BlockedDiscreteOperator(
                [[mass_matrix, None], [None, mass_matrix]]
            )

        else:

            self.discrete_preconditioner = None

    def _create_rhs_vector(self):

        inc_traces = self.source.calc_surface_traces(
            medium=self.material_exterior,
            spaces=self.surface_spaces,
        )

        self.rhs_vector = _np.concatenate(
            (inc_traces[0].projections(), inc_traces[1].projections())
        )

    def _create_discrete_system(self):

        self.discrete_operator = self.continous_operator.weak_form()

        if self.discrete_preconditioner:

            self.lhs_discrete_system = (
                self.discrete_preconditioner * self.discrete_operator
            )
            self.rhs_discrete_system = self.discrete_preconditioner * self.rhs_vector

        else:

            self.lhs_discrete_system = self.discrete_operator
            self.rhs_discrete_system = self.rhs_vector

    def _solve_linear_system(self):

        from .linalg import linear_solve

        self.solution_vector = linear_solve(
            self.lhs_discrete_system,
            self.rhs_discrete_system,
        )
