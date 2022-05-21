"""Acoustic models."""

import bempp.api as _bempp
import numpy as _np
from .common import Model as _Model


def create_default_model(source, geometry, exterior, interior):
    """
    Create an acoustics model with default settings.

    For multiple domains, a list of geometries and interior materials need
    to be specified, with equal length. They are matched by order.

    Parameters
    ----------
    source : optimus.source
        The Optimus representation of a source field.
    geometry : optimus.geometry
        The Optimus representation of the geometry, with the grid of
        the scatterers. For multiple domains, provide a list of geometries.
    exterior : optimus.material
        The Optimus representation of the material for the unbounded
        exterior region.
    interior : optimus.material
        The Optimus representation of the material for the bounded
        scatterer. For multiple domains, provide a list of materials.

    Returns
    ----------
    model : optimus.model
        The Optimus representation of the the BEM model of acoustic wave
        propagation in the interior and exterior domains.
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

        self.space = None
        self.continous_operator = None
        self.discrete_operator = None
        self.discrete_preconditioner = None
        self.rhs_vector = None
        self.lhs_discrete_system = None
        self.rhs_discrete_system = None
        self.solution_vector = None
        self.solution = None

    def solve(self):
        """
        Solve the PMCHWT model.
        """

        self._create_function_spaces()
        self._create_continuous_operator()
        self._create_preconditioner()
        self._create_rhs_vector()
        self._create_discrete_system()
        self._solve_linear_system()
        self._solution_vector_to_gridfunction()

    def _create_function_spaces(self):
        """
        Create the function spaces for the boundary integral operators.
        Continuous P1 elements will always be used for the Helmholtz equation.
        """
        self.space = [
            _bempp.function_space(geom.grid, "P", 1) for geom in self.geometry
        ]

    def _create_continuous_operator(self):
        """
        Create the continuous boundary integral operators of the system.
        """

        freq = self.source.frequency
        k_ext = self.material_exterior.compute_wavenumber(freq)
        k_int = [mat.compute_wavenumber(freq) for mat in self.material_interior]
        rho_ext = self.material_exterior.density
        rho_int = [mat.density for mat in self.material_interior]

        matrix = _bempp.BlockedOperator(2 * self.n_subdomains, 2 * self.n_subdomains)

        for dom in range(self.n_subdomains):
            for ran in range(self.n_subdomains):
                sl_ext, dl_ext, ad_ext, hs_ext = create_boundary_integral_operators(
                    self.space[dom],
                    self.space[ran],
                    k_ext,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
                row = 2 * ran
                col = 2 * dom
                matrix[row, col] = -dl_ext
                matrix[row, col + 1] = sl_ext
                matrix[row + 1, col] = hs_ext
                matrix[row + 1, col + 1] = ad_ext

        for dom in range(self.n_subdomains):
            sl_int, dl_int, ad_int, hs_int = create_boundary_integral_operators(
                self.space[dom],
                self.space[dom],
                k_int[dom],
                single_layer=True,
                double_layer=True,
                adjoint_double_layer=True,
                hypersingular=True,
            )
            matrix[2 * dom, 2 * dom] += -dl_int
            matrix[2 * dom, 2 * dom + 1] += (rho_int[dom] / rho_ext) * sl_int
            matrix[2 * dom + 1, 2 * dom] += (rho_ext / rho_int[dom]) * hs_int
            matrix[2 * dom + 1, 2 * dom + 1] += ad_int

        self.continous_operator = matrix

    def _create_preconditioner(self):
        """
        Assemble the operator preconditioner for the linear system.
        """

        if self.preconditioner == "mass":

            preconditioner_list = []
            for row in range(2 * self.n_subdomains):
                ran = row // 2
                row_list = []
                for col in range(2 * self.n_subdomains):
                    dom = col // 2
                    if row == col:
                        id_op = _bempp.operators.boundary.sparse.identity(
                            self.space[dom], self.space[ran], self.space[ran]
                        )
                        id_wf = id_op.weak_form()
                        id_inv = _bempp.InverseSparseDiscreteBoundaryOperator(id_wf)
                        row_list.append(id_inv)
                    else:
                        row_list.append(None)
                preconditioner_list.append(row_list)

            self.discrete_preconditioner = _bempp.BlockedDiscreteOperator(
                preconditioner_list
            )

        else:

            self.discrete_preconditioner = None

    def _create_rhs_vector(self):
        """
        Assemble the right-hand-side vector of the linear system.
        """

        inc_trace_projections = []
        for space in self.space:
            inc_traces = self.source.calc_surface_traces(
                medium=self.material_exterior,
                spaces=(space, space),
            )
            for trace in inc_traces:
                inc_trace_projections.append(trace.projections())

        self.rhs_vector = _np.concatenate(inc_trace_projections)

    def _create_discrete_system(self):
        """
        Discretise the system.
        """

        self.discrete_operator = self.continous_operator.weak_form()

        if self.discrete_preconditioner is None:

            self.lhs_discrete_system = self.discrete_operator
            self.rhs_discrete_system = self.rhs_vector

        else:

            self.lhs_discrete_system = (
                self.discrete_preconditioner * self.discrete_operator
            )
            self.rhs_discrete_system = self.discrete_preconditioner * self.rhs_vector

    def _solve_linear_system(self):
        """
        Solve the linear system of boundary integral equations.
        """

        from .linalg import linear_solve

        self.solution_vector = linear_solve(
            self.lhs_discrete_system,
            self.rhs_discrete_system,
        )

    def _solution_vector_to_gridfunction(self):
        """
        Convert the solution vector in grid functions.
        """
        from .common import _vector_to_gridfunction

        list_of_spaces = []
        for space in self.space:
            list_of_spaces.extend([space, space])

        self.solution = _vector_to_gridfunction(
            self.solution_vector,
            list_of_spaces,
        )


def create_boundary_integral_operators(
    space_domain,
    space_range,
    wavenumber,
    single_layer=False,
    double_layer=False,
    adjoint_double_layer=False,
    hypersingular=False,
):
    """
    Create boundary integral operators of the Helmholtz equation.

    Parameters
    ----------
    space_domain, space_range : bempp.api.FunctionSpace
        The function space for the domain and range of the Galerkin discretisation.
    wavenumber : complex
        The wavenumber of the Green's function.
    single_layer : bool
        Return the continuous single layer boundary integral operator.
    double_layer : bool
        Return the continuous double layer boundary integral operator.
    adjoint_double_layer : bool
        Return the continuous adjoint double layer boundary integral operator.
    hypersingular : bool
        Return the continuous hypersingular layer boundary integral operator.

    Returns
    -------
    operators : bempp.api.operators.boundary.helmholtz
        A list of boundary integral operators of the Helmholtz equation
    """

    operators = []

    if single_layer:
        sl_op = _bempp.operators.boundary.helmholtz.single_layer(
            space_domain,
            space_range,
            space_range,
            wavenumber,
            use_projection_spaces=False,
        )
        operators.append(sl_op)

    if double_layer:
        dl_op = _bempp.operators.boundary.helmholtz.double_layer(
            space_domain,
            space_range,
            space_range,
            wavenumber,
            use_projection_spaces=False,
        )
        operators.append(dl_op)

    if adjoint_double_layer:
        ad_op = _bempp.operators.boundary.helmholtz.adjoint_double_layer(
            space_domain,
            space_range,
            space_range,
            wavenumber,
            use_projection_spaces=False,
        )
        operators.append(ad_op)

    if hypersingular:
        hs_op = _bempp.operators.boundary.helmholtz.hypersingular(
            space_domain,
            space_range,
            space_range,
            wavenumber,
            use_projection_spaces=False,
        )
        operators.append(hs_op)

    return operators
