"""Acoustic models."""

import bempp.api as _bempp
import numpy as _np
from .common import Model as _Model
from .common import ExteriorModel as _ExteriorModel


def create_default_model(source, geometry, exterior, interior, label="default"):
    """Create an acoustics model with default settings.

    For multiple domains, a list of geometries and interior materials needs
    to be specified, with equal length. They are matched by order.

    Parameters
    ----------
    source : optimus.source.common.Source
        The Optimus representation of a source field.
    geometry : list of optimus.geometry.common.Geometry
        The Optimus representation of the geometry, with the grid of
        the scatterers. For multiple domains, provide a list of geometries.
    exterior : optimus.material.common.Material
        The Optimus representation of the material for the unbounded
        exterior region.
    interior : list of optimus.material.common.Material
        The Optimus representation of the material for the bounded
        scatterer. For multiple domains, provide a list of materials.
    label : str
        The label of the model.

    Returns
    -------
    model : optimus.Model
        The Optimus representation of the BEM model of acoustic wave propagation
        in the interior and exterior domains.
    """

    model = Pmchwt(
        source=source,
        geometry=geometry,
        material_exterior=exterior,
        material_interior=interior,
        preconditioner="mass",
        label=label,
    )
    return model


def create_acoustic_model(
    source,
    geometry,
    exterior,
    interior,
    formulation="pmchwt",
    formulation_parameters=None,
    preconditioner="mass",
    preconditioner_parameters=None,
    label="custom",
):
    """
    Create a preconditioned boundary integral equation for acoustic wave propagation.

    For multiple domains, a list of geometries and interior materials need
    to be specified, with equal length. They are matched by order.

    Parameters
    ----------
    source : optimus.source.common.Source
        The Optimus representation of a source field.
    geometry : list of optimus.geometry.common.Geometry
        The Optimus representation of the geometry, with the grid of
        the scatterers. For multiple domains, provide a list of geometries.
    exterior : optimus.material.common.Material
        The Optimus representation of the material for the unbounded
        exterior region.
    interior : list of optimus.material.common.Material
        The Optimus representation of the material for the bounded
        scatterer. For multiple domains, provide a list of materials.
    formulation : str
        The type of boundary integral formulation.
    formulation_parameters : dict
        The parameters for the boundary integral formulation.
    preconditioner : str
        The type of operator preconditioner.
    preconditioner_parameters : dict
        The parameters for the operator preconditioner.
    label : str
        The label of the model.

    Returns
    -------
    model : optimus.model.common.ExteriorModel
        The Optimus representation of the BEM model of acoustic wave
        propagation in the interior and exterior domains.
    """

    from .formulations import check_validity_exterior_formulation

    form_name, prec_name, model_params = check_validity_exterior_formulation(
        formulation, formulation_parameters, preconditioner, preconditioner_parameters
    )

    if form_name == "pmchwt":
        model = Pmchwt(
            source=source,
            geometry=geometry,
            material_exterior=exterior,
            material_interior=interior,
            preconditioner=prec_name,
            parameters=model_params,
            label=label,
        )
    else:
        raise NotImplementedError

    return model


class Pmchwt(_ExteriorModel):
    def __init__(
        self,
        source,
        geometry,
        material_exterior,
        material_interior,
        preconditioner,
        parameters=None,
        label="pmchwt",
    ):
        """
        Create a model based on the PMCHWT formulation.

        Parameters
        ----------
        source : optimus.source.common.Source
            The Optimus representation of a source field.
        geometry : list of optimus.geometry.common.Geometry
            The Optimus representation of the geometry, with the grid of
            the scatterers. For multiple domains, provide a list of geometries.
        material_exterior : optimus.material.common.Material
            The Optimus representation of the material for the unbounded
            exterior region.
        material_interior : list of optimus.material.common.Material
            The Optimus representation of the material for the bounded
            scatterer. For multiple domains, provide a list of materials.
        preconditioner : str
            The type of operator preconditioner.
        parameters : dict
            The parameters for the formulation and preconditioner.
        label : str
            The label of the model.
        """

        super().__init__(
            source,
            geometry,
            material_exterior,
            material_interior,
            "pmchwt",
            preconditioner,
            parameters,
            label,
        )

        return

    def solve(self):
        """
        Solve the PMCHWT model.

        Perform the BEM algorithm: create the operators, assemble the matrix,
        and solve the linear system.
        """
        from optimus import global_parameters

        global_parameters.bem.update_hmat_parameters("boundary")
        self._create_function_spaces()
        self._create_continuous_operator()
        self._create_preconditioner()
        self._create_rhs_vector()
        self._create_discrete_system()
        self._solve_linear_system()
        self._solution_vector_to_gridfunction()

        return

    def _create_function_spaces(self):
        """
        Create the function spaces for the boundary integral operators.

        Continuous P1 elements will always be used for the Helmholtz equation.

        Sets "self.space" to a list of bempp.api.FunctionSpace, one for each interface.
        """

        self.space = [
            _bempp.function_space(geom.grid, "P", 1) for geom in self.geometry
        ]

        return

    def _create_continuous_operator(self):
        """
        Create the continuous boundary integral operators of the system.

        Sets "self.continous_operator" to a bempp.api.BlockedOperator.
        """

        freq = self.source.frequency
        k_ext = self.material_exterior.compute_wavenumber(freq)
        k_int = [mat.compute_wavenumber(freq) for mat in self.material_interior]
        rho_ext = self.material_exterior.density
        rho_int = [mat.density for mat in self.material_interior]

        matrix = _bempp.BlockedOperator(2 * self.n_subdomains, 2 * self.n_subdomains)

        for dom in range(self.n_subdomains):
            for ran in range(self.n_subdomains):
                operators_ext = create_boundary_integral_operators(
                    self.space[dom],
                    self.space[ran],
                    k_ext,
                    identity=False,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
                row = 2 * ran
                col = 2 * dom
                matrix[row, col] = -operators_ext["double_layer"]
                matrix[row, col + 1] = operators_ext["single_layer"]
                matrix[row + 1, col] = operators_ext["hypersingular"]
                matrix[row + 1, col + 1] = operators_ext["adjoint_double_layer"]

        for dom in range(self.n_subdomains):
            operators_int = create_boundary_integral_operators(
                self.space[dom],
                self.space[dom],
                k_int[dom],
                identity=False,
                single_layer=True,
                double_layer=True,
                adjoint_double_layer=True,
                hypersingular=True,
            )
            matrix[2 * dom, 2 * dom] += -operators_int["double_layer"]
            matrix[2 * dom, 2 * dom + 1] += (rho_int[dom] / rho_ext) * operators_int[
                "single_layer"
            ]
            matrix[2 * dom + 1, 2 * dom] += (rho_ext / rho_int[dom]) * operators_int[
                "hypersingular"
            ]
            matrix[2 * dom + 1, 2 * dom + 1] += operators_int["adjoint_double_layer"]

        self.continous_operator = matrix

        return

    def _create_preconditioner(self):
        """
        Assemble the operator preconditioner for the linear system.

        Sets "self.discrete_preconditioner" to a bempp.api.BlockedDiscreteOperator.
        """

        if self.preconditioner == "none":
            self.discrete_preconditioner = None

        elif self.preconditioner == "mass":
            mass_matrices = [create_inverse_mass_matrix(space) for space in self.space]

            preconditioner_list = []
            for row in range(2 * self.n_subdomains):
                ran = row // 2
                row_list = []
                for col in range(2 * self.n_subdomains):
                    if row == col:
                        row_list.append(mass_matrices[ran])
                    else:
                        row_list.append(None)
                preconditioner_list.append(row_list)

            self.discrete_preconditioner = _bempp.BlockedDiscreteOperator(
                preconditioner_list
            )

        elif self.preconditioner == "osrc":
            if self.parameters["osrc_wavenumber"] == "ext":
                k_osrc = [
                    self.material_exterior.compute_wavenumber(self.source.frequency)
                ] * self.n_subdomains
            elif self.parameters["osrc_wavenumber"] == "int":
                k_osrc = [
                    material.compute_wavenumber(self.source.frequency)
                    for material in self.material_interior
                ]
            else:
                k_osrc = [self.parameters["osrc_wavenumber"]] * self.n_subdomains

            osrc_ops = [
                create_osrc_operators(
                    self.space[dom], k_osrc[dom], self.parameters, dtn=True, ntd=True
                )
                for dom in range(self.n_subdomains)
            ]

            dtn_matrices = [op[0].weak_form() for op in osrc_ops]
            ntd_matrices = [op[1].weak_form() for op in osrc_ops]

            mass_matrices = [create_inverse_mass_matrix(space) for space in self.space]

            preconditioner_list = []
            for row in range(2 * self.n_subdomains):
                ran = row // 2
                row_list = []
                for col in range(2 * self.n_subdomains):
                    dom = col // 2
                    if ran == dom:
                        mass_op = mass_matrices[ran]
                        if row == col + 1:
                            prec_op = mass_op * dtn_matrices[ran] * mass_op
                        elif col == row + 1:
                            prec_op = mass_op * ntd_matrices[ran] * mass_op
                        else:
                            prec_op = None
                    else:
                        prec_op = None
                    row_list.append(prec_op)
                preconditioner_list.append(row_list)

            self.discrete_preconditioner = _bempp.BlockedDiscreteOperator(
                preconditioner_list
            )

        else:
            raise ValueError("Unknown preconditioner type: " + str(self.preconditioner))

        return

    def _create_rhs_vector(self):
        """
        Assemble the right-hand-side vector of the linear system.

        Calculate the projection of the incident field on the function
        spaces at each interface. The projection corresponds to the
        weak formulation. Concatenate all projections to form the
        righ-hand-side vector of the unpreconditioned system.

        Sets "self.rhs_vector" to a numpy.ndarray.
        """

        inc_trace_projections = []
        for space in self.space:
            inc_traces = self.source.calc_surface_traces(
                medium=self.material_exterior,
                space_dirichlet=space,
                space_neumann=space,
                dirichlet_trace=True,
                neumann_trace=True,
            )
            for trace in inc_traces:
                inc_trace_projections.append(trace.projections())

        self.rhs_vector = _np.concatenate(inc_trace_projections)

        return

    def _create_discrete_system(self):
        """
        Discretise the system.

        Sets "self.discrete_operator" to the weak form of the model.

        Sets "self.lhs_discrete_system" and "self.rhs_discrete_system" to
        the preconditioned discrete operator and the preconditioned
        right-hand-side vector of the model.
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

        return

    def _solve_linear_system(self):
        """
        Solve the linear system of boundary integral equations.

        Use a linear solver to solve the discrete system of the preconditined
        boundary integral formulation.

        Sets "self.solution_vector" to the numpy.ndarray containing
        the solution vector.

        Sets "self.iteration_count" to the number of iterations of
        the linear solver.
        """

        from .linalg import linear_solve

        self.solution_vector, self.iteration_count = linear_solve(
            self.lhs_discrete_system,
            self.rhs_discrete_system,
            return_iteration_count=True,
        )

        return

    def _solution_vector_to_gridfunction(self):
        """
        Convert the solution vector to grid functions.

        Splits the solution vector into independent grid functions for
        each separate trace on each interface.

        Sets "self.solution" to a list of bempp.api.GridFunction.
        """

        from .common import _vector_to_gridfunction

        list_of_spaces = []
        for space in self.space:
            list_of_spaces.extend([space, space])

        self.solution = _vector_to_gridfunction(
            self.solution_vector,
            list_of_spaces,
        )

        return


class Analytical(_Model):
    def __init__(
        self,
        source,
        geometry,
        material_exterior,
        material_interior,
        label="analytical",
    ):
        """
        Create a model based on the Analytical formulation
        for the scattering of a plane wave on a homogenous
        sphere.

        Parameters
        ----------
        source : optimus.source.common.Source
            The Optimus representation of a source field.
        geometry : optimus.geometry.common.Geometry
            The geometry in Optimus representation that includes the grid of
            the scatterer.
        material_exterior : optimus.material.common.Material
            The material for the unbounded exterior region.
        material_interior : optimus.material.common.Material
            The material for the bounded scatterer.
        label : str
            The label of the model.
        """
        super().__init__(label)
        self.formulation = "analytical"

        self.source = self._setup_source(source)
        self.frequency = self.source.frequency

        self.geometry = self._setup_geometry(geometry)
        self.material_exterior = material_exterior
        self.material_interior = self._setup_material_interior(material_interior)

        self.scattered_coefficients = None
        self.interior_coefficients = None

    def _setup_source(self, source):
        """
        Checks for the source being a plane wave field.

        Parameters
        ----------
        source : optimus.source.common.Source
            The Optimus representation of a source field.

        Returns
        -------
        source : optimus.source.common.Source
            The Optimus representation of a source field.
        """
        from ..source.planewave import PlaneWave

        if not isinstance(source, PlaneWave):
            raise NotImplementedError("Analytical model supports a planewave only.")

        return source

    def _setup_geometry(self, geometry):
        """
        Check if geometry is set to a single sphere.

        Parameters
        ----------
        geometry : optimus.geometry.common.Geometry
            The geometry in Optimus representation that includes the grid of
            the scatterer.

        Returns
        -------
        geometry : optimus.geometry.common.Geometry
            The spherical geometry
        """

        if isinstance(geometry, (list, tuple)):
            if len(geometry) > 1:
                raise NotImplementedError(
                    "Analytical model takes a single subdomain (sphere)."
                )
            else:
                geometry = geometry[0]

        if geometry.shape != "sphere":
            raise NotImplementedError(
                "Analytical model is available only for the sphere."
                + " Not for "
                + self.geometry.shape
            )

        return geometry

    def _setup_material_interior(self, material):
        """
        Check for single material.

        Parameters
        ----------
        material : optimus.material.common.Material
            The material for the bounded scatterer.

        Returns
        -------
        material : optimus.material.common.Material
            The material for the bounded scatterer.
        """

        if isinstance(material, (list, tuple)):
            if len(material) > 1:
                raise NotImplementedError(
                    "Analytical model does not support multiple subdomains."
                )
            else:
                material = material[0]

        return material

    def solve(self, n_iter=100):
        """
        Compute analytical coefficients.

        Parameters
        ----------
        n_iter : int
            number of coefficients terms to be computed
        """
        from scipy.special import sph_jn, sph_yn

        self.scattered_coefficients = _np.full(n_iter, _np.nan, dtype=_np.complex128)
        self.interior_coefficients = _np.full(n_iter, _np.nan, dtype=_np.complex128)

        k_ext = self.material_exterior.compute_wavenumber(self.source.frequency)
        k_int = self.material_interior.compute_wavenumber(self.source.frequency)

        rho_ext = self.material_exterior.density
        rho_int = self.material_interior.density

        rho = rho_int / rho_ext
        k = k_ext / k_int

        r = self.geometry.radius

        #
        # Compute spherical Bessel function for the exterior and interior domain
        # hn = jn - i*yn, denotes the Hankel function of second kind.
        #
        jn_ext, d_jn_ext = sph_jn(n_iter, k_ext * r)
        yn_ext, d_yn_ext = sph_yn(n_iter, k_ext * r)
        h1n_ext, d_h1n_ext = (jn_ext + 1j * yn_ext, d_jn_ext + 1j * d_yn_ext)

        jn_int, d_jn_int = sph_jn(n_iter, k_int * r)

        coef_sca = (d_jn_int * jn_ext - rho * k * jn_int * d_jn_ext) / (
            rho * k * jn_int * d_h1n_ext - d_jn_int * h1n_ext
        )

        weights = _np.array([(2 * n + 1) * 1j**n for n in range(n_iter + 1)])

        self.scattered_coefficients = coef_sca * weights
        self.interior_coefficients = ((jn_ext + coef_sca * h1n_ext) / jn_int) * weights


class AnalyticalTwoSpheres(_Model):
    def __init__(
        self,
        source,
        geometry,
        material_exterior,
        material_interior,
        label="analytical",
    ):
        """
        Create a model based on the Analytical formulation
        for the scattering of a plane wave on a homogenous
        sphere.

        Parameters
        ----------
        source : optimus.source.common.Source
            The Optimus representation of a source field.
        geometry : optimus.geometry.common.Geometry
            The geometry in Optimus representation that includes the grid of
            the scatterer.
        material_exterior : optimus.material.common.Material
            The material for the unbounded exterior region.
        material_interior : optimus.material.common.Material
            The material for the bounded scatterer.
        label : str
            The label of the model.
        """
        super().__init__(label)
        self.formulation = "analytical"

        self.source = self._setup_source(source)
        self.frequency = self.source.frequency

        self.geometry = self._setup_geometry(geometry)
        self.material_exterior = material_exterior
        self.material_interior = self._setup_material_interior(material_interior)

        self.scattered_coefficients = None
        self.interior_coefficients = None

    def _setup_source(self, source):
        """
        Checks for the source being a plane wave field.

        Parameters
        ----------
        source : optimus.source.common.Source
            The Optimus representation of a source field.

        Returns
        -------
        source : optimus.source.common.Source
            The Optimus representation of a source field.
        """
        from ..source.planewave import PlaneWave

        if not isinstance(source, PlaneWave):
            raise NotImplementedError("Analytical model supports a planewave only.")

        return source

    def _setup_geometry(self, geometry):
        """
        Check if geometry is set to a single sphere.

        Parameters
        ----------
        geometry : optimus.geometry.common.Geometry
            The geometry in Optimus representation that includes the grid of
            the scatterer.

        Returns
        -------
        geometry : optimus.geometry.common.Geometry
            The spherical geometry
        """

        if isinstance(geometry, (list, tuple)):
            if len(geometry) > 2:
                raise NotImplementedError(
                    "AnalyticalTwoSpheres model takes exactly two subdomains."
                )
        else:
            raise NotImplementedError(
                "For a single subdomain, use the Analytical class."
            )

        for count in range(len(geometry)):
            if geometry[count].shape != "sphere":
                raise NotImplementedError(
                    "AnalyticalTwoSpheres model is available only for spheres."
                    + " Not for "
                    + geometry[count].shape
                )

        if geometry[0].origin != geometry[1].origin:
            raise NotImplementedError(
                "AnalyticalTwoSpheres model requires spheres to be concentric."
            )

        if geometry[0].radius < geometry[1].radius:
            raise NotImplementedError(
                "AnalyticalTwoSpheres model requires first sphere to have larger\
                    radius than second sphere."
            )

        return geometry

    def _setup_material_interior(self, material):
        """
        Check for single material.

        Parameters
        ----------
        material : optimus.material.common.Material
            The material for the bounded scatterer.

        Returns
        -------
        material : optimus.material.common.Material
            The material for the bounded scatterer.
        """

        if isinstance(material, (list, tuple)):
            if len(material) > 2:
                raise NotImplementedError(
                    "AnalyticalTwoSphere model does not support more than 2 subdomains."
                )
        else:
            raise NotImplementedError(
                "AnalyticalTwoSphere model must have exactly 2 subdomains."
            )

        return material

    def solve(self, n_iter=100):
        """
        Compute analytical coefficients for scattering of plane wave by two concentric
        spheres based on the formulation derived by McNew J, Lavarello R, O’Brien Jr WD.
        Sound scattering from two concentric fluid spheres. The Journal of the
        Acoustical Society of America. 2009 Jan;125(1):1-4.

        Parameters
        ----------
        n_iter : int
            number of coefficients terms to be computed
        """
        from scipy.special import sph_jn, sph_yn

        self.scattered_coefficients = _np.full(n_iter, _np.nan, dtype=_np.complex128)

        k_ext = self.material_exterior.compute_wavenumber(self.source.frequency)

        weights = _np.array([(2 * n + 1) * 1j**n for n in range(n_iter + 1)])

        k_int = []
        for material in self.material_interior:
            k_int.append(material.compute_wavenumber(self.source.frequency))

        density_wavenumber_ratio = []
        for material in [self.material_exterior] + self.material_interior:
            density_wavenumber_ratio.append(
                material.density / material.compute_wavenumber(self.source.frequency)
            )

        Z = tuple(
            [
                2 * _np.pi * self.source.frequency * rho_k_ratio
                for rho_k_ratio in density_wavenumber_ratio
            ]
        )

        r = []
        for geometry in self.geometry:
            r.append(geometry.radius)

        #
        # Compute spherical Bessel function for the exterior and interior domains
        # hn = jn - i*yn, denotes the Hankel function of second kind.
        #
        jn_k0r1, d_jn_k0r1 = sph_jn(n_iter, k_ext * r[0])
        jn_k1r1, d_jn_k1r1 = sph_jn(n_iter, k_int[0] * r[0])
        jn_k1r2, d_jn_k1r2 = sph_jn(n_iter, k_int[0] * r[1])
        jn_k2r2, d_jn_k2r2 = sph_jn(n_iter, k_int[1] * r[1])

        yn_k0r1, d_yn_k0r1 = sph_yn(n_iter, k_ext * r[0])
        yn_k1r1, d_yn_k1r1 = sph_yn(n_iter, k_int[0] * r[0])
        yn_k1r2, d_yn_k1r2 = sph_yn(n_iter, k_int[0] * r[1])

        h1n_k0r1, d_h1n_k0r1 = (jn_k0r1 + 1j * yn_k0r1, d_jn_k0r1 + 1j * d_yn_k0r1)
        h1n_k1r1, d_h1n_k1r1 = (jn_k1r1 + 1j * yn_k1r1, d_jn_k1r1 + 1j * d_yn_k1r1)
        h1n_k1r2, d_h1n_k1r2 = (jn_k1r2 + 1j * yn_k1r2, d_jn_k1r2 + 1j * d_yn_k1r2)

        H = _np.zeros((4, 4), dtype=_np.complex128)
        b = _np.zeros((4), dtype=_np.complex128)

        solution_coefficients = _np.zeros((4, n_iter + 1), _np.complex128)

        # Obtain the coefficients for each harmonic through matrix inversion
        for i in range(n_iter + 1):
            H[0, 0] = d_h1n_k0r1[i] / Z[0]
            H[0, 1] = -d_h1n_k1r1[i] / Z[1]
            H[0, 2] = -d_jn_k1r1[i] / Z[1]

            H[1, 0] = -h1n_k0r1[i]
            H[1, 1] = h1n_k1r1[i]
            H[1, 2] = jn_k1r1[i]

            H[2, 1] = d_h1n_k1r2[i] / Z[1]
            H[2, 2] = d_jn_k1r2[i] / Z[1]
            H[2, 3] = -d_jn_k2r2[i] / Z[2]

            H[3, 1] = h1n_k1r2[i]
            H[3, 2] = jn_k1r2[i]
            H[3, 3] = -jn_k2r2[i]

            b[0] = -weights[i] * d_jn_k0r1[i] / Z[0]
            b[1] = weights[i] * jn_k0r1[i]

            solution_coefficients[:, i] = _np.linalg.solve(H, b)

        self.scattered_coefficients = solution_coefficients[0, :]
        self.interior_coefficients = solution_coefficients[1::]


def create_boundary_integral_operators(
    space_domain,
    space_range,
    wavenumber,
    identity=False,
    single_layer=False,
    double_layer=False,
    adjoint_double_layer=False,
    hypersingular=False,
):
    """Create boundary integral operators of the Helmholtz equation.

    Parameters
    ----------
    space_domain, space_range : bempp.api.FunctionSpace
        The function space for the domain and range of the Galerkin discretisation.
    wavenumber : complex
        The wavenumber of the Green's function.
    identity : bool
        Return the continuous identity boundary integral operator.
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
    operators : dict[str, bempp.api.operators.boundary]
        A dictionary of boundary integral operators of the Helmholtz equation.
    """

    operators = {}

    if identity:
        id_op = _bempp.operators.boundary.sparse.identity(
            space_domain,
            space_range,
            space_range,
        )
        operators["identity"] = id_op

    if _np.isclose(_np.abs(wavenumber), 0):
        if single_layer:
            sl_op = _bempp.operators.boundary.laplace.single_layer(
                space_domain,
                space_range,
                space_range,
                use_projection_spaces=False,
            )
            operators["single_layer"] = sl_op

        if double_layer:
            dl_op = _bempp.operators.boundary.laplace.double_layer(
                space_domain,
                space_range,
                space_range,
                use_projection_spaces=False,
            )
            operators["double_layer"] = dl_op

        if adjoint_double_layer:
            ad_op = _bempp.operators.boundary.laplace.adjoint_double_layer(
                space_domain,
                space_range,
                space_range,
                use_projection_spaces=False,
            )
            operators["adjoint_double_layer"] = ad_op

        if hypersingular:
            hs_op = _bempp.operators.boundary.laplace.hypersingular(
                space_domain,
                space_range,
                space_range,
                use_projection_spaces=False,
            )
            operators["hypersingular"] = hs_op

    else:
        if single_layer:
            sl_op = _bempp.operators.boundary.helmholtz.single_layer(
                space_domain,
                space_range,
                space_range,
                wavenumber,
                use_projection_spaces=False,
            )
            operators["single_layer"] = sl_op

        if double_layer:
            dl_op = _bempp.operators.boundary.helmholtz.double_layer(
                space_domain,
                space_range,
                space_range,
                wavenumber,
                use_projection_spaces=False,
            )
            operators["double_layer"] = dl_op

        if adjoint_double_layer:
            ad_op = _bempp.operators.boundary.helmholtz.adjoint_double_layer(
                space_domain,
                space_range,
                space_range,
                wavenumber,
                use_projection_spaces=False,
            )
            operators["adjoint_double_layer"] = ad_op

        if hypersingular:
            hs_op = _bempp.operators.boundary.helmholtz.hypersingular(
                space_domain,
                space_range,
                space_range,
                wavenumber,
                use_projection_spaces=False,
            )
            operators["hypersingular"] = hs_op

    return operators


def create_inverse_mass_matrix(space):
    """
    Create the inverse mass matrix of the function space.

    Parameters
    ----------
    space : bempp.api.FunctionSpace
        The function space for the domain and range of the Galerkin discretisation.

    Returns
    -------
    matrix : linear operator
        The linear operator with the sparse LU factorisation of the inverse
        mass matrix.
    """

    id_op = _bempp.operators.boundary.sparse.identity(
        space,
        space,
        space,
    )
    id_wf = id_op.weak_form()
    id_inv = _bempp.InverseSparseDiscreteBoundaryOperator(id_wf)

    return id_inv


def create_osrc_operators(
    space,
    wavenumber,
    parameters,
    dtn=False,
    ntd=False,
):
    """
    Create OSRC operators of the Helmholtz equation.

    Parameters
    ----------
    space : bempp.api.FunctionSpace
        The function space for the domain and range of the Galerkin discretisation.
    wavenumber : complex
        The wavenumber of the Green's function.
    parameters : dict
        The parameters of the OSRC operators.
    dtn : bool
        Return the OSRC approximated DtN map.
    ntd : bool
        Return the OSRC approximated NtD map.

    Returns
    -------
    operators : list[bempp.api.operators.boundary.Helmholtz]
        A list of OSRC operators of the Helmholtz equation
    """

    operators = []

    if dtn:
        dtn_op = _bempp.operators.boundary.helmholtz.osrc_dtn(
            space,
            wavenumber,
            npade=parameters["osrc_npade"],
            theta=parameters["osrc_theta"],
            damped_wavenumber=parameters["osrc_damped_wavenumber"],
        )
        operators.append(dtn_op)

    if ntd:
        ntd_op = _bempp.operators.boundary.helmholtz.osrc_ntd(
            space,
            wavenumber,
            npade=parameters["osrc_npade"],
            theta=parameters["osrc_theta"],
            damped_wavenumber=parameters["osrc_damped_wavenumber"],
        )
        operators.append(ntd_op)

    return operators
