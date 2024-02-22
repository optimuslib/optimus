"""Nested models."""

import numpy as _np
from .common import GraphModel as _GraphModel


def create_default_nested_model(topology, frequency, label="default"):
    """Create an acoustics model on nested domains with default settings.

    A boundary integral formulation will be created with default parameters, for
    a geometry consisting of nested subdomains.

    Parameters
    ----------
    topology : optimus.geometry.Graph
        The graph topology representing the geometry.
    frequency : float
        The frequency of the harmonic wave propagation model.
    label : str
        The label of the model.

    Returns
    -------
    model : optimus.Model
        The Optimus representation of the BEM model of acoustic wave propagation
        in the nested domains.
    """

    freq = _check_frequency(frequency, topology.subdomain_nodes)

    model = NestedModel(
        topology=topology,
        frequency=freq,
        formulation=["none"] + ["pmchwt"]*(topology.number_interface_nodes()-1),
        preconditioner=["none"] + ["mass"]*(topology.number_interface_nodes()-1),
        parameters=None,
        label=label,
    )

    return model


def create_nested_model(
    topology,
    frequency,
    formulation,
    preconditioner,
    parameters=None,
    label="nested",
):
    """Create an acoustics model on nested domains.

    A boundary integral formulation will be created for
    a geometry consisting of nested subdomains.

    Parameters
    ----------
    topology : optimus.geometry.Graph
        The graph topology representing the geometry.
    frequency : float
        The frequency of the harmonic wave propagation model.
    formulation : str, list[str]
        The type of formulation, possibly different for each interface.
    preconditioner : str, list[str]
        The type of operator preconditioner, possibly different for each interface.
    parameters : dict, None
        The parameters for the formulation and preconditioner.
    label : str
        The label of the model.

    Returns
    -------
    model : optimus.model.nested.NestedModel
        The Optimus representation of the BEM model of acoustic wave propagation
        in the nested domains.
    """

    freq = _check_frequency(frequency, topology.subdomain_nodes)

    form_names, prec_names = _check_preconditioned_formulation(
        formulation,
        preconditioner,
        topology.number_interface_nodes(),
    )

    model = NestedModel(
        topology=topology,
        frequency=freq,
        formulation=form_names,
        preconditioner=prec_names,
        parameters=parameters,
        label=label,
    )

    return model


def _check_frequency(frequency, subdomain_nodes):
    """
    Check validity of frequency.

    Check if all sources have the same frequency.

    Parameters
    ----------
    frequency : float
        The frequency of the harmonic wave propagation model.
    subdomain_nodes : list[optimus.geometry.graph_topology.SubdomainNode]
        The list of subdomain nodes.

    Returns
    -------
    freq : float
        The frequency of the harmonic wave propagation model.
    """

    from ..utils.conversions import convert_to_positive_float

    freq = convert_to_positive_float(frequency, label="frequency", nonnegative=True)

    sources = []
    for subdomain in subdomain_nodes:
        if subdomain.is_active():
            sources.extend(subdomain.sources)

    for source in sources:
        if source.frequency != freq:
            raise ValueError(
                "All sources must have frequency {}. Source {} has "
                "frequency {} instead.".format(
                    freq, source.label, source.frequency
                )
            )

    return freq


def _check_preconditioned_formulation(formulation, preconditioner, n_interfaces):
    """
    Check the validity of the preconditioned formulation.

    Parameters
    ----------
    formulation : str, list[str], tuple[str]
        The type of formulation, possibly different for each interface.
    preconditioner : str, list[str], tuple[str]
        The type of operator preconditioner, possibly different for each interface.
    n_interfaces : int
        The number of interfaces in the nested geometry, including the unbounded
        one at infinity.

    Returns
    -------
    formulations : list[str]
        The type of formulation for each interface.
    preconditioners : list[str]
        The type of operator preconditioner for each interface.
    """

    if isinstance(formulation, str):
        formulations = [formulation] * (n_interfaces - 1)
    elif isinstance(formulation, (list, tuple)):
        formulations = list(formulation)
        for form in formulations:
            if not isinstance(form, str):
                raise ValueError("The formulation name must be a string.")
    else:
        raise ValueError("The formulation must be a string, list or tuple.")

    if isinstance(preconditioner, str):
        preconditioners = [preconditioner] * (n_interfaces - 1)
    elif isinstance(preconditioner, (list, tuple)):
        preconditioners = list(preconditioner)
        for prec in preconditioners:
            if not isinstance(prec, str):
                raise ValueError("The preconditioner name must be a string.")
    else:
        raise ValueError("The preconditioner must be a string, list or tuple.")

    # If necessary, prepend the default formulation and preconditioner for the
    # unbounded exterior surface, on which no integral equation is defined.

    if len(formulations) == n_interfaces - 1:
        formulations = ["none"] + formulations

    if len(preconditioners) == n_interfaces - 1:
        preconditioners = ["none"] + preconditioners

    if len(formulations) != n_interfaces:
        raise ValueError(
            "The number of formulations must be equal to the number of interfaces."
        )

    if len(preconditioners) != n_interfaces:
        raise ValueError(
            "The number of preconditioners must be equal to the number of interfaces."
        )

    # The unbounded exterior surface has no boundary integral equation.
    # Hence, the formulation and preconditioner must be 'none'.
    if formulations[0] != "none":
        raise ValueError(
            "The formulation for the unbounded exterior surface must be 'none'."
        )
    if preconditioners[0] != "none":
        raise ValueError(
            "The preconditioner for the unbounded exterior surface must be 'none'."
        )

    # Check if the specified formulations and preconditioners have been implemented.

    def clean_string_names(name):
        return name.lower().replace("ü", "u")

    formulations = tuple([clean_string_names(form) for form in formulations])
    preconditioners = tuple([clean_string_names(prec) for prec in preconditioners])

    for form in formulations[1:]:
        if form not in ("pmchwt", "muller"):
            raise ValueError(
                "The formulation must be one of: "
                "'pmchwt' or 'muller'."
            )

    for prec in preconditioners[1:]:
        if prec not in ("none", "mass", "osrc"):
            raise ValueError(
                "The preconditioner must be one of: "
                "'none', 'mass', or 'osrc'."
            )

    # Check the consistency of the set of preconditioner and formulation types.

    weak = ("none",)
    strong = ("mass", "osrc")
    weak_preconditioners = [prec in weak for prec in preconditioners[1:]]
    strong_preconditioners = [prec in strong for prec in preconditioners[1:]]
    if not (all(weak_preconditioners) or all(strong_preconditioners)):
        raise NotImplementedError(
            "The preconditioner must be the same weak/strong discretisation type."
        )

    for form, prec in zip(formulations, preconditioners):
        if prec == "osrc" and form not in ("pmchwt",):
            raise ValueError(
                "The OSRC preconditioner only works for the PMCHWT formulation."
            )

    return formulations, preconditioners


class NestedModel(_GraphModel):
    def __init__(
        self,
        topology,
        frequency,
        formulation,
        preconditioner,
        parameters=None,
        label="nested_model",
    ):
        """
        Create a model for nested domains.

        Parameters
        ----------
        topology : optimus.geometry.Graph
            The graph topology representing the geometry.
        frequency : float
            The frequency of the harmonic wave propagation model.
        formulation : list[str]
            The type of formulation for each interface.
        preconditioner : list[str]
            The type of operator preconditioner for each interface.
        parameters : dict, None
            The parameters for the formulation and preconditioner.
        label : str
            The label of the model.
        """

        super().__init__(topology, label)

        self.frequency = frequency
        self.formulation = formulation
        self.preconditioner = preconditioner
        self.parameters = parameters

        self.representation = self._assign_representation(formulation)

        self.spaces = None
        self.continuous_preconditioners = None
        self.discrete_preconditioners = None
        self.continuous_operators = None
        self.discrete_operators = None
        self.source_projections = None
        self.rhs_discrete_system = None
        self.vector_interface_split = None
        self.solution_vector = None
        self.iteration_count = None
        self.timings = {}

        return

    def solve(self, timing=False):
        """
        Solve the nested model.

        Parameters
        ----------
        timing : bool
            Store the computation time of the solution process.
        """
        from optimus import global_parameters
        import time

        global_parameters.bem.update_hmat_parameters("boundary")

        self._create_function_spaces()

        self._create_continuous_preconditioners()

        self._create_continuous_operators()

        if timing:
            start = time.time()
            self._create_discrete_preconditioners()
            self.timings["assembly preconditioners"] = time.time() - start
        else:
            self._create_discrete_preconditioners()

        if timing:
            start = time.time()
            self._create_discrete_operators()
            self.timings["assembly operators"] = time.time() - start
        else:
            self._create_discrete_operators()

        if timing:
            start = time.time()
            self._create_rhs_vector()
            self.timings["assembly rhs"] = time.time() - start
        else:
            self._create_rhs_vector()

        if timing:
            start = time.time()
            self._solve_linear_system()
            self.timings["linear solve"] = time.time() - start
        else:
            self._solve_linear_system()

        self._solution_vector_to_gridfunction()

        return

    @staticmethod
    def _assign_representation(formulations):
        """
        Assign a representation to each interface, according to the formulation.

        Parameters
        ----------
        formulations : list[str]
            The type of boundary integral formulation for each interface.

        Returns
        -------
        representations : list[str]
            The type of representation formula for each interface.
        """

        representations = []
        for form in formulations:
            if form == "none":
                representations.append("none")
            elif form in ("pmchwt", "muller"):
                representations.append("direct")
            else:
                raise ValueError("Unknown formulation: " + form + ".")

        return representations

    def _create_function_spaces(self):
        """
        Create the function spaces for nested domains.

        By default, continuous P1 elements are used.

        Sets "self.spaces" to a list of function spaces, one for each interface.
        """

        from bempp.api import function_space

        self.spaces = []
        for interface in self.topology.interface_nodes:
            if interface.is_active() and interface.bounded:
                self.spaces.append(function_space(interface.geometry.grid, "P", 1))
            else:
                self.spaces.append(None)

        return

    def _create_continuous_preconditioners(self):
        """
        Create the preconditioners for nested domains.

        For each interface, specify the continuous preconditioning
        operators and store all of them in a list that correspond to the same
        interface. Each element in the list is a special object for
        preconditioner operators.

        For the moment, only block-diagonal preconditioners are available.
        That is, a single preconditioner for self-interactions.

        Sets "self.continuous_preconditioners" to a list of PreconditionerOperators.
        """

        self.continuous_preconditioners = []
        for interface in self.topology.interface_nodes:
            if interface.is_active():
                interface_id = interface.identifier
                preconditioner = self.preconditioner[interface_id]

                if preconditioner in ("none", "mass"):
                    # No preconditioner is needed for 'mass' since the strong form
                    # of the operator will be assembled.
                    self.continuous_preconditioners.append(None)

                elif preconditioner == "osrc":
                    self.continuous_preconditioners.append(
                        PreconditionerOperators(
                            identifier=len(self.continuous_preconditioners),
                            node=interface,
                            formulation=self.formulation[interface_id],
                            preconditioner=preconditioner,
                            space=self.spaces[interface_id],
                            materials=(
                                self.topology.subdomain_nodes[
                                    interface.parent_subdomain_id
                                ].material,
                                self.topology.subdomain_nodes[
                                    interface.child_subdomain_id
                                ].material,
                            ),
                            frequency=self.frequency,
                            parameters=self.parameters,
                        )
                    )

                else:
                    raise ValueError("Unknown preconditioner: " + preconditioner + ".")

            else:
                self.continuous_preconditioners.append(None)

        return

    def _create_continuous_operators(self):
        """
        Create the continuous boundary integral operators for nested domains.

        For each interface connector, specify the continuous boundary integral
        operators and store all of them in a list that correspond to the same
        interface connector. Each element in the list is a special object for
        boundary integral operators.

        Sets "self.continuous_operators" to a list of BoundaryIntegralOperators.
        """

        self.continuous_operators = []
        for connector in self.topology.interface_connectors:
            if connector.is_active():
                interface_ids = connector.interfaces_ids
                subdomain_id = connector.subdomain_id

                # The type of boundary integral formulation has to be consistent
                # with the range domain of the operator.
                formulation_name = self.formulation[interface_ids[1]]

                # Left preconditioners use the range domain of the operator.
                preconditioner_name = self.preconditioner[interface_ids[1]]

                spaces = (
                    self.spaces[interface_ids[0]],
                    self.spaces[interface_ids[1]],
                )
                geometries = (
                    self.topology.interface_nodes[interface_ids[0]].geometry,
                    self.topology.interface_nodes[interface_ids[1]].geometry,
                )
                self.continuous_operators.append(
                    BoundaryIntegralOperators(
                        len(self.continuous_operators),
                        connector,
                        formulation_name,
                        preconditioner_name,
                        spaces,
                        self.topology.subdomain_nodes[subdomain_id].material,
                        geometries,
                        self.frequency,
                        self._get_outside_densities(connector),
                    )
                )

            else:
                self.continuous_operators.append(None)

        return

    def _get_outside_densities(self, connector):
        """
        Get the outside densities for a given interface connector.

        Get the density of the medium outside the propagating domain for
        both interfaces.

        Parameters
        ----------
        connector : optimus.geometry.graph_topology.InterfaceConnector
            The interface connector.

        Returns
        -------
        densities : tuple[float]
            The two outside densities.
        """

        topology = connector.topology

        if topology == "self-exterior":
            interface = self.topology.interface_nodes[connector.interfaces_ids[0]]
            interior_domain = self.topology.subdomain_nodes[
                interface.child_subdomain_id
            ]
            density = interior_domain.material.density
            densities = (density, density)

        elif topology == "self-interior":
            interface = self.topology.interface_nodes[connector.interfaces_ids[0]]
            exterior_domain = self.topology.subdomain_nodes[
                interface.parent_subdomain_id
            ]
            density = exterior_domain.material.density
            densities = (density, density)

        elif topology == "parent-child":
            parent_interface_id, child_interface_id = connector.interfaces_ids
            parent_interface = self.topology.interface_nodes[parent_interface_id]
            child_interface = self.topology.interface_nodes[child_interface_id]
            exterior_domain = self.topology.subdomain_nodes[
                parent_interface.parent_subdomain_id
            ]
            interior_domain = self.topology.subdomain_nodes[
                child_interface.child_subdomain_id
            ]
            densities = (
                exterior_domain.material.density,
                interior_domain.material.density,
            )

        elif topology == "sibling":
            sibling_interfaces = [
                self.topology.interface_nodes[node] for node in connector.interfaces_ids
            ]
            sibling_domains = [
                self.topology.subdomain_nodes[interface.child_subdomain_id]
                for interface in sibling_interfaces
            ]
            densities = tuple([domain.material.density for domain in sibling_domains])

        else:
            raise AssertionError(
                "Unknown interface connector topology: " + topology + "."
            )

        return densities

    def _create_discrete_operators(self):
        """
        Assemble the continuous boundary integral operators for nested domains.

        Calculate the discrete boundary integral operators for each interface connector
        and store them in a list that correspond to the same interface connector.

        Sets "self.discrete_operators" to a list of discrete boundary integral
        operators.
        """

        self.discrete_operators = []
        for continuous_operator in self.continuous_operators:
            if continuous_operator is None:
                self.discrete_operators.append(None)
            else:
                self.discrete_operators.append(
                    continuous_operator.assemble_boundary_integral_operators()
                )

        return

    def _create_discrete_preconditioners(self):
        """
        Assemble the continuous preconditioners for nested domains.

        Calculate the discrete preconditioner for each interface node
        and store them in a list that correspond to the same interface.

        Sets "self.discrete_preconditioners" to a list of discrete boundary integral
        operators.
        """

        self.discrete_preconditioners = []
        for continuous_preconditioner in self.continuous_preconditioners:
            if continuous_preconditioner is None:
                self.discrete_preconditioners.append(None)
            else:
                self.discrete_preconditioners.append(
                    continuous_preconditioner.assemble_boundary_integral_operators()
                )

        return

    def _create_rhs_vector(self):
        """
        Assemble the right-hand-side vectors for nested domains.

        At each interface, the projection of the sources towards the interfaces need
        to be calculated. This has to be consistent with the preconditioned formulation.
        For the GMRES linear solver, also assemble the full right-hand-side vector.

        For the moment, it only works for exterior sources.

        Sets "self.source_projections" to a list of SourceProjection objects, for each
        interface node.
        """

        self.source_projections = []
        for interface in self.topology.interface_nodes:
            if interface.is_active() and interface.bounded:
                interface_id = interface.identifier
                parent_subdomain = self.topology.subdomain_nodes[
                    interface.parent_subdomain_id
                ]
                child_subdomain = self.topology.subdomain_nodes[
                    interface.child_subdomain_id
                ]

                if len(parent_subdomain.sources) == 0:
                    self.source_projections.append(
                        EmptySourceProjection(
                            space=self.spaces[interface_id],
                            formulation=self.formulation[interface_id],
                        )
                    )
                elif len(parent_subdomain.sources) == 1:
                    if not parent_subdomain.bounded:
                        prec_ops = self.continuous_preconditioners[interface_id]
                        self.source_projections.append(
                            SourceProjection(
                                source=parent_subdomain.sources[0],
                                space=self.spaces[interface_id],
                                material=parent_subdomain.material,
                                formulation=self.formulation[interface_id],
                                preconditioner_name=self.preconditioner[interface_id],
                                preconditioner_operators=prec_ops,
                            )
                        )
                    else:
                        raise NotImplementedError(
                            "Sources inside bounded subdomains are not implemented yet."
                        )
                else:
                    raise NotImplementedError(
                        "Multiple sources are not implemented yet."
                    )

                if len(child_subdomain.sources) > 0:
                    raise NotImplementedError(
                        "Sources inside bounded subdomains are not implemented yet."
                    )

            else:
                self.source_projections.append(None)

        rhs_vectors = []
        rhs_vector_sizes = []
        for source_projection in self.source_projections:
            if source_projection is None:
                rhs_vector_sizes.append(0)
            else:
                source_projection.assemble_rhs_vector()
                rhs_vectors.append(source_projection.rhs_vector)
                rhs_vector_sizes.append(source_projection.rhs_vector.size)

        self.rhs_discrete_system = _np.concatenate(rhs_vectors)
        self.vector_interface_split = _np.cumsum(rhs_vector_sizes)[:-1]

        return

    def _solve_linear_system(self):
        """
        Solve the linear system for nested domains.

        The system matrix, preconditioner, and right-hand-side vector should
        already have been assembled, separately for each interface node and
        connector.

        Sets "self.solution_vector" to a list of solution vectors, one for
        each interface.
        """

        from .linalg import linear_solve
        from scipy.sparse.linalg import LinearOperator

        # noinspection PyArgumentList
        self.lhs_linear_operator = LinearOperator(
            shape=(self.rhs_discrete_system.size, self.rhs_discrete_system.size),
            matvec=self._nested_matvec,
            dtype=_np.complex128,
        )

        self.solution_vector, self.iteration_count = linear_solve(
            self.lhs_linear_operator,
            self.rhs_discrete_system,
            return_iteration_count=True,
        )

        return

    def _nested_matvec(self, vector):
        """
        Matrix-vector product for the linear system of nested domains.

        Parameters
        ----------
        vector : numpy.ndarray
            The vector to multiply.

        Returns
        -------
        numpy.ndarray
            The product of the preconditioned system matrix with the vector.
        """

        # The vector split has always arrays, also for inactive interfaces, in
        # which case the length is zero.
        rhs_vector_interface = _np.split(vector, self.vector_interface_split)
        result_interface = [_np.zeros_like(vec) for vec in rhs_vector_interface]

        for interface in self.topology.interface_nodes:
            if interface.is_active() and interface.bounded:
                interface_id = interface.identifier

                rhs_vector = rhs_vector_interface[interface_id]

                # Perform the matrix-vector product corresponding to the
                # column of the system matrix for each interface.
                column_matvec_vectors = self._interface_matvec(interface_id, rhs_vector)

                # Add the results of the column matrix-vector products for
                # each interface to the result vector.
                for row, matvec_vector in enumerate(column_matvec_vectors):
                    if matvec_vector is not None:
                        result_interface[row] += matvec_vector

        # Perform the preconditioning step for each interface.
        # This should be done after the system matvec is finished, since
        # that is a column-based operation.
        for interface in self.topology.interface_nodes:
            if interface.is_active() and interface.bounded:
                interface_id = interface.identifier
                preconditioner_operator = self.continuous_preconditioners[interface_id]
                if preconditioner_operator is not None:
                    result_interface[interface_id] = preconditioner_operator.matvec(
                        result_interface[interface_id]
                    )

        result = _np.concatenate(result_interface)

        return result

    def _interface_matvec(self, interface_id, vector):
        """
        Matrix-vector product for the linear system of a single interface node.

        The vector is part of the right-hand-side vector of the linear system,
        corresponding to the specific interface node given by the identifier.
        The matrix-vector multiplication consists of all parts of the full
        system matvec that have this interface as source. In other words, it
        corresponds to the column of the system matrix corresponding to the
        interface. Hence, the output is a list of vectors for each interface node.

        Parameters
        ----------
        interface_id : int
            The identifier of the interface node.
        vector : numpy.ndarray
            The vector to multiply.

        Returns
        -------
        result : list[None, numpy.ndarray]
            The product of the system matrix column with the vector for each interface.
        """

        result = [None] * self.topology.number_interface_nodes()

        for operator in self.continuous_operators:
            interface_ids = operator.connector.interfaces_ids

            if interface_id in interface_ids:
                local_matvec = operator.matvec(interface_id, vector)

                other_interface_id = interface_ids[
                    1 - interface_ids.index(interface_id)
                ]

                if result[other_interface_id] is None:
                    result[other_interface_id] = local_matvec

                else:
                    result[other_interface_id] += local_matvec

                    if interface_id != other_interface_id:
                        raise AssertionError(
                            "Summing interface matvecs twice should only happen "
                            "for self interactions."
                        )

        return result

    def _solution_vector_to_gridfunction(self):
        """
        Convert the solution vector to separate grid functions.

        Sets "self.solution" to a list of solutions, each element corresponding
        to an interface. If inactive, the solution is None. Otherwise, it is a
        list of grid functions, one for each space.
        """
        from .common import _vector_to_gridfunction

        solution_vector_interfaces = _np.split(
            self.solution_vector, self.vector_interface_split
        )

        self.solution = []
        for vector, space, formulation in zip(
            solution_vector_interfaces, self.spaces, self.formulation
        ):
            if space is None:
                self.solution.append(None)
            else:
                if formulation in ("pmchwt", "muller"):
                    self.solution.append(
                        _vector_to_gridfunction(vector, [space, space])
                    )
                else:
                    raise ValueError("Unknown formulation: " + formulation + ".")

        return


class BoundaryIntegralOperators:
    def __init__(
        self,
        identifier,
        connector,
        formulation,
        preconditioner,
        spaces,
        material,
        geometries,
        frequency,
        densities,
    ):
        """
        Create the boundary integral operators for an interface connector.

        Parameters
        ----------
        identifier : int
            The unique identifier of the boundary integral operator in the model.
        connector : optimus.geometry.graph_topology.InterfaceConnector
            The connector between two interfaces.
        formulation : str
            The type of boundary integral formulation.
        preconditioner : str
            The type of preconditioner.
        spaces : tuple[bempp.api.FunctionSpace]
            The function spaces of the two interfaces.
        material : optimus.material.Material
            The material of the propagating medium.
        geometries : tuple[optimus.geometry.Geometry]
            The geometries of the two interfaces.
        frequency : float
            The frequency of the propagating wave.
        densities : tuple[float]
            The densities of the media outside the interface connector.
            That is, the media at the other side of the two interfaces,
            not the propagating medium. The order is the same as the order
            of the interface nodes.
        """

        self.identifier = identifier
        self.connector = connector
        self.formulation = self._check_formulation(formulation)
        self.preconditioner = preconditioner
        self.spaces = spaces
        self.material = material
        self.geometries = geometries
        self.frequency = frequency
        self.densities = densities

        self.wavenumber = self.material.compute_wavenumber(self.frequency)

        self.continuous_operators = self.create_connector_operators()
        self.discrete_operators = None

        return

    @staticmethod
    def _check_formulation(formulation):
        """
        Check the validity of the boundary integral formulation.

        Parameters
        ----------
        formulation : str
            The type of boundary integral formulation.

        Returns
        -------
        formulation : str
            The type of boundary integral formulation.
        """

        if formulation not in ("pmchwt", "muller"):
            raise NotImplementedError(
                "Formulation not implemented: " + formulation + "."
            )

        return formulation

    def create_connector_operators(self):
        """
        Create the boundary integral operators for an interface connector.

        Returns
        -------
        operators : tuple[dict[str, bempp.api.operators.boundary]]
            The continuous boundary integral operators. The tuple contains
            two items, one for each direction of the connector. Each item
            contains a dictionary with the necessary boundary integral
            operators.
        """

        from .acoustics import create_boundary_integral_operators

        if self.formulation in ("pmchwt", "muller"):
            if self.connector.topology in ("self-exterior", "self-interior"):
                operators_start_end = create_boundary_integral_operators(
                    self.spaces[0],
                    self.spaces[1],
                    self.wavenumber,
                    identity=True,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
                operators_end_start = operators_start_end
            else:
                operators_start_end = create_boundary_integral_operators(
                    self.spaces[0],
                    self.spaces[1],
                    self.wavenumber,
                    identity=False,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
                operators_end_start = create_boundary_integral_operators(
                    self.spaces[1],
                    self.spaces[0],
                    self.wavenumber,
                    identity=False,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
        else:
            raise ValueError("Unknown formulation: " + self.formulation + ".")

        operators = (operators_start_end, operators_end_start)

        return operators

    def assemble_boundary_integral_operators(self):
        """
        Assemble the boundary integral operators for the interface connector.

        Sets "self.discrete_operators" to a tuple of dictionaries,
        one for each direction.

        Returns
        -------
        discrete_operators : tuple[dict[str, bempp.api.operators.boundary]]
            The discrete boundary integral operators. The tuple is for each direction.
        """

        def assemble(continuous_operators):
            """Assemble a set of boundary integral operators."""

            discrete_operators = {}
            if self.preconditioner == "none":
                for key, operator in continuous_operators.items():
                    discrete_operators[key] = operator.weak_form()
            elif self.preconditioner in ("mass", "osrc"):
                for key, operator in continuous_operators.items():
                    discrete_operators[key] = operator.strong_form()
            else:
                raise ValueError("Unknown preconditioner: " + self.preconditioner + ".")

            return discrete_operators

        assembled_operators = (
            assemble(self.continuous_operators[0]),
            assemble(self.continuous_operators[1]),
        )

        self.discrete_operators = assembled_operators

        return assembled_operators

    def matvec(self, interface_id, vector):
        """
        Matrix-vector product for a single one-way interface connector.

        The identifier indicates the source interface, on which the vector is defined.
        The vector is the entire vector for the connector, and may include
        multiple traces depending on the formulation.

        Parameters
        ----------
        interface_id : int
            The identifier of the source interface.
        vector : numpy.ndarray
            The vector to multiply.

        Returns
        -------
        result : numpy.ndarray
            The product of the interface connector with the vector.
        """

        interface_ids = self.connector.interfaces_ids

        if interface_id not in interface_ids:
            raise AssertionError(
                "The interface " + str(interface_id) + " is not part of the connector."
            )
        else:
            index = interface_ids.index(interface_id)

        if self.discrete_operators is None:
            _ = self.assemble_boundary_integral_operators()

        if self.formulation == "pmchwt":
            result = self._apply_pmchwt_operators(index, vector)
        elif self.formulation == "muller":
            result = self._apply_muller_operators(index, vector)
        else:
            raise ValueError("Unknown formulation: " + self.formulation + ".")

        return result

    def _apply_pmchwt_operators(self, index, vector):
        """
        Apply the PMCHWT operators to a vector to perform a local matvec.

        The index indicates the source/domain interface of the connector, on which
        the vector is defined. The vector is the entire vector for the connector,
        which includes the Dirichlet and Neumann traces.

        Parameters
        ----------
        index : int
            The index of the pair of discrete operators to use: either 0 or 1.
        vector : numpy.ndarray
            The vector to multiply.

        Returns
        -------
        result : numpy.ndarray
            The product of the boundary integral operators with the vector.
        """

        operators = self.discrete_operators[index]
        connector_topology = self.connector.topology

        trace_vectors = _np.split(vector, [self.spaces[index].global_dof_count])

        if connector_topology in ("self-exterior", "sibling"):
            matvec_vectors = self._apply_calderon(operators, trace_vectors)

        elif connector_topology == "self-interior":
            rho_ext = self.densities[0]
            rho_int = self.material.density
            scaled_vector = [
                trace_vectors[0],
                (rho_int / rho_ext) * trace_vectors[1],
            ]
            calderon_vectors = self._apply_calderon(operators, scaled_vector)
            matvec_vectors = [
                calderon_vectors[0],
                (rho_ext / rho_int) * calderon_vectors[1],
            ]

        elif connector_topology == "parent-child":
            if index == 0:
                rho_ext = self.densities[0]
                rho_int = self.material.density
                scaled_vector = [
                    -trace_vectors[0],
                    -(rho_int / rho_ext) * trace_vectors[1],
                ]
                matvec_vectors = self._apply_calderon(operators, scaled_vector)
            elif index == 1:
                rho_ext = self.densities[0]
                rho_int = self.material.density
                calderon_vectors = self._apply_calderon(operators, trace_vectors)
                matvec_vectors = [
                    -calderon_vectors[0],
                    -(rho_ext / rho_int) * calderon_vectors[1],
                ]
            else:
                raise AssertionError(
                    "Index must be 0 or 1, not: " + str(index) + "."
                )

        else:
            raise AssertionError(
                "Unknown topology of interface connector: " + connector_topology
            )

        result = _np.concatenate(matvec_vectors)

        return result

    def _apply_muller_operators(self, index, vector):
        """
        Apply the Müller operators to a vector to perform a local matvec.

        The index indicates the source/domain interface of the connector, on which
        the vector is defined. The vector is the entire vector for the connector,
        which includes the Dirichlet and Neumann traces.

        Parameters
        ----------
        index : int
            The index of the pair of discrete operators to use: either 0 or 1.
        vector : numpy.ndarray
            The vector to multiply.

        Returns
        -------
        result : numpy.ndarray
            The product of the boundary integral operators with the vector.
        """

        operators = self.discrete_operators[index]
        connector_topology = self.connector.topology

        trace_vectors = _np.split(vector, [self.spaces[index].global_dof_count])

        if connector_topology == "self-exterior":
            identity_vectors = self._apply_identity(operators, trace_vectors)
            calderon_vectors = self._apply_calderon(operators, trace_vectors)
            matvec_vectors = [
                identity_vectors[0] + calderon_vectors[0],
                identity_vectors[1] + calderon_vectors[1],
            ]

        elif connector_topology == "self-interior":
            rho_ext = self.densities[0]
            rho_int = self.material.density
            scaled_vector = [
                trace_vectors[0],
                (rho_int / rho_ext) * trace_vectors[1],
            ]
            calderon_vectors = self._apply_calderon(operators, scaled_vector)
            matvec_vectors = [
                -calderon_vectors[0],
                -(rho_ext / rho_int) * calderon_vectors[1],
            ]

        elif connector_topology == "sibling":
            matvec_vectors = self._apply_calderon(operators, trace_vectors)

        elif connector_topology == "parent-child":
            if index == 0:
                rho_ext = self.densities[0]
                rho_int = self.material.density
                scaled_vector = [
                    -trace_vectors[0],
                    -(rho_int / rho_ext) * trace_vectors[1],
                ]
                matvec_vectors = self._apply_calderon(operators, scaled_vector)
            elif index == 1:
                rho_ext = self.densities[0]
                rho_int = self.material.density
                calderon_vectors = self._apply_calderon(operators, trace_vectors)
                matvec_vectors = [
                    calderon_vectors[0],
                    (rho_ext / rho_int) * calderon_vectors[1],
                ]
            else:
                raise AssertionError(
                    "Index must be 0 or 1, not: " + str(index) + "."
                )

        else:
            raise AssertionError(
                "Unknown topology of interface connector: " + connector_topology
            )

        result = _np.concatenate(matvec_vectors)

        return result

    @staticmethod
    def _apply_identity(operators, vector):
        """
        Apply the identity operator to a vector.

        Parameters
        ----------
        operators : dict[str, bempp.api.operators.boundary.sparse]
            The dictionary with discrete boundary integral operators.
        vector : list[numpy.ndarray]
            The two parts of the vector to multiply: Dirichlet and Neumann traces.

        Returns
        -------
        result : list[numpy.ndarray]
            The two parts of the result vector: Dirichlet and Neumann traces.
        """

        dirichlet_matvec = operators["identity"] * vector[0]
        neumann_matvec = operators["identity"] * vector[1]

        return [dirichlet_matvec, neumann_matvec]

    @staticmethod
    def _apply_calderon(operators, vector):
        """
        Apply the Calderón operator to a vector.

        Parameters
        ----------
        operators : dict[str, bempp.api.operators.boundary.Helmholtz]
            The dictionary with discrete boundary integral operators.
        vector : list[numpy.ndarray]
            The two parts of the vector to multiply: Dirichlet and Neumann traces.

        Returns
        -------
        result : list[numpy.ndarray]
            The two parts of the result vector: Dirichlet and Neumann traces.
        """

        dirichlet_matvec = (
            -operators["double_layer"] * vector[0]
            + operators["single_layer"] * vector[1]
        )
        neumann_matvec = (
            operators["hypersingular"] * vector[0]
            + operators["adjoint_double_layer"] * vector[1]
        )

        return [dirichlet_matvec, neumann_matvec]


class PreconditionerOperators:
    def __init__(
        self,
        identifier,
        node,
        formulation,
        preconditioner,
        space,
        materials,
        frequency,
        parameters,
    ):
        """
        Create the preconditioner operators for an interface node.

        Parameters
        ----------
        identifier : int
            The unique identifier of the preconditioner in the model.
        node : optimus.geometry.graph_topology.InterfaceNode
            The interface node.
        formulation : str
            The type of boundary integral formulation.
        preconditioner : str
            The type of preconditioner.
        space : bempp.api.FunctionSpace
            The function space at the interface.
        materials : tuple[optimus.material.Material]
            The materials exterior and interior to the interface.
        frequency : float
            The frequency of the propagating wave.
        parameters : dict
            The parameters for the preconditioner.
        """

        self.identifier = identifier
        self.node = node
        self.formulation = formulation
        self.preconditioner = preconditioner
        self.space = space
        self.materials = materials
        self.frequency = frequency
        self.parameters = parameters

        self.continuous_operators = self.create_interface_operators()
        self.discrete_operators = None

        return

    def create_interface_operators(self):
        """
        Create the preconditioner operators for an interface node.

        Returns
        -------
        operators : dict[str, bempp.api.operators.boundary.Helmholtz]
            The continuous preconditioner operators with their names.
        """

        if self.preconditioner == "osrc" and self.formulation == "pmchwt":
            operators = self.create_osrc_operators(dtn=True, ntd=True)
        else:
            raise NotImplementedError

        return operators

    def create_osrc_operators(self, dtn, ntd):
        """
        Create the OSRC preconditioner operators for an interface node.

        Parameters
        ----------
        dtn : bool
            Whether to create the DtN operator.
        ntd : bool
            Whether to create the NtD operator.

        Returns
        -------
        operators : dict[str, bempp.api.operators.boundary.Helmholtz]
            The continuous OSRC preconditioner operators with
            their names: 'NtD' or 'DtN'.
        """

        from .common import _process_osrc_parameters
        from .acoustics import create_osrc_operators
        from optimus import global_parameters

        osrc_params = _process_osrc_parameters(self.parameters)

        osrc_wavenumber_type = global_parameters.preconditioning.osrc.wavenumber
        if osrc_wavenumber_type == "ext":
            osrc_wavenumber = self.materials[0].compute_wavenumber(self.frequency)
        elif osrc_wavenumber_type == "int":
            osrc_wavenumber = self.materials[1].compute_wavenumber(self.frequency)
        else:
            raise ValueError(
                "Unknown wavenumber type for OSRC preconditioner: "
                + osrc_wavenumber_type
                + "."
            )

        osrc_operators = create_osrc_operators(
            space=self.space,
            wavenumber=osrc_wavenumber,
            parameters=osrc_params,
            dtn=dtn,
            ntd=ntd,
        )

        operators = {}
        if dtn:
            operators["DtN"] = osrc_operators.pop(0)
        if ntd:
            operators["NtD"] = osrc_operators[0]

        return operators

    def assemble_boundary_integral_operators(self):
        """
        Assemble the boundary integral operators for the interface node.

        Sets "self.discrete_operators" to a dictionary with assembled operators.

        Returns
        -------
        discrete_operators : dict[str, bempp.api.operators.boundary.Helmholtz]
            The discrete boundary integral operators of the preconditioner.
        """

        assembled_operators = {}

        if self.preconditioner == "osrc":
            for key, operator in self.continuous_operators.items():
                assembled_operators[key] = operator.strong_form()
        else:
            raise ValueError("Unknown preconditioner: " + self.preconditioner + ".")

        self.discrete_operators = assembled_operators

        return assembled_operators

    def matvec(self, vector):
        """
        Matrix-vector product for a interface node.

        The identifier indicates the interface, on which the vector is defined.
        The vector is the entire vector for the interface, and may include
        multiple traces depending on the formulation.
        Only self-interactions are implemented for a preconditioner.

        Parameters
        ----------
        vector : numpy.ndarray
            The vector to multiply.

        Returns
        -------
        result : numpy.ndarray
            The product of the preconditioner with the vector.
        """

        if self.discrete_operators is None:
            _ = self.assemble_boundary_integral_operators()

        if self.preconditioner == "osrc":
            result = self._apply_osrc_operators(vector)
        else:
            raise ValueError("Unknown preconditioner: " + self.preconditioner + ".")

        return result

    def _apply_osrc_operators(self, vector):
        """
        Apply the OSRC preconditioner operators to a vector.

        The matrix-vector product depends on the specific formulation.
        The vector is the entire vector for the interface, possibly
        with multiple traces.

        Parameters
        ----------
        vector : numpy.ndarray
            The vector to multiply.

        Returns
        -------
        result : numpy.ndarray
            The product of the preconditioner with the vector.
        """

        if self.formulation == "pmchwt":
            trace_dir, trace_neu = _np.split(vector, [self.space.global_dof_count])
            result_dir = self.discrete_operators["NtD"] * trace_neu
            result_neu = self.discrete_operators["DtN"] * trace_dir
            result = _np.concatenate([result_dir, result_neu])
        else:
            raise ValueError("Unknown formulation: " + self.formulation + ".")

        return result


class _SourceInterface:
    def __init__(self, source, space):
        """
        Base class for sources at interfaces.

        Parameters
        ----------
        source : optimus.source.Source, None
            The source of the wave propagation model.
        space : bempp.api.FunctionSpace
            The function space of the interface.
        """

        self.source = source
        self.space = space
        self.rhs_vector = None

        return

    def assemble_rhs_vector(self):
        """Assemble the right-hand-side vector of the discrete system."""
        raise NotImplementedError


class SourceProjection(_SourceInterface):
    def __init__(
        self,
        source,
        space,
        material,
        formulation,
        preconditioner_name,
        preconditioner_operators=None,
    ):
        """
        Create a source projection object.

        Each object stores the information of the sources projected to an interface.
        For the moment, we assume a single source from the exterior domain, but
        this can be extended to an arbitrary number of sources and from different
        subdomains.

        The right-hand-side vector of the discrete system will be assembled according
        to the preconditioned formulation.

        Parameters
        ----------
        source : optimus.source.Source
            The source of the wave propagation model.
        space : bempp.api.FunctionSpace
            The function space of the interface,
            which contains the surface mesh.
        material : optimus.material.Material
            The material of the propagating medium.
        formulation : str
            The type of boundary integral formulation.
        preconditioner_name : str
            The type of preconditioner.
        preconditioner_operators : optimus.model.nested.PreconditionerOperators, None
            The preconditioner object, if it exists.
        """

        super().__init__(source, space)

        self.material = material
        self.formulation = formulation
        self.preconditioner_name = preconditioner_name
        self.preconditioner_operators = preconditioner_operators

        self.traces = None
        self.trace_vectors = None

        return

    def assemble_rhs_vector(self):
        """
        Assemble the right-hand-side vector corresponding to the source projection.

        Sets "self.traces" to a tuple of the necessary traces of the formulations,
        which are of type "bempp.api.GridFunction".

        Sets "self.trace_vectors" to a tuple of the necessary discrete traces of the
        formulations, which are of type "numpy.ndarray" and may be either the
        coefficients or the projections of the grid function, depending on
        the preconditioner.

        Sets "self.rhs_vector" to the assembled vector, corresponding to the
        preconditioned formulation. Its type is "numpy.ndarray".
        """

        if self.formulation in ("pmchwt", "muller"):
            trace_dirichlet, trace_neumann = self.source.calc_surface_traces(
                medium=self.material,
                space_dirichlet=self.space,
                space_neumann=self.space,
                dirichlet_trace=True,
                neumann_trace=True,
            )
            self.traces = {
                "dirichlet": trace_dirichlet,
                "neumann": trace_neumann,
            }

            if self.preconditioner_name == "none":
                self.trace_vectors = (
                    trace_dirichlet.projections(),
                    trace_neumann.projections(),
                )
            elif self.preconditioner_name in ("mass", "osrc"):
                self.trace_vectors = (
                    trace_dirichlet.coefficients,
                    trace_neumann.coefficients,
                )
            else:
                raise ValueError(
                    "Unknown preconditioner: " + self.preconditioner_name + "."
                )

            source_vector = _np.concatenate(self.trace_vectors)

            if self.preconditioner_operators is None:
                self.rhs_vector = source_vector
            else:
                self.rhs_vector = self.preconditioner_operators.matvec(source_vector)

        else:
            raise ValueError("Unknown formulation: " + self.formulation + ".")

        return


class EmptySourceProjection(_SourceInterface):
    def __init__(
        self,
        space,
        formulation,
    ):
        """
        Create an empty source projection object.

        When no source is present, the right-hand-side vector of the discrete system
        is a zero vector, with the size depending on the function space and
        boundary integral formulation.

        Parameters
        ----------
        space : bempp.api.FunctionSpace
            The function space of the interface.
        formulation : str
            The type of boundary integral formulation.
        """

        super().__init__(None, space)

        self.formulation = formulation

        self.trace_vectors = None

        return

    def assemble_rhs_vector(self):
        """
        Assemble the right-hand-side vector corresponding to the source projection.

        Sets "self.rhs_vector" to the assembled vector, corresponding to the
        boundary integral formulation.
        """

        if self.formulation == "pmchwt":
            self.trace_vectors = (
                _np.zeros(self.space.global_dof_count, dtype=_np.complex128),
                _np.zeros(self.space.global_dof_count, dtype=_np.complex128),
            )
            self.rhs_vector = _np.concatenate(self.trace_vectors)
        else:
            raise ValueError("Unknown formulation: " + self.formulation + ".")

        return
