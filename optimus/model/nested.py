"""Nested models."""

import numpy as _np
from .common import GraphModel as _GraphModel


def create_default_nested_model(topology, frequency, label="default"):
    """
    Create an acoustics model on nested domains with default settings.

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

    model = NestedModel(
        topology=topology,
        frequency=frequency,
        formulation=["none"] + ["pmchwt"] * (topology.number_interface_nodes() - 1),
        preconditioner=["none"] + ["mass"] * (topology.number_interface_nodes() - 1),
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
    """
    Create an acoustics model on nested domains.

    A boundary integral formulation will be created with specified parameters, for
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

    model = NestedModel(
        topology=topology,
        frequency=frequency,
        formulation=formulation,
        preconditioner=preconditioner,
        parameters=parameters,
        label=label,
    )

    return model


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

        from .formulations import (check_validity_nested_formulation, check_sources,
                                   assign_representation)

        super().__init__(topology, label)

        self.frequency = check_sources(frequency, topology.subdomain_nodes)
        self.formulation, self.preconditioner = check_validity_nested_formulation(
            formulation,
            preconditioner,
            topology.number_interface_nodes(),
        )
        self.representation = assign_representation(self.formulation)
        self.parameters = parameters

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

        self._create_continuous_operators()

        self._create_continuous_preconditioners()

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
                self.spaces.append(
                    function_space(interface.geometry.grid, "P", 1)
                )
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

                elif preconditioner in ("osrc", "calderon"):

                    if preconditioner == "calderon":
                        calderon_operators = self._get_calderon_operators(interface_id)
                    else:
                        calderon_operators = None

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
                            calderon_operators=calderon_operators,
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

                # The formulation type at both ends of the connector need to be
                # passed on.
                formulation_names = (
                    self.formulation[interface_ids[0]],
                    self.formulation[interface_ids[1]],
                )

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
                        identifier=len(self.continuous_operators),
                        connector=connector,
                        formulations=formulation_names,
                        preconditioner=preconditioner_name,
                        spaces=spaces,
                        material=self.topology.subdomain_nodes[subdomain_id].material,
                        geometries=geometries,
                        frequency=self.frequency,
                        densities=self._get_outside_densities(connector),
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

    def _get_calderon_operators(self, interface_id):
        """
        Get the Calderón operators for a given interface.

        Parameters
        ----------
        interface_id : int
            The identifier of the interface.

        Returns
        -------
        operators : tuple[bempp.api.assembly.boundary_operator.BoundaryOperator]
            The Calderón operators for the exterior and interior side of the interface.
        """

        if self.continuous_operators is None:
            raise AssertionError(
                "Continuous operators must have been created to obtain the "
                "Calderón operators."
            )

        interface_connector_self_exterior = (
            self.topology.find_interface_connectors_of_interface(
                interface_id, "self-exterior"
            )
        )
        if len(interface_connector_self_exterior) != 1:
            raise AssertionError(
                "There must be exactly one self-exterior interface connector."
            )
        else:
            interface_connector_id_exterior \
                = interface_connector_self_exterior[0].identifier

        interface_connector_self_interior = (
            self.topology.find_interface_connectors_of_interface(
                interface_id, "self-interior"
            )
        )
        if len(interface_connector_self_interior) != 1:
            raise AssertionError(
                "There must be exactly one self-interior interface connector."
            )
        else:
            interface_connector_id_interior \
                = interface_connector_self_interior[0].identifier

        calderon_operators = (
            self.continuous_operators[interface_connector_id_exterior],
            self.continuous_operators[interface_connector_id_interior]
        )

        return calderon_operators

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

        Sets object attributes:
         - self.source_projections: a list of SourceProjection objects, for each
            interface node.
         - self.rhs_discrete_system: an array of the entire right-hand-side vector.
         - self.vector_interface_split: the indices of the vector on which to split
            according to the size of the formulation for each interface.
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

        Sets "self.solution_vector" to an array of the solution vector, where
        its size is the aggregate of the sizes of the traces on each interface.
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
                elif formulation == "multitrace":
                    self.solution.append(
                        _vector_to_gridfunction(vector, [space, space, space, space])
                    )
                else:
                    raise ValueError("Unknown formulation: " + formulation + ".")

        return


class BoundaryIntegralOperators:
    def __init__(
        self,
        identifier,
        connector,
        formulations,
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
        formulations : tuple[str]
            The type of boundary integral formulation at the two interfaces.
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
        self.formulations = formulations
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

    def create_connector_operators(self):
        """
        Create the boundary integral operators for an interface connector.

        Returns
        -------
        operators : tuple[dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]]
            The continuous boundary integral operators. The tuple contains
            two items, one for each direction of the connector. Each item
            contains a dictionary with the necessary boundary integral
            operators.
        """

        from .acoustics import create_boundary_integral_operators

        if self.connector.topology in ("self-exterior", "self-interior"):
            if self.formulations[0] != self.formulations[1]:
                raise AssertionError(
                    "The two interfaces of the self-connector must have the "
                    "same formulation."
                )
            if self.formulations[0] in ("pmchwt", "muller", "multitrace"):
                # The PMCHWT, Müller, and multitrace formulations need all four
                # boundary integral operators, plus the identity operator
                # for the self-interactions.
                operators_start_end = create_boundary_integral_operators(
                    space_domain=self.spaces[0],
                    space_range=self.spaces[1],
                    wavenumber=self.wavenumber,
                    identity=True,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
                # The self interactions are symmetric, so the same operators.
                operators_end_start = operators_start_end
            else:
                raise AssertionError("Unknown formulation: " + self.formulations[0])

        elif self.connector.topology in ("sibling", "parent-child"):
            # The formulation type relates to the range space of the operator.
            if self.formulations[1] in ("pmchwt", "muller", "multitrace"):
                # The PMCHWT, Müller, and multitrace formulations need all four
                # boundary integral operators.
                operators_start_end = create_boundary_integral_operators(
                    space_domain=self.spaces[0],
                    space_range=self.spaces[1],
                    wavenumber=self.wavenumber,
                    identity=False,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
            else:
                raise AssertionError("Unknown formulation: " + self.formulations[0])
            if self.formulations[0] in ("pmchwt", "muller", "multitrace"):
                # The PMCHWT, Müller, and multitrace formulations need all four
                # boundary integral operators.
                operators_end_start = create_boundary_integral_operators(
                    space_domain=self.spaces[1],
                    space_range=self.spaces[0],
                    wavenumber=self.wavenumber,
                    identity=False,
                    single_layer=True,
                    double_layer=True,
                    adjoint_double_layer=True,
                    hypersingular=True,
                )
            else:
                raise AssertionError("Unknown formulation: " + self.formulations[0])

        else:
            raise AssertionError(
                "Unknown connector topology: " + self.connector.topology
            )

        operators = (operators_start_end, operators_end_start)

        return operators

    def assemble_boundary_integral_operators(self):
        """
        Assemble the boundary integral operators for the interface connector.

        Sets "self.discrete_operators" to a tuple of dictionaries,
        one for each direction.

        Returns
        -------
        operators : tuple[dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]]
            The discrete boundary integral operators. The tuple is for each direction.
        """

        def assemble(continuous_operators_oneway):
            """Assemble a set of boundary integral operators."""

            discrete_operators = {}
            if self.preconditioner == "none":
                for key, operator in continuous_operators_oneway.items():
                    discrete_operators[key] = operator.weak_form()
            elif self.preconditioner in ("mass", "osrc", "calderon"):
                for key, operator in continuous_operators_oneway.items():
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
        The vector is the entire vector for the connector, whose size depends on
        the number of traces in the specific boundary integral formulation.

        The matvec operation depends on the boundary integral formulation
        at the range interface, while the vector depends on the formulation
        at the source interface.

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

        if self.discrete_operators is None:
            _ = self.assemble_boundary_integral_operators()

        interface_ids = self.connector.interfaces_ids

        # The local index is the position of the source interface in the connector:
        #  - 0: the source interface is the start interface of the connector
        #  - 1: the source interface is the end interface of the connector
        if interface_id not in interface_ids:
            raise AssertionError(
                "The interface " + str(interface_id) + " is not part of the connector."
            )
        else:
            source_index = interface_ids.index(interface_id)
            range_index = 1 - source_index

        # Split the source vector according to the formulation at the source interface.
        trace_vectors = self._split_vector(vector, source_index)

        # Use the matvec according to the formulation of the range interface.
        range_formulation = self.formulations[range_index]
        # The boundary operators map from the source to range, so take the source index.
        boundary_operators = self.discrete_operators[source_index]
        if range_formulation == "pmchwt":
            result = self._apply_pmchwt_operators(
                boundary_operators, trace_vectors, source_index
            )
        elif range_formulation == "muller":
            result = self._apply_muller_operators(
                boundary_operators, trace_vectors, source_index
            )
        elif range_formulation == "multitrace":
            result = self._apply_multitrace_operators(
                boundary_operators, trace_vectors, source_index
            )
        else:
            raise AssertionError("Unknown formulation: " + range_formulation)

        return result

    def _split_vector(self, vector, local_index):
        """
        Split the vector according to the boundary integral formulation.

        The direct single-trace formulations have the exterior Dirichlet and
        Neumann traces as surface potentials.
        The multitrace formulation has the exterior and interior Dirichlet and
        Neumann traces as surface potentials.

        Parameters
        ----------
        vector : numpy.ndarray
            The vector to split.
        local_index : int
            The index of the local orientation (0,1).

        Returns
        -------
        traces : dict[str, numpy.ndarray]
            The vector split into trace components.
            Possible keys are: "dirichlet-exterior", "neumann-exterior",
            "dirichlet-interior", "neumann-interior".
        """

        formulation = self.formulations[local_index]
        ndof = self.spaces[local_index].global_dof_count

        if formulation in ("pmchwt", "muller"):
            trace_vectors = _np.split(vector, indices_or_sections=[ndof])
            traces = {
                "dirichlet-exterior": trace_vectors[0],
                "neumann-exterior": trace_vectors[1],
            }
        elif formulation == "multitrace":
            trace_vectors = _np.split(
                vector, indices_or_sections=_np.cumsum([ndof] * 3)
            )
            traces = {
                "dirichlet-exterior": trace_vectors[0],
                "neumann-exterior": trace_vectors[1],
                "dirichlet-interior": trace_vectors[2],
                "neumann-interior": trace_vectors[3],
            }
        else:
            raise AssertionError(
                "Unknown formulation: " + formulation
            )

        return traces

    def _apply_pmchwt_operators(self, operators, vectors, source_index):
        """
        Apply the PMCHWT operators to a vector to perform a local matvec.

        The operators have to map from the source to range interface and provided
        in a dictionary with the names of the operators.
        The vectors are on the source interface and are provided in a dictionary
        with the names of the traces.

        Parameters
        ----------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The boundary integral operators that map from the source to range interface.
        vectors : dict[str, numpy.ndarray]
            The trace vectors to multiply.
        source_index : int
            The index of the source interface in the connector, either 0 or 1.

        Returns
        -------
        result : numpy.ndarray
            The product of the boundary integral operators with the vector.
        """

        trace_vectors = (
            vectors["dirichlet-exterior"],
            vectors["neumann-exterior"],
        )

        connector_topology = self.connector.topology

        if connector_topology in ("self-exterior", "sibling"):
            matvec_vectors = self._apply_calderon(operators, trace_vectors)

        elif connector_topology == "self-interior":
            rho_ext = self.densities[0]
            rho_int = self.material.density
            scaled_vector = (
                trace_vectors[0],
                (rho_int / rho_ext) * trace_vectors[1],
            )
            calderon_vectors = self._apply_calderon(operators, scaled_vector)
            matvec_vectors = (
                calderon_vectors[0],
                (rho_ext / rho_int) * calderon_vectors[1],
            )

        elif connector_topology == "parent-child":
            if source_index == 0:
                # from parent to child
                rho_ext = self.densities[0]
                rho_int = self.material.density
                scaled_vector = (
                    -trace_vectors[0],
                    -(rho_int / rho_ext) * trace_vectors[1],
                )
                matvec_vectors = self._apply_calderon(operators, scaled_vector)
            elif source_index == 1:
                # from child to parent
                rho_ext = self.densities[0]
                rho_int = self.material.density
                calderon_vectors = self._apply_calderon(operators, trace_vectors)
                matvec_vectors = (
                    -calderon_vectors[0],
                    -(rho_ext / rho_int) * calderon_vectors[1],
                )
            else:
                raise AssertionError("Index must be 0 or 1, not: " + str(source_index))

        else:
            raise AssertionError(
                "Unknown topology of interface connector: " + connector_topology
            )

        result = _np.concatenate(matvec_vectors)

        return result

    def _apply_muller_operators(self, operators, vectors, source_index):
        """
        Apply the Müller operators to a vector to perform a local matvec.

        The operators have to map from the source to range interface and provided
        in a dictionary with the names of the operators.
        The vectors are on the source interface and are provided in a dictionary
        with the names of the traces.

        Parameters
        ----------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The boundary integral operators that map from the source to range interface.
        vectors : dict[str, numpy.ndarray]
            The trace vectors to multiply.
        source_index : int
            The index of the source interface in the connector, either 0 or 1.

        Returns
        -------
        result : numpy.ndarray
            The product of the boundary integral operators with the vector.
        """

        trace_vectors = (
            vectors["dirichlet-exterior"],
            vectors["neumann-exterior"],
        )

        connector_topology = self.connector.topology

        if connector_topology == "self-exterior":
            identity_vectors = self._apply_identity(operators, trace_vectors)
            calderon_vectors = self._apply_calderon(operators, trace_vectors)
            matvec_vectors = (
                identity_vectors[0] + calderon_vectors[0],
                identity_vectors[1] + calderon_vectors[1],
            )

        elif connector_topology == "self-interior":
            rho_ext = self.densities[0]
            rho_int = self.material.density
            scaled_vector = (
                trace_vectors[0],
                (rho_int / rho_ext) * trace_vectors[1],
            )
            calderon_vectors = self._apply_calderon(operators, scaled_vector)
            matvec_vectors = (
                -calderon_vectors[0],
                -(rho_ext / rho_int) * calderon_vectors[1],
            )

        elif connector_topology == "sibling":
            matvec_vectors = self._apply_calderon(operators, trace_vectors)

        elif connector_topology == "parent-child":
            if source_index == 0:
                # from parent to child
                rho_ext = self.densities[0]
                rho_int = self.material.density
                scaled_vector = (
                    -trace_vectors[0],
                    -(rho_int / rho_ext) * trace_vectors[1],
                )
                matvec_vectors = self._apply_calderon(operators, scaled_vector)
            elif source_index == 1:
                # from child to parent
                rho_ext = self.densities[0]
                rho_int = self.material.density
                calderon_vectors = self._apply_calderon(operators, trace_vectors)
                matvec_vectors = (
                    calderon_vectors[0],
                    (rho_ext / rho_int) * calderon_vectors[1],
                )
            else:
                raise AssertionError("Index must be 0 or 1, not: " + str(source_index))

        else:
            raise AssertionError(
                "Unknown topology of interface connector: " + connector_topology
            )

        result = _np.concatenate(matvec_vectors)

        return result

    def _apply_multitrace_operators(self, operators, vectors, source_index):
        """
        Apply the multitrace operators to a vector to perform a local matvec.

        The operators have to map from the source to range interface and provided
        in a dictionary with the names of the operators.
        The vectors are on the source interface and are provided in a dictionary
        with the names of the traces.

        Parameters
        ----------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The boundary integral operators that map from the source to range interface.
        vectors : dict[str, numpy.ndarray]
            The trace vectors to multiply.
        source_index : int
            The index of the source interface in the connector, either 0 or 1.

        Returns
        -------
        result : numpy.ndarray
            The product of the boundary integral operators with the vector.
        """

        connector_topology = self.connector.topology

        trace_dir_ext = vectors["dirichlet-exterior"]
        trace_neu_ext = vectors["neumann-exterior"]
        if "dirichlet-interior" in vectors:
            trace_dir_int = vectors["dirichlet-interior"]
        else:
            trace_dir_int = trace_dir_ext
        if "neumann-interior" in vectors:
            trace_neu_int = vectors["neumann-interior"]
        else:
            if connector_topology in ("self-exterior", "sibling") or \
                    (connector_topology == "parent-child" and source_index == 1):
                rho_int_source = self.densities[source_index]
                rho_ext_source = self.material.density
            elif connector_topology == "self-interior" or \
                    (connector_topology == "parent-child" and source_index == 0):
                rho_int_source = self.material.density
                rho_ext_source = self.densities[source_index]
            else:
                raise AssertionError("Unknown topology: " + connector_topology)
            trace_neu_int = (rho_int_source / rho_ext_source) * trace_neu_ext

        trace_vectors_ext = (trace_dir_ext, trace_neu_ext)
        trace_vectors_int = (trace_dir_int, trace_neu_int)

        if connector_topology == "self-exterior":
            rho_ext = self.material.density
            rho_int = self.densities[source_index]
            calderon_vectors = self._apply_calderon(operators, trace_vectors_ext)
            identity_vectors = self._apply_identity(operators, trace_vectors_int)
            matvec_vectors = (
                0.5 * identity_vectors[0] + calderon_vectors[0],
                (0.5 * rho_ext / rho_int) * identity_vectors[1] + calderon_vectors[1],
                _np.zeros_like(calderon_vectors[0]),
                _np.zeros_like(calderon_vectors[1]),
            )

        elif connector_topology == "self-interior":
            rho_ext = self.densities[source_index]
            rho_int = self.material.density
            identity_vectors = self._apply_identity(operators, trace_vectors_ext)
            calderon_vectors = self._apply_calderon(operators, trace_vectors_int)
            matvec_vectors = (
                _np.zeros_like(calderon_vectors[0]),
                _np.zeros_like(calderon_vectors[1]),
                0.5 * identity_vectors[0] - calderon_vectors[0],
                (0.5 * rho_int / rho_ext) * identity_vectors[1] - calderon_vectors[1],
            )

        elif connector_topology == "sibling":
            calderon_vectors = self._apply_calderon(operators, trace_vectors_ext)
            matvec_vectors = (
                calderon_vectors[0],
                calderon_vectors[1],
                _np.zeros_like(calderon_vectors[0]),
                _np.zeros_like(calderon_vectors[1]),
            )

        elif connector_topology == "parent-child":
            if source_index == 0:
                # from parent to child
                calderon_vectors = self._apply_calderon(operators, trace_vectors_int)
                matvec_vectors = (
                    -calderon_vectors[0],
                    -calderon_vectors[1],
                    _np.zeros_like(calderon_vectors[0]),
                    _np.zeros_like(calderon_vectors[1]),
                )
            elif source_index == 1:
                # from child to parent
                calderon_vectors = self._apply_calderon(operators, trace_vectors_ext)
                matvec_vectors = (
                    _np.zeros_like(calderon_vectors[0]),
                    _np.zeros_like(calderon_vectors[1]),
                    calderon_vectors[0],
                    calderon_vectors[1],
                )
            else:
                raise AssertionError("Index must be 0 or 1, not: " + str(source_index))

        else:
            raise AssertionError(
                "Unknown topology of interface connector: " + connector_topology
            )

        result = _np.concatenate(matvec_vectors)

        return result

    @staticmethod
    def _apply_identity(operators, vectors):
        """
        Apply the discrete identity operator to a vector consisting of the
        Dirichlet and Neumann spaces.

        Parameters
        ----------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The dictionary with discrete boundary integral operators. It must
            include the identity operator as "identity", defined on a single space.
        vectors : tuple[numpy.ndarray]
            The two parts of the vector to multiply: Dirichlet and Neumann traces.

        Returns
        -------
        result : tuple[numpy.ndarray]
            The two parts of the result vector: Dirichlet and Neumann traces.
        """

        dirichlet_matvec = operators["identity"] * vectors[0]
        neumann_matvec = operators["identity"] * vectors[1]

        return dirichlet_matvec, neumann_matvec

    @staticmethod
    def _apply_calderon(operators, vectors):
        """
        Apply the Calderón operator to a vector consisting of the
        Dirichlet and Neumann spaces.

        Parameters
        ----------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The dictionary with discrete boundary integral operators. It must include
            four operators, labelled as "single_layer", "double_layer",
            "adjoint_double_layer", and "hypersingular".
        vectors : tuple[numpy.ndarray]
            The two parts of the vector to multiply: Dirichlet and Neumann traces.

        Returns
        -------
        result : tuple[numpy.ndarray]
            The two parts of the result vector: Dirichlet and Neumann traces.
        """

        dirichlet_matvec = (
            -operators["double_layer"] * vectors[0]
            + operators["single_layer"] * vectors[1]
        )
        neumann_matvec = (
            operators["hypersingular"] * vectors[0]
            + operators["adjoint_double_layer"] * vectors[1]
        )

        return dirichlet_matvec, neumann_matvec


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
        calderon_operators=None,
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
        calderon_operators : None, tuple[BoundaryIntegralOperators]
            The continuous Calderón operators for the interface node, corresponding
            to the exterior and interior side of the interface. This is
            only necessary for the Calderón preconditioner.
        """

        self.identifier = identifier
        self.node = node
        self.formulation = formulation
        self.preconditioner = preconditioner
        self.space = space
        self.materials = materials
        self.frequency = frequency
        self.parameters = parameters
        self.calderon_operators = calderon_operators

        self.continuous_operators = self.create_interface_operators()
        self.discrete_operators = None

        return

    def create_interface_operators(self):
        """
        Create the preconditioner operators for an interface node.

        Returns
        -------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The continuous preconditioner operators with their names.
        """

        if self.preconditioner == "osrc" and self.formulation == "pmchwt":
            operators = self.create_osrc_operators(dtn=True, ntd=True)
        elif self.preconditioner == "calderon" and self.formulation == "pmchwt":
            operators = self.create_calderon_operators()
        else:
            raise ValueError(
                "Unknown formulation and preconditioner combination: " +
                self.formulation + ", " + self.preconditioner + "."
            )

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
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The continuous OSRC preconditioner operators with
            their names: 'NtD' or 'DtN'.
        """

        from .formulations import process_osrc_parameters
        from .acoustics import create_osrc_operators
        from optimus import global_parameters

        osrc_params = process_osrc_parameters(self.parameters)

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

    def create_calderon_operators(self):
        """
        Create the Calderón preconditioner operators for an interface node.

        Returns
        -------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The continuous Calderón preconditioner operators with their names.
        """

        from .formulations import process_calderon_parameters

        calderon_params = process_calderon_parameters(self.parameters)

        if self.calderon_operators is None:
            raise AssertionError(
                "The continuous Calderón operators have not been created yet."
            )

        if calderon_params["domain"] == "exterior":
            operators_object = self.calderon_operators[0]
        elif calderon_params["domain"] == "interior":
            operators_object = self.calderon_operators[1]
        else:
            raise ValueError(
                "Unknown domain for Calderón preconditioner: "
                + calderon_params["domain"] + "."
            )

        operators_dict = operators_object.continuous_operators[0]

        return operators_dict

    def assemble_boundary_integral_operators(self):
        """
        Assemble the boundary integral operators for the interface node.

        Sets "self.discrete_operators" to a dictionary with assembled operators.

        Returns
        -------
        operators : dict[str, bempp.api.assembly.boundary_operator.BoundaryOperator]
            The discrete boundary integral operators of the preconditioner.
        """

        assembled_operators = {}

        if not isinstance(self.continuous_operators, dict):
            raise AssertionError(
                "The continuous preconditioner operators have not been created yet."
            )

        for key, operator in self.continuous_operators.items():
            assembled_operators[key] = operator.strong_form()

        self.discrete_operators = assembled_operators

        return assembled_operators

    def matvec(self, vector):
        """
        Matrix-vector product for a interface node.

        The vector is the entire vector for the interface, and may include
        multiple traces depending on the formulation.
        Only single self-interactions are implemented for a preconditioner.

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
        elif self.preconditioner == "calderon":
            result = self._apply_calderon_operators(vector)
        else:
            raise ValueError("Unknown preconditioner: " + self.preconditioner + ".")

        return result

    def _apply_osrc_operators(self, vector):
        """
        Apply the OSRC preconditioner system to a vector.

        The OSRC preconditioner applies the OSRC-NtD operator to the Neumann
        trace to calculate a Dirichlet trace, and applies the OSRD-DtN operator
        to the Dirichlet trace to calculate a Neumann trace.
        The vector is the entire vector for the interface, including first the
        Dirichlet, and then the Neumann trace.

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
            trace_dir, trace_neu = _np.split(
                vector,
                indices_or_sections=[self.space.global_dof_count],
            )
            result_dir = self.discrete_operators["NtD"] * trace_neu
            result_neu = self.discrete_operators["DtN"] * trace_dir
            result = _np.concatenate([result_dir, result_neu])
        else:
            raise ValueError("Unknown formulation: " + self.formulation + ".")

        return result

    def _apply_calderon_operators(self, vector):
        """
        Apply the Calderón preconditioner system to a vector.

        The Calderón preconditioner applies the Calderón operator to the Dirichlet
        and Neumann traces. Here, we assume that the input vector has first
        the Dirichlet trace, and then the Neumann trace. Similarly, the output
        vector also has first the Dirichlet trace, and then the Neumann trace.
        In this case, the standard Calderón system needs to applied to the input
        vector to apply the preconditioner.

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
            trace_dir, trace_neu = _np.split(
                vector,
                indices_or_sections=[self.space.global_dof_count],
            )
            result_dir = (
                -self.discrete_operators["double_layer"] * trace_dir
                + self.discrete_operators["single_layer"] * trace_neu
            )
            result_neu = (
                self.discrete_operators["hypersingular"] * trace_dir
                + self.discrete_operators["adjoint_double_layer"] * trace_neu
            )
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

        if self.formulation in ("pmchwt", "muller", "multitrace"):
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
            elif self.preconditioner_name in ("mass", "osrc", "calderon"):
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

            if self.formulation == "multitrace":
                self.rhs_vector = _np.concatenate(
                    [self.rhs_vector, _np.zeros_like(self.rhs_vector)]
                )

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

        zero_vector = _np.zeros(self.space.global_dof_count, dtype=_np.complex128)

        if self.formulation in ("pmchwt", "muller"):
            n_spaces = 2
        elif self.formulation == "multitrace":
            n_spaces = 4
        else:
            raise ValueError("Unknown formulation: " + self.formulation + ".")

        self.rhs_vector = _np.concatenate([zero_vector] * n_spaces)

        return
