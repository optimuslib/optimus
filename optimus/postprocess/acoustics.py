"""Calculate acoustic fields from surface potentials."""

import numpy as _np
import bempp.api as _bempp
import time as _time


class Field:
    def __init__(self, model, verbose=False):
        """
        Base class for field visualisation.

        Parameters
        ----------
        model: optimus.model.common.Model
            The boundary element model.
        verbose: boolean
            Display the logs.
        """

        self.model = model
        self.verbose = verbose

        # Retrieve all surface grids from the model. This is a list
        # of Bempp grids, for each (active/inactive) interface in the model,
        # with the same ordering as the interfaces.
        self._domain_grids = self._get_domain_grids(model)

        # An array of all 3D points for visualisation.
        self.points = None

        # Segmentation of the visualisation points.
        self.points_interior = None
        self.points_exterior = None
        self.points_boundary = None
        self.index_interior = None
        self.index_exterior = None
        self.index_boundary = None
        self.regions = None
        self.region_legend = None

        # The total, scattered, and incident field in the specified points.
        self.total_field = None
        self.scattered_field = None
        self.incident_field = None

        return

    @staticmethod
    def _get_domain_grids(model):
        """
        Retrieve the grids of the domains from the model.

        Needs to be overridden by a specific postprocessing type.

        Parameters
        ----------
        model: optimus.model.common.Model
            The boundary element model.

        Returns
        -------
        domain_grids: list[bempp.api.Grid, None]
            The grids of the subdomains, if available.
        """
        raise NotImplementedError

    def segmentation(self, points):
        """
        Perform segmentation of the field points into interior, exterior and
        boundary points for each subdomain.

        Notice that interior points mean inside the interface. Hence, they
        may be inside an interface that is interior to this interface.
        If there is no point in the interior/boundary/exterior, the corresponding
        element is a numpy array of size (3,0).
        For inactivated interfaces, the element for interior and boundary points
        and indices is None.

        Parameters
        ----------
        points: numpy.ndarray
            The field points. The size of the array should be (3,N).
        """

        from ..utils.conversions import convert_to_3n_array
        from .topology import find_int_ext_points

        self.points = convert_to_3n_array(points)

        (
            self.points_interior,
            self.points_exterior,
            self.points_boundary,
            self.index_interior,
            self.index_exterior,
            self.index_boundary,
        ) = find_int_ext_points(self._domain_grids, self.points, self.verbose)

        self.create_regions()

        return

    def create_regions(self):
        """
        Create regions for the field points.

        Sets a label for each point, to identify its unique region.
        """

        from .topology import create_regions as create_regions_topology

        self.regions, self.region_legend = create_regions_topology(
            self.points,
            self.index_exterior,
            self.index_interior,
            self.index_boundary,
        )

        return


class ExteriorField(Field):
    def __init__(self, model, verbose=False):
        """
        Class for field visualisation for exterior BEM models with
        disjoint penetrable scatterers.

        Parameters
        ----------
        model: optimus.model.common.ExteriorModel
            The boundary element model.
        verbose: boolean
            Display the logs.
        """

        from ..model.common import ExteriorModel

        if not isinstance(model, ExteriorModel):
            raise AssertionError("The model is not an exterior model.")

        super().__init__(model, verbose)

        return

    @staticmethod
    def _get_domain_grids(model):
        """
        Retrieve the grids of the domains from the model.

        Parameters
        ----------
        model: optimus.model.common.ExteriorModel
            The boundary element model.

        Returns
        -------
        domain_grids: list[bempp.api.Grid]
            The grids of the subdomains.
        """

        domain_grids = [
            model.geometry[n_sub].grid for n_sub in range(model.n_subdomains)
        ]

        return domain_grids

    # noinspection DuplicatedCode, PyUnresolvedReferences
    def compute_fields(self):
        """
        Calculate the acoustic field in the visualisation points.

        In each visualisation point, the incident, scattered and total
        acoustic field needs to be computed, according to the model.
        Notice that for exterior models, only the PMCHWT formulation is
        available. The field is computed separately for each subdomain,
        with special care for the boundary points.
        """

        from ..utils.generic import chunker
        from optimus import global_parameters

        if self.model.solution is None:
            raise ValueError(
                "The model does not contain a solution, so no fields can be computed. "
                "Please solve the model first."
            )

        global_parameters.bem.update_hmat_parameters("potential")

        start_time_pot_ops = _time.time()
        if self.verbose:
            print(
                "\n Calculating the interior and exterior potential operators. "
                "Started at: ",
                _time.strftime("%a, %d %b %Y %H:%M:%S", _time.localtime()),
            )

        total_field = _np.full(self.points.shape[1], _np.nan, dtype=complex)
        scattered_field = _np.full(self.points.shape[1], _np.nan, dtype=complex)
        incident_exterior_field = _np.full(self.points.shape[1], _np.nan, dtype=complex)

        if self.index_exterior.any():
            exterior_values = _np.zeros(
                (1, self.points_exterior.shape[1]), dtype="complex128"
            )
            ext_calc_flag = True
        else:
            exterior_values = None
            ext_calc_flag = False

        if self.index_boundary:
            bound_calc_flag = True
        else:
            bound_calc_flag = False

        i = 0
        for (
            solution_pair,
            space,
            interior_point,
            interior_idx,
            interior_material,
        ) in zip(
            chunker(self.model.solution, 2),
            self.model.space,
            self.points_interior,
            self.index_interior,
            self.model.material_interior,
        ):
            if self.verbose:
                print("Calculating the fields of Domain {0}".format(i + 1))
                print(
                    interior_point.shape,
                    interior_idx.shape,
                    interior_material.compute_wavenumber(self.model.source.frequency),
                    interior_material,
                )

            if interior_idx.any():
                pot_int_sl = _bempp.operators.potential.helmholtz.single_layer(
                    space,
                    interior_point,
                    interior_material.compute_wavenumber(self.model.source.frequency),
                )
                pot_int_dl = _bempp.operators.potential.helmholtz.double_layer(
                    space,
                    interior_point,
                    interior_material.compute_wavenumber(self.model.source.frequency),
                )
                rho_ratio = (
                    interior_material.density / self.model.material_exterior.density
                )
                interior_value = (
                    pot_int_sl * solution_pair[1] * rho_ratio
                    - pot_int_dl * solution_pair[0]
                )
                total_field[interior_idx] = interior_value.ravel()

            if ext_calc_flag:
                pot_ext_sl = _bempp.operators.potential.helmholtz.single_layer(
                    space,
                    self.points_exterior,
                    self.model.material_exterior.compute_wavenumber(
                        self.model.source.frequency
                    ),
                )
                pot_ext_dl = _bempp.operators.potential.helmholtz.double_layer(
                    space,
                    self.points_exterior,
                    self.model.material_exterior.compute_wavenumber(
                        self.model.source.frequency
                    ),
                )
                exterior_values += (
                    -pot_ext_sl * solution_pair[1] + pot_ext_dl * solution_pair[0]
                )

            i += 1

            if self.verbose:
                end_time_pot_ops = _time.time()
                print(
                    "\n Calculating the interior and exterior potential operators "
                    "Finished... Duration in secs: ",
                    end_time_pot_ops - start_time_pot_ops,
                )

        if ext_calc_flag:
            start_time_pinc = _time.time()
            if self.verbose:
                print(
                    "\n Calculating the incident field Started at: ",
                    _time.strftime("%a, %d %b %Y %H:%M:%S", _time.localtime()),
                )

            incident_exterior = self.model.source.pressure_field(
                self.model.material_exterior, self.points_exterior
            )

            end_time_pinc = _time.time()
            if self.verbose:
                print(
                    "\n Calculating the incident field Finished... Duration in secs: ",
                    end_time_pinc - start_time_pinc,
                )
            incident_exterior_field[self.index_exterior] = incident_exterior.ravel()
            scattered_field[self.index_exterior] = exterior_values.ravel()
            total_field[self.index_exterior] = (
                scattered_field[self.index_exterior]
                + incident_exterior_field[self.index_exterior]
            )

        if bound_calc_flag:
            for subdomain_number in range(self.model.n_subdomains):
                grid = self.model.geometry[subdomain_number].grid
                dirichlet_solution = self.model.solution[
                    2 * subdomain_number
                ].coefficients
                subdomain_boundary_points = self.points_boundary[subdomain_number]

                total_field[
                    self.index_boundary[subdomain_number]
                ] = compute_pressure_boundary(
                    grid, subdomain_boundary_points, dirichlet_solution
                )

        self.total_field = total_field
        self.scattered_field = scattered_field
        self.incident_field = incident_exterior_field

        return


class AnalyticalField(Field):
    def __init__(self, model, verbose=False):
        """
        Class for field visualisation for analytical acoustic models.

        The analytical model is based on spherical harmonics for a single
        penetrable sphere.

        Parameters
        ----------
        model: optimus.model.acoustics.Analytical
            The boundary element model.
        verbose: boolean
            Display the logs.
        """

        from ..model.acoustics import Analytical

        if not isinstance(model, Analytical):
            raise AssertionError("The model is not an analytical model.")

        super().__init__(model, verbose)

        return

    @staticmethod
    def _get_domain_grids(model):
        """
        Retrieve the grids of the domains from the model.

        Parameters
        ----------
        model: optimus.model.acoustics.Analytical
            The boundary element model.

        Returns
        -------
        domain_grids: list[bempp.api.Grid]
            The grids of the subdomains.
        """

        domain_grids = [model.geometry.grid]

        return domain_grids

    # noinspection DuplicatedCode, PyUnresolvedReferences
    def compute_fields(self):
        """
        Calculate the scattered and total pressure fields for visualisation
        in the analytical model.

        Returns
        -------
        total_field : numpy.ndarray
            An array of size (1,N) with complex values of the total pressure field.
        scattered_field : numpy.ndarray
            An array of size (1,N) with complex values of the scatterd pressure field.
        incident_exterior_field : numpy.ndarray
            An array of size (1,N) with complex values of the incident pressure field
            in the exterior domain.
        """

        from scipy.special import sph_jn, sph_yn, eval_legendre

        if (self.model.scattered_coefficients is None or
                self.model.interior_coefficients is None):
            raise ValueError(
                "The model does not contain a solution, so no fields can be computed. "
                "Please solve the model first."
            )

        total_field = _np.full(self.points.shape[1], _np.nan, dtype=complex)
        scattered_field = _np.full(self.points.shape[1], _np.nan, dtype=complex)
        incident_exterior_field = _np.full(self.points.shape[1], _np.nan, dtype=complex)

        frequency = self.model.source.frequency
        k_ext = self.model.material_exterior.compute_wavenumber(frequency)
        k_int = self.model.material_interior.compute_wavenumber(frequency)

        n_iter = self.model.interior_coefficients.size

        #
        # Interior
        #
        pi = self.points_interior[0]
        ii = self.index_interior[0]
        if ii.any():
            radial_space = _np.linalg.norm(pi, axis=0)
            directional_space = _np.dot(self.model.source.direction_vector, pi)
            directional_space /= radial_space

            jn, djn = _np.array(
                list(zip(*[sph_jn(n_iter - 1, k_int * r) for r in radial_space]))
            )

            legendre = _np.array(
                [eval_legendre(n, directional_space) for n in range(n_iter)]
            )

            total_field[ii] = _np.dot(self.model.interior_coefficients, jn.T * legendre)

        #
        # Exterior
        #
        pe = self.points_exterior
        ie = self.index_exterior
        if ie.any():
            radial_space = _np.linalg.norm(pe, axis=0)
            directional_space = _np.dot(self.model.source.direction_vector, pe)
            directional_space /= radial_space

            jn, djn = _np.array(
                list(zip(*[sph_jn(n_iter - 1, k_ext * r) for r in radial_space]))
            )
            yn, dyn = _np.array(
                list(zip(*[sph_yn(n_iter - 1, k_ext * r) for r in radial_space]))
            )
            h1n, dh1n = jn.T + 1j * yn.T, djn.T + 1j * dyn.T

            legendre = _np.array(
                [eval_legendre(n, directional_space) for n in range(n_iter)]
            )

            scattered_field[ie] = _np.dot(
                self.model.scattered_coefficients, h1n * legendre
            )

            incident_exterior_field[ie] = _np.dot(
                _np.array([(2 * n + 1) * 1j**n for n in range(n_iter)]),
                jn.T * legendre,
            )

            total_field[ie] = scattered_field[ie] + incident_exterior_field[ie]

        #
        # Boundary
        #
        pb = self.points_boundary[0]
        ib = self.index_boundary[0]
        if ib.any():
            # We use the interior field to compute the boundary points
            radial_space = _np.linalg.norm(pb, axis=0)
            directional_space = _np.dot(self.model.source.direction_vector, pb)
            directional_space /= radial_space

            jn, djn = _np.array(
                list(zip(*[sph_jn(n_iter - 1, k_int * r) for r in radial_space]))
            )

            legendre = _np.array(
                [eval_legendre(n, directional_space) for n in range(n_iter)]
            )

            total_field[ib] = _np.dot(self.model.interior_coefficients, jn.T * legendre)

        self.total_field = total_field
        self.scattered_field = scattered_field
        self.incident_field = incident_exterior_field

        return


class NestedField(Field):
    def __init__(self, model, verbose=False):
        """
        Class for field visualisation for nested BEM models.

        Parameters
        ----------
        model: optimus.model.nested.NestedModel
            The boundary element model.
        verbose: boolean
            Display the logs.
        """

        from ..model.nested import NestedModel

        if not isinstance(model, NestedModel):
            raise AssertionError("The model is not a nested model.")

        super().__init__(model, verbose)

        self.graph = self.model.topology

        self.index_subdomain = None
        self.points_subdomain = None

        self.potential_operators = None
        self.fields_interface = None

        return

    @staticmethod
    def _get_domain_grids(model):
        """
        Retrieve the grids of the domains from the model.

        Parameters
        ----------
        model: optimus.model.common.GraphModel
            The boundary element model.

        Returns
        -------
        domains_grids: list[bempp.api.Grid]
            The grids of the subdomains.
        """

        domains_grids = []
        for interface in model.topology.interface_nodes:
            if interface.geometry is None:
                domains_grids.append(None)
            else:
                domains_grids.append(interface.geometry.grid)

        return domains_grids

    def create_regions(self):
        """
        Create regions for the field points in a nested topology.

        Sets a label for each point, to identify its unique region.
        """

        from copy import deepcopy
        from .topology import create_regions as create_regions_topology

        self._subdomain_segmentation()

        exterior_subdomain_id = None
        for subdomain in self.graph.subdomain_nodes:
            if subdomain.is_active() and not subdomain.bounded:
                exterior_subdomain_id = subdomain.identifier
                break

        indices_interior = deepcopy(self.index_subdomain)
        index_exterior = indices_interior.pop(exterior_subdomain_id)
        indices_boundary = deepcopy(self.index_boundary)
        indices_boundary.pop(exterior_subdomain_id)

        self.regions, self.region_legend = create_regions_topology(
            self.points,
            index_exterior,
            indices_interior,
            indices_boundary,
        )

        return

    def _subdomain_segmentation(self):
        """
        Create a subdomain segmentation of the visualisation points.

        The points should already have been segmented into regions
        interior to the interfaces. Remember that interior to an
        interface may mean interior to another interface inside the
        subdomain. Here, create a subdomain segmentation, where
        each point is in one unique subdomain.
        """

        self.index_subdomain = []

        for subdomain in self.graph.subdomain_nodes:
            if subdomain.is_active():
                if subdomain.bounded:
                    index_subdomain = self.index_interior[subdomain.parent_interface_id]
                    for child_interface_id in subdomain.child_interfaces_ids:
                        index_child = _np.logical_or(
                            self.index_interior[child_interface_id],
                            self.index_boundary[child_interface_id],
                        )
                        index_subdomain = _np.logical_and(
                            index_subdomain, _np.logical_not(index_child)
                        )
                    self.index_subdomain.append(index_subdomain)
                else:
                    self.index_subdomain.append(self.index_exterior)
            else:
                self.index_subdomain.append(None)

        self.points_subdomain = [
            self.points[:, indices] for indices in self.index_subdomain
        ]

        return

    def compute_fields(self):
        """
        Calculate the acoustic field in the visualisation points.

        In each visualisation point, the incident, scattered and total
        acoustic field needs to be computed, according to the boundary
        integral formulation.
        """

        from optimus import global_parameters

        if self.model.solution is None:
            raise ValueError(
                "The model does not contain a solution, so no fields can be computed. "
                "Please solve the model first."
            )

        global_parameters.bem.update_hmat_parameters("potential")

        self.assemble_potential_operators()
        self.calculate_subdomain_fields()
        self.calculate_interface_fields()
        self.construct_fields()

        return

    def assemble_potential_operators(self):
        """
        Assemble potential operators corresponding to nested domains.

        For each edge in the graph topology, determine the representation
        formula corresponding to the boundary integral formulation and
        assemble the potential operators.
        An 'edge' in the graph topology is a connection between a material
        interface and a propagation subdomain.

        Sets "self.potential_operators" to a list of PotentialIntegralOperators, one
        for each edge object.
        """

        self.potential_operators = []

        for edge in self.graph.edges:
            if edge.is_active():
                interface_node = self.graph.interface_nodes[edge.interface_id]
                subdomain_node = self.graph.subdomain_nodes[edge.subdomain_id]

                if edge.orientation == "interface_interior_to_subdomain":
                    opposite_subdomain_id = interface_node.child_subdomain_id
                elif edge.orientation == "interface_exterior_to_subdomain":
                    opposite_subdomain_id = interface_node.parent_subdomain_id
                else:
                    raise AssertionError(
                        "Unknown edge orientation: " + edge.orientation
                    )
                opposite_subdomain = self.graph.subdomain_nodes[opposite_subdomain_id]

                self.potential_operators.append(
                    PotentialIntegralOperators(
                        identifier=len(self.potential_operators),
                        edge=edge,
                        representation=self.model.representation[edge.interface_id],
                        formulation=self.model.formulation[edge.interface_id],
                        surface_potentials=self.model.solution[edge.interface_id],
                        points=self.points_subdomain[edge.subdomain_id],
                        material=subdomain_node.material,
                        density=opposite_subdomain.material.density,
                        frequency=self.model.frequency,
                    )
                )
            else:
                self.potential_operators.append(None)

        return

    def calculate_subdomain_fields(self):
        """
        Calculate the field for each edge in the graph topology.

        Apply the representation formula for each edge in the graph topology
        so that the corresponding field component inside each subdomain is
        calculated.

        The fields will be stored inside the stored PotentialIntegralOperators.
        """

        for operators in self.potential_operators:
            if operators is not None:
                operators.apply_representation_formula()

        return

    def calculate_interface_fields(self):
        """
        Calculate the field for all points that are close to a physical interface.

        When visualisation points are located close to a physical interface,
        the default field calculation is inaccurate due to the singularity
        in Green's function. One needs to apply specialized quadrature rules.
        The implemented routine requires the Dirichlet trace as input.

        Sets "self.fields_interface" to a list of numpy arrays with field values
        at each interface.
        """

        self.fields_interface = []
        for interface in self.graph.interface_nodes:
            if self.index_boundary[interface.identifier] is not None:
                formulation = self.model.formulation[interface.identifier]
                if formulation in ("pmchwt", "muller", "multitrace"):
                    dirichlet_potential = self.model.solution[interface.identifier][0]
                else:
                    raise AssertionError(
                        "Unknown formulation for boundary points: " + formulation
                    )
                self.fields_interface.append(
                    compute_pressure_boundary(
                        interface.geometry.grid,
                        self.points_boundary[interface.identifier],
                        dirichlet_potential.coefficients,
                    )
                )
            else:
                self.fields_interface.append(None)

        return

    def construct_fields(self):
        """
        Construct the acoustic field in the visualisation points.

        The representation formula computes the field inside each subdomain
        from each connecting interface. The interpretation of this field
        depends on the representation formula but is typically the scattered
        field in the exterior and the total field in bounded subdomains.
        All these scattered fields and the incident field need to be summed
        to obtain the total field.

        For the moment, the function is restricted to representation formulas for
        the total field in the interior and the scattered field in the exterior.

        Sets the attributes:
         - self.incident_field: the incident field in the visualisation points.
         - self.scattered_field: the scattered field in the visualisation points.
         - self.total_field: the total field in the visualisation points.
        """

        # The attribute self.points_subdomain is a (3,N) array, also if N=0.
        interior_fields = [
            _np.zeros(points.shape[1], dtype=complex)
            for points in self.points_subdomain
        ]

        # The attribute operators.field is None if no interior points are present.
        for edge, operators in zip(self.graph.edges, self.potential_operators):
            if operators is not None and operators.field is not None:
                interior_fields[edge.subdomain_id] += operators.field

        self.incident_field = self.calculate_incident_field()
        self.scattered_field = _np.zeros(self.points.shape[1], dtype=complex)
        self.total_field = _np.zeros(self.points.shape[1], dtype=complex)

        for subdomain_id, subdomain in enumerate(self.graph.subdomain_nodes):
            indices = self.index_subdomain[subdomain_id]
            if indices.size > 0:
                if subdomain.bounded:
                    self.total_field[indices] = interior_fields[subdomain_id]
                    self.scattered_field[indices] = (
                        self.total_field[indices] - self.incident_field[indices]
                    )
                else:
                    self.scattered_field[indices] = interior_fields[subdomain_id]
                    self.total_field[indices] = (
                        self.scattered_field[indices] + self.incident_field[indices]
                    )

        for interface_id, field in enumerate(self.fields_interface):
            if field is not None:
                indices = self.index_boundary[interface_id]
                self.total_field[indices] = field
                self.scattered_field[indices] = (
                    self.total_field[indices] - self.incident_field[indices]
                )

        return

    def calculate_incident_field(self):
        """
        Calculate the incident field in all points.

        The incident field is the field that would be present without
        any scatterer. Hence, the propagating medium is the exterior.

        For the moment, the function is limited to sources from the
        exterior domain only.

        Returns
        -------
        incident_field : numpy.ndarray
            The incident field in all N points.
        """

        exterior_subdomain = None
        for subdomain in self.graph.subdomain_nodes:
            if subdomain.is_active() and not subdomain.bounded:
                if exterior_subdomain is None:
                    exterior_subdomain = subdomain
                else:
                    raise AssertionError("There should be only one exterior subdomain.")
        if exterior_subdomain is None:
            raise AssertionError("There should be one exterior subdomain.")

        incident_field = _np.zeros(self.points.shape[1], dtype=complex)
        for source in exterior_subdomain.sources:
            incident_field += source.pressure_field(
                exterior_subdomain.material,
                self.points,
            )

        return incident_field


class PotentialIntegralOperators:
    def __init__(
        self,
        identifier,
        edge,
        representation,
        formulation,
        surface_potentials,
        points,
        material,
        density,
        frequency,
    ):
        """
        Create the potential integral operators for an active edge in the graph
        topology for nested BEM.

        Parameters
        ----------
        identifier : int
            The unique identifier of the potential integral operator in the model.
        edge : optimus.geometry.graph_topology.Edge
            The edge of the graph topology.
        representation : str
            The type of boundary integral representation.
        formulation : str
            The type of boundary integral formulation.
        surface_potentials : tuple[bempp.api.GridFunction]
            The function space of the boundary potential.
        points : numpy.ndarray
            The points where the potential is to be evaluated.
            Array of size (3,N) with N the number of points, which may be zero.
        material : optimus.material.Material
            The material of the propagating medium.
        density : float
            The density of the medium at the other side of the edge than
            the propagating medium.
        frequency : float
            The frequency of the propagating wave.
        """

        self.identifier = identifier

        self.edge = edge
        self.representation = representation
        self.formulation = formulation
        self.points = points
        self.material = material
        self.densities = self._process_densities(density)
        self.frequency = frequency

        self.wavenumber = self.material.compute_wavenumber(self.frequency)

        self.surface_potentials, self.spaces = self.process_surface_potentials(
            surface_potentials
        )
        self.potential_operators = self.create_potential_operators()

        self.field = None

        return

    def _process_densities(self, density):
        """
        Process the densities of the exterior and interior media.

        Parameters
        ----------
        density : float
            The density of the medium at the other side of the edge than
            the propagating medium.

        Returns
        -------
        densities : dict[str, float]
            The densities of the exterior and interior media.
        """

        if self.edge.orientation == "interface_interior_to_subdomain":
            density_exterior = self.material.density
            density_interior = density
        elif self.edge.orientation == "interface_exterior_to_subdomain":
            density_exterior = density
            density_interior = self.material.density
        else:
            raise AssertionError("Edge orientation not recognised.")

        densities = {
            "exterior": density_exterior,
            "interior": density_interior,
        }

        return densities

    def process_surface_potentials(self, surface_potentials):
        """
        Process the surface potentials.

        The surface potentials come from the solved BEM. Depending on
        the formulation, each interface can have multiple potentials.
        Check the input and store the surface potentials.

        The PMCHWT and Müller formulation have the exterior Dirichlet and Neumann
        traces of the field as solution potentials.
        The multi-trace formulation has all four traces of the field as solution.

        Parameters
        ----------
        surface_potentials : tuple[bempp.api.GridFunction]
            The surface potentials.

        Returns
        -------
        surface_potentials : dict[str, bempp.api.GridFunction]
            The surface potentials.
        """

        if not isinstance(surface_potentials, (list, tuple)):
            raise AssertionError("Surface potentials should be a list or tuple.")

        for potential in surface_potentials:
            if not isinstance(potential, _bempp.GridFunction):
                raise AssertionError("Surface potentials should be GridFunctions.")

        if self.formulation in ("pmchwt", "muller"):
            if len(surface_potentials) != 2:
                raise AssertionError(
                    "The PMCHWT and Müller formulations require two surface potentials."
                )
            else:
                surface_potentials_dict = {
                    "dirichlet": surface_potentials[0],
                    "neumann": surface_potentials[1],
                }
                function_spaces = {
                    "dirichlet": surface_potentials[0].space,
                    "neumann": surface_potentials[1].space,
                }

        elif self.formulation == "multitrace":
            if len(surface_potentials) != 4:
                raise AssertionError(
                    "The multi-trace formulation requires four surface potentials."
                )
            else:
                surface_potentials_dict = {
                    "dirichlet-exterior": surface_potentials[0],
                    "neumann-exterior": surface_potentials[1],
                    "dirichlet-interior": surface_potentials[2],
                    "neumann-interior": surface_potentials[3],
                }
                function_spaces = {
                    "dirichlet": surface_potentials[0].space,
                    "neumann": surface_potentials[1].space,
                }

        else:
            raise AssertionError("Unknown formulation: " + self.formulation)

        return surface_potentials_dict, function_spaces

    def compute_field_trace(self, trace_type):
        """
        Compute the trace of the field at the interface, given the surface potential.

        The traces of the field depend on the specific formulation:
         - For the PMCHWT and Müller formulations, the Dirichlet potential is the
            Dirichlet trace of the total field, while the Neumann potential is the
            exterior Neumann trace of the total field. The interior Neumann trace
            of the total field can be computed by a scaling of the density ratio.
         - For the multi-trace formulation, the solution potentials are the traces
            of the total field.

        Parameters
        ----------
        trace_type : str
            The type of trace to be computed:
             - "interior_dirichlet"
             - "exterior_dirichlet"
             - "interior_neumann"
             - "exterior_neumann"

        Returns
        -------
        field_trace : bempp.api.GridFunction
            The trace of the field.
        """

        if self.formulation in ("pmchwt", "muller"):
            if trace_type == "exterior_dirichlet":
                field_trace = self.surface_potentials["dirichlet"]
            elif trace_type == "interior_dirichlet":
                field_trace = self.surface_potentials["dirichlet"]
            elif trace_type == "exterior_neumann":
                field_trace = self.surface_potentials["neumann"]
            elif trace_type == "interior_neumann":
                rho_int = self.densities["interior"]
                rho_ext = self.densities["exterior"]
                field_trace = (rho_int / rho_ext) * self.surface_potentials["neumann"]
            else:
                raise AssertionError("Unknown trace type: " + trace_type)
        elif self.formulation == "multitrace":
            if trace_type == "exterior_dirichlet":
                field_trace = self.surface_potentials["dirichlet-exterior"]
            elif trace_type == "interior_dirichlet":
                field_trace = self.surface_potentials["dirichlet-interior"]
            elif trace_type == "exterior_neumann":
                field_trace = self.surface_potentials["neumann-exterior"]
            elif trace_type == "interior_neumann":
                field_trace = self.surface_potentials["neumann-interior"]
            else:
                raise AssertionError("Unknown trace type: " + trace_type)
        else:
            raise AssertionError("Unknown formulation: " + self.formulation)

        return field_trace

    def create_potential_operators(self):
        """
        Create the potential operators for the edge.

        The points are numpy arrays with shape (3,N) with N the number of points,
        possible zero, in which case it remains a 2D array.

        Returns
        -------
        potential_operators : None, dict[str, bempp.api.operators.potential.Helmholtz]
            The potential operators of the Helmholtz equation.
            Both Dirichlet and Neumann operators are stored, depending on the
            representation formula.
            Returns None if no point is located in the subdomain of the edge.
        """

        if self.edge.is_active() and self.points.shape[1] > 0:
            if self.representation == "direct":
                sl_pot = _bempp.operators.potential.helmholtz.single_layer(
                    self.spaces["neumann"],
                    self.points,
                    self.wavenumber,
                )
                dl_pot = _bempp.operators.potential.helmholtz.double_layer(
                    self.spaces["dirichlet"],
                    self.points,
                    self.wavenumber,
                )
                potential_operators = {
                    "single_layer": sl_pot,
                    "double_layer": dl_pot,
                }
            else:
                raise AssertionError("Unknown representation: " + self.representation)
        else:
            potential_operators = None

        return potential_operators

    def apply_representation_formula(self):
        """
        Compute the field by applying the potential operators to the surface potentials.

        The calculation depends on the representation formula.

        Sets the "self.field" attribute to a numpy.ndarray of shape (N,)
        with the field values, for N the number of points in the subdomain.
        The attribute is None for inactive edges and also when no point
        is located in the subdomain.
        """

        if self.potential_operators is None:
            field = None
        else:
            if self.representation == "direct":
                if self.edge.orientation == "interface_interior_to_subdomain":
                    trace_dirichlet = -self.compute_field_trace("exterior_dirichlet")
                    trace_neumann = -self.compute_field_trace("exterior_neumann")
                elif self.edge.orientation == "interface_exterior_to_subdomain":
                    trace_dirichlet = self.compute_field_trace("interior_dirichlet")
                    trace_neumann = self.compute_field_trace("interior_neumann")
                else:
                    raise AssertionError("Edge orientation not recognised.")

                field = (
                    self.potential_operators["single_layer"] * trace_neumann
                    - self.potential_operators["double_layer"] * trace_dirichlet
                )
                # BEMPP generates an (1,N) array, but we want an (N,) array
                field = field.ravel()
            else:
                raise AssertionError("Unknown representation: " + self.representation)

        self.field = field

        return


# noinspection PyUnresolvedReferences
def compute_pressure_boundary(grid, boundary_points, dirichlet_solution):
    """
    Calculate pressure for points near or at the boundary of a domain. When the solid
    angle associated with a boundary vertex is below 0.1, it is assumed to lie on the
    boundary.

    Parameters
    ----------
    grid : bempp.api.Grid
        The surface mesh of bempp.
    boundary_points : numpy.ndarray
        An array of size (3,N) with the coordinates of vertices
        on the domain boundary.
    dirichlet_solution : numpy.ndarray
        An array of size (N,) with the Dirichlet component of the
        solution vector on the boundary.

    Returns
    -------
    total_boundary_pressure : numpy.ndarray
        An array of size (N,) with complex values of the pressure field.
    """

    from ..utils.linalg import normalize_vector

    vertices = grid.leaf_view.vertices
    elements = grid.leaf_view.elements
    centroids = _np.mean(vertices[:, elements], axis=1)

    # Initialise arrays with None-values for the element indices
    n = boundary_points.shape[1]
    element_index = _np.array([None] * n)

    # Loop over all centroids and find the elements within which boundary points lie
    for i in range(n):
        eucl_norm = _np.linalg.norm(
            centroids - _np.atleast_2d(boundary_points[:, i]).transpose(), axis=0
        )

        comp = _np.where(eucl_norm == _np.min(eucl_norm))[0]

        if comp.size != 0:
            element_index[i] = comp[0]

    space = _bempp.function_space(grid, "P", 1)
    grid_function = _bempp.GridFunction(space, coefficients=dirichlet_solution)
    local_coords = _np.zeros((2, n), dtype=float)
    total_boundary_pressure = _np.zeros(n, dtype="complex128")

    # Loop over elements within which near points lie
    for i in range(n):
        # Obtain vertices of element
        vertices_elem = vertices[:, elements[:, element_index[i]]].transpose()

        # Translate element so that first vertex is global origin
        vertices_translated = vertices_elem - vertices_elem[0, :]
        boundary_point_translated = boundary_points[:, i] - vertices_elem[0, :]

        # Compute element normal
        vector_a = vertices_translated[1, :] - vertices_translated[0, :]
        vector_b = vertices_translated[2, :] - vertices_translated[0, :]
        vector_a_cross_vector_b = _np.cross(vector_a, vector_b)
        element_normal = normalize_vector(vector_a_cross_vector_b)

        # Obtain first rotation matrix for coordinate transformation
        h = _np.sqrt(element_normal[0] ** 2 + element_normal[1] ** 2)
        if h != 0:
            r_z = _np.array(
                [
                    [element_normal[0] / h, element_normal[1] / h, 0],
                    [-element_normal[1] / h, element_normal[0] / h, 0],
                    [0, 0, 1],
                ]
            )
        else:
            r_z = _np.identity(3, dtype=float)

        # Obtain rotated element normal
        element_normal_rotated = _np.matmul(r_z, element_normal)

        # Obtain second rotation matrix for coordinate transformation
        r_y = _np.array(
            [
                [element_normal_rotated[2], 0, -element_normal_rotated[0]],
                [0, 1, 0],
                [element_normal_rotated[0], 0, element_normal_rotated[2]],
            ]
        )

        # Obtain total rotation matrix
        r_y_mult_r_z = _np.matmul(r_y, r_z)
        vertices_0_transformed = _np.matmul(r_y_mult_r_z, vertices_translated[0, :])
        vertices_1_transformed = _np.matmul(r_y_mult_r_z, vertices_translated[1, :])
        vertices_2_transformed = _np.matmul(r_y_mult_r_z, vertices_translated[2, :])
        boundary_point_transformed = _np.matmul(r_y_mult_r_z, boundary_point_translated)

        # Extract vertex coordinates in rotated coordinate system in x-y plane
        x = boundary_point_transformed[0]
        y = boundary_point_transformed[1]
        x0 = vertices_0_transformed[0]
        y0 = vertices_0_transformed[1]
        x1 = vertices_1_transformed[0]
        y1 = vertices_1_transformed[1]
        x2 = vertices_2_transformed[0]
        y2 = vertices_2_transformed[1]

        # Obtain local coordinates in orthonormal system for element
        transformation_matrix = _np.array([[x1 - x0, x2 - x0], [y1 - y0, y2 - y0]])
        transformation_matrix_inv = _np.linalg.inv(transformation_matrix)
        rhs = _np.vstack((x - x0, y - y0))
        local_coords[:, i] = _np.matmul(transformation_matrix_inv, rhs).transpose()

        # Required format for element and local coordinates for GridFunction.evaluate
        elem = list(grid.leaf_view.entity_iterator(0))[element_index[i]]
        coord = _np.array([[local_coords[0, i]], [local_coords[1, i]]])

        # Calculate pressure phase and magnitude at near point
        total_boundary_pressure[i] = grid_function.evaluate(elem, coord)

    return total_boundary_pressure
