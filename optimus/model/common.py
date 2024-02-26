"""Common functionality for the models."""

import bempp.api as _bempp
import numpy as _np


class Model:
    def __init__(
        self,
        label="model",
    ):
        """
        Base class for wave propagation models.
        """

        self.label = label
        self.formulation = None
        self.solution = None
        self.frequency = None

        return


class ExteriorModel(Model):
    def __init__(
        self,
        source,
        geometry,
        material_exterior,
        material_interior,
        formulation,
        preconditioner,
        parameters,
        label="exterior_model",
    ):
        """
        Base class for discrete exterior wave propagation models.

        The geometry has one unbounded exterior domain and may have multiple
        bounded subdomains, which are assumed to be disjoint.

        Parameters
        ----------
        source : optimus.source.common.Source
            The Optimus representation of a source field.
        geometry : list of optimus.geometry.common.Geometry
            The list of geometries, in Optimus representation that includes the grid of
            the scatterers.
        material_exterior : optimus.material.common.Material
            The Optimus representation of the material for the unbounded
            exterior region.
        material_interior : list of optimus.material.common.Material
            The Optimus representation of the material for the bounded
            scatterers.
        formulation : str
            The type of boundary integral formulation.
        preconditioner : str
            The type of operator preconditioner.
        parameters : dict
            The parameters for the formulation and preconditioner.
        label : str
            The label of the model.
        """

        super().__init__(label)

        self.source = source
        self.frequency = source.frequency
        self.material_exterior = material_exterior
        (
            self.n_subdomains,
            self.geometry,
            self.material_interior,
        ) = self._preprocess_domains(geometry, material_interior)

        self.formulation = formulation
        self.preconditioner = preconditioner
        self.parameters = parameters

        self.space = None
        self.continous_operator = None
        self.discrete_operator = None
        self.discrete_preconditioner = None
        self.rhs_vector = None
        self.lhs_discrete_system = None
        self.rhs_discrete_system = None
        self.solution_vector = None

        self.iteration_count = None

        return

    def solve(self):
        """
        Solve the model.

        Needs to be overwritten by specific model.
        """

        raise NotImplementedError

    @staticmethod
    def _preprocess_domains(geometry, material_interior):
        """
        Preprocess the input variables for the geometry and materials.

        Parameters
        ----------
        geometry : list of optimus.geometry.common.Geometry
            The list of geometries, in Optimus representation that includes the grid of
            the scatterers.
        material_interior : list of optimus.material.common.Material
            The Optimus representation of the material for the bounded
            scatterers.

        Returns
        -------
        n_bounded_domains : int
            The number of bounded subdomains.
        geometries : tuple[optimus.geometry.common.Geometry]
            The list of geometries, in Optimus representation that includes the grid of
            the scatterers.
        materials_interior : tuple[optimus.material.common.Material]
            The Optimus representation of the material for the bounded
            scatterers.
        """

        from optimus.geometry.common import Geometry
        from optimus.material.common import Material

        if isinstance(geometry, tuple):
            geometries = geometry
        elif isinstance(geometry, list):
            geometries = tuple(geometry)
        else:
            geometries = (geometry,)
        for subdomain in geometries:
            if not isinstance(subdomain, Geometry):
                raise TypeError(
                    "The subdomain needs to be specified as an Optimus Geometry object."
                )
        n_bounded_domains = len(geometries)

        if isinstance(material_interior, tuple):
            materials_interior = material_interior
        elif isinstance(material_interior, list):
            materials_interior = tuple(material_interior)
        else:
            materials_interior = (material_interior,)
        for material in materials_interior:
            if not isinstance(material, Material):
                raise TypeError(
                    "The material needs to be specified as an Optimus Material object."
                )
        if len(materials_interior) != n_bounded_domains:
            raise ValueError(
                "The number of geometries and interior materials should be the same."
            )

        return n_bounded_domains, geometries, materials_interior


class GraphModel(Model):
    def __init__(
        self,
        topology,
        label="graph_model",
    ):
        """
        Base class for wave propagation models for graph domains.

        The geometry has one unbounded exterior domain and may have multiple
        bounded subdomains, which are assumed to form a graph topology, i.e,
        are nested domains without junctions.

        Parameters
        ----------
        topology : optimus.geometry.Graph
            The graph topology representing the geometry.
        label : str
            The label of the model.
        """

        super().__init__(label)

        self.topology = self._check_topology(topology)

        return

    @staticmethod
    def _check_topology(topology):
        """
        Check the validity of the topology.

        Parameters
        ----------
        topology : optimus.geometry.Graph
            The graph topology representing the geometry.

        Returns
        -------
        topology : optimus.geometry.Graph
            The graph topology representing the geometry.
        """

        from optimus.geometry import Graph

        if not isinstance(topology, Graph):
            raise TypeError(
                "The topology needs to be specified as an Optimus Graph object."
            )

        # Check if all interface nodes have a geometry
        for node in topology.interface_nodes:
            if node.is_active() and node.bounded and node.geometry is None:
                raise AttributeError(
                    "The interface node " + node.label + " needs to have a geometry."
                )

        return topology


def _vector_to_gridfunction(vector, spaces):
    """
    Convert an array with values representing the coefficients
    of one or several discrete spaces into Bempp grid functions.

    Parameters
    ----------
    vector : numpy.ndarray
        Vector of coefficients.
    spaces : tuple[bempp.api.FunctionSpace], list[bempp.api.FunctionSpace]
        The function spaces.

    Returns
    -------
    gridfunctions : tuple[bempp.api.GridFunction]
        The list of grid functions.
    """

    ndofs = [space.global_dof_count for space in spaces]
    partitioning = _np.cumsum(ndofs)
    subvectors = _np.split(vector, partitioning[:-1])
    gridfunctions = [
        _bempp.GridFunction(space, coefficients=vec)
        for vec, space in zip(subvectors, spaces)
    ]

    return gridfunctions
