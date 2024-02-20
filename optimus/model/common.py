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


def _check_validity_formulation(
    formulation, formulation_parameters, preconditioner, preconditioner_parameters
):
    """
    Check if the specified formulation and preconditioner is a valid choice.

    Parameters
    ----------
    formulation : str
        The type of boundary integral formulation.
    formulation_parameters : dict
        The parameters for the boundary integral formulation.
    preconditioner : str
        The type of operator preconditioner.
    preconditioner_parameters : dict
        The parameters for the operator preconditioner.

    Returns
    -------
    formulation_name : str
        The name of the boundary integral formulation.
    preconditioner_name : str
        The name of the operator preconditioner.
    model_parameters : dict
        The parameters for the preconditioned boundary integral formulation.
    """

    if not isinstance(formulation, str):
        raise TypeError(
            "The boundary integral formulation needs to be specified as a string."
        )
    else:
        formulation_name = formulation.lower()

    if preconditioner is None:
        preconditioner_name = "none"
    elif not isinstance(preconditioner, str):
        raise TypeError("The preconditioner needs to be specified as a string.")
    else:
        preconditioner_name = preconditioner.lower()

    if formulation_parameters is None:
        formulation_parameters = {}
    elif not isinstance(formulation_parameters, dict):
        raise TypeError(
            "The parameters of the boundary integral formulation need to be "
            "specified as a dictionary."
        )

    if preconditioner_parameters is None:
        preconditioner_parameters = {}
    elif not isinstance(preconditioner_parameters, dict):
        raise TypeError(
            "The parameters of the preconditioner need to be specified as a dictionary."
        )

    if formulation_name not in ["pmchwt"]:
        raise NotImplementedError("Unknown boundary integral formulation type.")

    if preconditioner_name in ["none", "mass"]:
        prec_params = {}
    elif preconditioner_name == "osrc":
        prec_params = _process_osrc_parameters(preconditioner_parameters)
    else:
        raise NotImplementedError("Unknown preconditioner type.")

    model_parameters = {**formulation_parameters, **prec_params}

    return formulation_name, preconditioner_name, model_parameters


def _process_osrc_parameters(preconditioner_parameters):
    """
    Process the parameters for the OSRC preconditioner.

    If the OSRC parameter is not specified in the input,
    the global parameter is used.
    The OSRC parameters are:
        - npade: number of Padé expansions
        - theta: angle of the branch cut of the Padé series
        - damped_wavenumber: damped wavenumber
        - wavenumber: wavenumber

    Parameters
    ----------
    preconditioner_parameters : dict, None
        The parameters of the preconditioner.

    Returns
    -------
    osrc_parameters : dict
        The parameters of the OSRC preconditioner.
    """

    from optimus import global_parameters

    global_params_osrc = global_parameters.preconditioning.osrc

    if preconditioner_parameters is None:
        preconditioner_parameters = {}

    osrc_parameters = {}

    if "npade" in preconditioner_parameters:
        npade = preconditioner_parameters["npade"]
        if not (isinstance(npade, int) and npade > 0):
            raise TypeError(
                "The number of Padé expansions for the OSRC operator needs to be "
                "a positive integer."
            )
        osrc_parameters["osrc_npade"] = npade
    else:
        osrc_parameters["osrc_npade"] = global_params_osrc.npade

    if "theta" in preconditioner_parameters:
        theta = preconditioner_parameters["theta"]
        if not isinstance(theta, (int, float)):
            raise TypeError(
                "The angle of the branch cut of the Padé series for the "
                "OSRC operator needs to be a float."
            )
        osrc_parameters["osrc_theta"] = theta
    else:
        osrc_parameters["osrc_theta"] = global_params_osrc.theta

    if "damped_wavenumber" in preconditioner_parameters:
        k_damped = preconditioner_parameters["damped_wavenumber"]
        if not (isinstance(k_damped, (int, float, complex)) or k_damped is None):
            raise TypeError(
                "The damped wavenumber for the OSRC operators needs to be "
                "a complex number."
            )
        osrc_parameters["osrc_damped_wavenumber"] = k_damped
    else:
        osrc_parameters["osrc_damped_wavenumber"] = global_params_osrc.damped_wavenumber

    if "wavenumber" in preconditioner_parameters:
        k_osrc = preconditioner_parameters["wavenumber"]
        if not isinstance(k_osrc, (int, float, complex, str)):
            raise TypeError(
                "The wavenumber for the OSRC operators needs to be a complex number "
                "or a string."
            )
        elif isinstance(k_osrc, str):
            if k_osrc not in ["int", "ext"]:
                raise TypeError(
                    "The wavenumber for the OSRC operators needs to be a complex "
                    "number, or one of the labels 'int' and 'ext'."
                )
        osrc_parameters["osrc_wavenumber"] = k_osrc
    else:
        osrc_parameters["osrc_wavenumber"] = global_params_osrc.wavenumber

    return osrc_parameters
