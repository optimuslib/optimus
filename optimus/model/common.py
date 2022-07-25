"""Common functionality for the models."""

import bempp.api as _bempp
import numpy as _np


class Model:
    def __init__(
        self,
        source,
        geometry,
        material_exterior,
        material_interior,
        formulation,
        preconditioner,
    ):
        """
        Base class for BEM models.
        """

        self.source = source
        self.geometry = geometry
        self.material_exterior = material_exterior
        self.material_interior = material_interior
        self.n_subdomains = self._preprocess_domains()

        self.formulation = formulation
        self.preconditioner = preconditioner

    def solve(self):
        """
        Solve the model.
        Needs to be overwritten by specific model.
        """
        raise NotImplementedError

    def _preprocess_domains(self):
        """
        Preprocess the input variables for the geometry and materials.

        Returns
        ----------
        n_bounded_domains : int
            The number of bounded subdomains.
        """

        from optimus.geometry.common import Geometry
        from optimus.material.common import Material

        if not isinstance(self.geometry, (list, tuple)):
            self.geometry = (self.geometry,)
        for subdomain in self.geometry:
            if not isinstance(subdomain, Geometry):
                raise TypeError(
                    "The subdomain needs to be specified as an Optimus Geometry object."
                )
        n_bounded_domains = len(self.geometry)

        if not isinstance(self.material_interior, (list, tuple)):
            self.material_interior = (self.material_interior,)
        for material in self.material_interior:
            if not isinstance(material, Material):
                raise TypeError(
                    "The material needs to be specified as an Optimus Material object."
                )
        if len(self.material_interior) != n_bounded_domains:
            raise ValueError(
                "The number of geometries and interior materials should be the same."
            )

        return n_bounded_domains


def _vector_to_gridfunction(vector, spaces):
    """
    Convert an array with values representing the coefficients
    of one or several discrete spaces into Bempp grid functions.

    Parameters
    ----------
    vector : np.ndarray
        Vector of coefficients.
    spaces : tuple of bempp.api.FunctionSpace
        The function spaces.

    Returns
    ----------
    gridfunctions : tuple of bempp.api.GridFunction
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
    ----------
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

    Parameters
    ----------
    preconditioner_parameters : dict
        The parameters of the preconditioner.

    Returns
    -------
    osrc_parameters : dict
        The parameters of the OSRC preconditioner.
    """

    import optimus

    global_params_osrc = optimus.global_parameters.preconditioning.osrc

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
