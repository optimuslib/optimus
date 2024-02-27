"""Preprocessing of preconditioned boundary integral formulations."""


def check_validity_exterior_formulation(
    formulation, formulation_parameters, preconditioner, preconditioner_parameters
):
    """
    Check if the specified formulation and preconditioner is a valid choice
    for exterior acoustic models.

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
        prec_params = process_osrc_parameters(preconditioner_parameters)
    else:
        raise NotImplementedError("Unknown preconditioner type.")

    model_parameters = {**formulation_parameters, **prec_params}

    return formulation_name, preconditioner_name, model_parameters


def check_validity_nested_formulation(
    formulation, preconditioner, n_interfaces
):
    """
    Check if the specified formulation and preconditioner is a valid choice
    for nested acoustic models.

    Available formulations:
        - 'pmchwt'
        - 'muller'
        - 'multitrace'
        - 'none' (unbounded exterior surface)

    Available preconditioners:
        - 'none'
        - 'mass'
        - 'osrc'
        - 'calderon'

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

    formulations = tuple([clean_string_names(form) for form in formulations])
    preconditioners = tuple([clean_string_names(prec) for prec in preconditioners])

    for form in formulations[1:]:
        if form not in ("pmchwt", "muller", "multitrace"):
            raise ValueError(
                "The formulation must be one of:"
                " - 'pmchwt'"
                " - 'muller'"
                " - 'multitrace'"
            )

    for prec in preconditioners[1:]:
        if prec not in ("none", "mass", "osrc", "calderon"):
            raise ValueError(
                "The preconditioner must be one of: "
                " - 'none'"
                " - 'mass'"
                " - 'osrc'"
                " - 'calderon'"
            )

    # Check the consistency of the set of preconditioner and formulation types.

    weak = ("none",)
    strong = ("mass", "osrc", "calderon")
    weak_preconditioners = [prec in weak for prec in preconditioners[1:]]
    strong_preconditioners = [prec in strong for prec in preconditioners[1:]]
    if not (all(weak_preconditioners) or all(strong_preconditioners)):
        raise NotImplementedError(
            "The preconditioner must be the same weak/strong discretisation type."
        )

    for form, prec in zip(formulations, preconditioners):
        if prec == "osrc" and form not in ("pmchwt", "multitrace"):
            raise ValueError(
                "The OSRC preconditioner only works for the PMCHWT and "
                "multitrace formulations."
            )

    for form, prec in zip(formulations, preconditioners):
        if prec == "calderon" and form not in ("pmchwt", "multitrace"):
            raise ValueError(
                "The Calderón preconditioner only works for the PMCHWT and "
                "multitrace formulations."
            )

    return formulations, preconditioners


def assign_representation(formulations):
    """
    Assign the type of representation formula according to the
    boundary integral formulation name.

    Parameters
    ----------
    formulations : list[str]
        The names of the boundary integral formulation.

    Returns
    -------
    representations : list[str]
        The names of the representation formula:
         - 'none' for inactive subdomains
         - 'direct' for the PMCHWT and Müller formulations
    """

    representations = []
    for form in formulations:
        if form == "none":
            representations.append("none")
        elif form in ("pmchwt", "muller", "multitrace"):
            representations.append("direct")
        else:
            raise ValueError("Unknown formulation: " + form + ".")

    return representations


def process_osrc_parameters(preconditioner_parameters):
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


def process_calderon_parameters(preconditioner_parameters):
    """
    Process the parameters for the Calderón preconditioner.

    Default values are taken from the global parameters.
    The Calderón parameters are:
        - domain: domain of the Calderón operator (exterior or interior)

    Parameters
    ----------
    preconditioner_parameters : dict, None
        The parameters of the preconditioner.

    Returns
    -------
    calderon_parameters : dict
        The parameters of the Calderón preconditioner.
    """

    from optimus import global_parameters

    global_params_calderon = global_parameters.preconditioning.calderon

    if preconditioner_parameters is None:
        preconditioner_parameters = {}

    calderon_parameters = {}

    if "domain" in preconditioner_parameters:
        domain = preconditioner_parameters["domain"]
        if not isinstance(domain, str):
            raise TypeError(
                "The domain of the Calderón preconditioner needs to be a string."
            )
        elif domain not in ("exterior", "interior"):
            raise ValueError(
                "The domain of the Calderón preconditioner needs to be either "
                "'exterior' or 'interior'."
            )
        else:
            calderon_parameters["domain"] = domain
    else:
        calderon_parameters["domain"] = global_params_calderon.domain

    return calderon_parameters


def check_sources(frequency, subdomain_nodes):
    """
    Check validity of frequency and sources in nested domains.

    Perform the following checks:
        - The frequency must be a positive float.
        - All sources must have the same frequency.

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
                "frequency {} instead.".format(freq, source.label, source.frequency)
            )

    return freq


def clean_string_names(name):
    """Clean up syntax of string names."""
    clean_name = name.lower().replace("ü", "u").replace("ó", "o").replace("-", "")
    return clean_name
