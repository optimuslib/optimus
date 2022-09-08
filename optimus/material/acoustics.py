import numpy as _np
from .common import Material as _Material
from .common import get_material_properties as _get_material_properties
from .common import write_material_database as _write_material_database


def load_material(name):
    """Load the physical properties of the specified material.

    Parameters
    ----------
    name : str, tuple str
        The name(s) of the material(s)

    Returns
    -------
    material : optimus.Material
        An (list of) optimus material object(s)
    """

    if isinstance(name, str):
        properties = _get_material_properties(name)
        return _Material(properties)
    elif isinstance(name, (list, tuple)):
        if all(isinstance(item, str) for item in name):
            properties = list(map(_get_material_properties, name))
            return list(map(_Material, properties))
        else:
            raise ValueError("All elements of the list must be strings.")
    else:
        raise TypeError(
            "Name of material must be specified as a string or a list of strings."
        )


def create_material(
    name,
    density,
    speed_of_sound,
    attenuation_coeff_a=0,
    attenuation_pow_b=0,
    save_to_file=False,
    **properties_user
):
    """Create an optimus material object with the specified parameters.

    Parameters
    ----------
    name : str
        The name of the material.
    density : float
        The mass density in kg/m^3
    speed_of_sound : float
        The speed of sound in m/s
    attenuation_coeff_a: float
        Attenuation coefficient in power law Np/m/MHz
        Default: 0 (no attenuation)
    attenuation_pow_b: float
        Attenuation power in power law, dimensionless
        Default: 0 (no attenuation)
    save_to_file: boolean
        Write the material to the user-defined database.
    **properties_user : dict
        A dictionary of additional material properties.

    Returns
    -------
    material : optimus.Material
        The optimus material object
    """

    list_args = [
        "name",
        "density",
        "speed_of_sound",
        "attenuation_coeff_a",
        "attenuation_pow_b",
    ]
    args_val = [
        name,
        density,
        speed_of_sound,
        attenuation_coeff_a,
        attenuation_pow_b,
    ]
    properties = dict((key, val) for key, val in zip(list_args, args_val))

    keys = list(properties.keys())
    keys.remove("name")

    if not isinstance(properties["name"], str):
        raise TypeError("Name of material needs to be specified as a string.")
    elif not all(isinstance(properties[key], (float, int)) for key in keys):
        raise TypeError("Material properties must be float/integer.")
    else:
        properties["name"] = properties["name"].lower()
        for key in keys:
            properties[key] = _np.float(properties[key])
        if save_to_file:
            _write_material_database(properties)

    if properties_user:
        import warnings

        warnings.warn(
            "Ignored material properties " + str(list(properties_user.keys())),
            RuntimeWarning,
        )

    return _Material(properties)
