"""Material properties."""

import numpy as _np
from .common import Material as _Material
from .common import get_material_properties as _get_material_properties
from .common import write_material_database as _write_material_database


def load_material(name):
    """
    Load the physical properties of the specified material.

    Input
    ----------
    name : string or a list of strings
        The name of the material(s)
    Output
    ----------
    Material object: an(list of) optimus material object(s)
    """

    if isinstance(name, str):
        properties = _get_material_properties(name)
        return _Material(properties)
    elif isinstance(name, list):
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
    density=0,
    speed_of_sound=0,
    attenuation_coeff_a=0,
    attenuation_pow_b=0,
    save_to_file=False,
    **properties_user
):
    """
    Create an optimus material object with the specified parameters.

    Input argument
    ----------
    density : float
        The mass density in [kg/m3]
    speed_of_sound : float
        The speed of sound in [m/s]
    attenuation_coeff_a: float
        Attenuation coefficient in power law [Np/m/MHz]
    attenuation_pow_b: float
        Attenuation power in power law [dimensionless]
    save_to_file: boolean
        to write the user-defined material to the user-defined database file or not.
    **properties_user : dict
        A dictionary of the material properties with the keys like input arguments, see below.

    Output
    ----------
    Material object: optimus material object
    """
    if not len(properties_user):
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

    return _Material(properties)
