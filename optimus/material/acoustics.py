"""Material properties."""

import numpy as _np
from .common import Material as _Material
from .common import get_material_properties as _get_material_properties
from .common import write_material_database as _write_material_database


def load_material(name):
    """
    Load an acoustic material with default parameters.

    Parameters
    ----------
    name : string or a list of strings
        The name of the material(s)
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


def create_material(properties, save_to_file=False):
    """
    Create an acoustic material with the specified parameters.

    Input argument
    ----------
    properties : dict
        A dictionary of the material properties with the keys like parameters below.

    Parameters
    ----------
    density : float
        The mass density in [kg/m3]
    speed_of_sound : float
        The speed of sound in [m/s]
    attenuation_coeff_a: float
        Attenuation coefficient in power law [Np/m/MHz]
    attenuation_pow_b: float
        Attenuation power in power law [dimensionless]
    """

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
