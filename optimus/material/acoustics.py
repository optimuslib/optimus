"""Material properties."""

import numpy as _np
from .common import Material as _Material


def load_material(name):
    """
    Load an acoustic material with default parameters.

    Parameters
    ----------
    name : string
        The name of the material
    """

    if not isinstance(name, str):
        raise TypeError("Name of material needs to be specified as a string.")
    else:
        name = name.lower()

    if name == "water":
        return _Material(name, 1000, 1500)
    elif name == "fat":
        return _Material(name, 917, 1412)
    elif name == "bone":
        return _Material(name, 1912, 4080)
    else:
        raise ValueError("Unknown material type.")


def create_material(name, density, wavespeed):
    """
    Create an acoustic material with the specified parameters.

    Parameters
    ----------
    name : string
        The name of the material
    density : float
        The mass density
    wavespeed : float
        The speed of acoustic waves
    """

    if not isinstance(name, str):
        raise TypeError("Name of material needs to be specified as a string.")
    else:
        name = name.lower()

    if not isinstance(density, (float, int)):
        raise TypeError("The density needs to be specified as a number.")
    else:
        density = _np.float(density)

    if not isinstance(wavespeed, (float, int)):
        raise TypeError("The wavespeed needs to be specified as a number.")
    else:
        wavespeed = _np.float(wavespeed)

    return _Material(name, density, wavespeed)
