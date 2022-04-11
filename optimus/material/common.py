"""Common functionality for acoustic materials."""

import numpy as _np


class Material:
    def __init__(self, name, density, wavespeed, attenuation=None):
        """
        Physical properties of a material.

        Parameters
        ----------
        name : string
            The name of the material
        density : float
            The mass density
        wavespeed : float
            The speed of acoustic waves
        """

        self.name = name
        self.density = density
        self.wavespeed = wavespeed
        self.attenuation = attenuation

    def wavenumber(self, frequency):
        """Calculate the wavenumber for the specified frequency."""
        return 2 * _np.pi * frequency / self.wavespeed

    def wavelength(self, frequency):
        """Calculate the wave length for the specified frequency."""
        return self.wavespeed / frequency
