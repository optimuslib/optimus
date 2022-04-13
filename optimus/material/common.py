"""Common functionality for acoustic materials."""

import numpy as _np
import pandas as pd
import os


def read_database(file_name="Material_database.xls", sheet="default"):
    datadir = os.path.dirname(__file__)
    database_file = os.path.join(datadir, file_name)
    dataframe = pd.read_excel(database_file, sheet_name=sheet, header=[0, 1])
    return dataframe


def get_material_database(*name):
    """
    Load the material database from an xls file.

    Input arguments
    ----------
    None

    Output arguments
    ----------
    dataframe: pandas dataframe object

    """
    dataframe = read_database()

    if len(name):
        name = name[0].lower()
        data_mask = dataframe[("Tissue", "Name")].str.lower().isin([name])
        if not data_mask.any():
            raise ValueError(
                "the material: \033[1m" + name + "\033[0m  is not in the database."
            )
        else:
            material = dataframe.loc[data_mask]

        density = material[("Density (kg/m3)", "Average")].get_values()[0]
        speed_of_sound = material[("Speed of Sound [m/s]", "Average")].get_values()[0]
        attenuation_coeff_a = material[
            ("Attenuation Constant", "a [Np/m/MHz]")
        ].get_values()[0]
        attenuation_pow_b = material[("Attenuation Constant", "b")].get_values()[0]
        output = {
            "name": name,
            "density": density,
            "speed_of_sound": speed_of_sound,
            "attenuation_coeff_a": attenuation_coeff_a,
            "attenuation_pow_b": attenuation_pow_b,
        }
    else:
        output = dataframe
    return output


def write_material_database(properties):
    dataframe = read_database(
        file_name="Material_database-user-defined.xls", sheet="user-defined"
    )
    dataframe.append(
        {
            ("Tissue", "Name"): properties["name"],
            ("Density (kg/m3)", "Average"): properties["density"],
            ("Speed of Sound [m/s]", "Average"): properties["speed_of_sound"],
            ("Attenuation Constant", "a [Np/m/MHz]"): properties["attenuation_coeff_a"],
            ("Attenuation Constant", "b"): properties["attenuation_pow_b"],
        },
        ignore_index=True,
    )

    datadir = os.path.dirname(__file__)
    database_file = os.path.join(datadir, "Material_database-user-defined.xls")
    dataframe.to_excel(
        database_file, sheet_name="user-defined", startrow=3, header=False
    )


class Material:
    def __init__(self, properties):
        """
        Physical properties of a material.

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

        # name = name.lower()
        # data_mask = dataframe[("Tissue", "Name")].str.lower().isin([name])
        # if not data_mask.any():
        #     raise ValueError(
        #         "the material: \033[1m" + name + "\033[0m  is not in the database."
        #     )
        # else:
        #     material = dataframe.loc[data_mask]

        self.name = properties["name"]
        self.density = properties["density"]
        self.speed_of_sound = properties["speed_of_sound"]
        self.attenuation_coeff_a = properties["attenuation_coeff_a"]
        self.attenuation_pow_b = properties["attenuation_pow_b"]

    def compute_wavenumber(self, frequency):
        """Calculate the wavenumber for the specified frequency."""
        return (
            2 * _np.pi * frequency / self.speed_of_sound
            + 1j * self.compute_attenuation(frequency)
        )

    def compute_wavelength(self, frequency):
        """Calculate the wave length for the specified frequency."""
        return self.speed_of_sound / frequency

    def compute_attenuation(self, frequency):
        """Calculate the power law attenuation coefficient (alpha) for the specified frequency."""
        return self.attenuation_coeff_a * (frequency * 1e-6) ** self.attenuation_pow_b
