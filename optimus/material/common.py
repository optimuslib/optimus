"""Common functionality for acoustic materials."""

import numpy as _np
import pandas as pd
import os


def get_excel_database(
    database="default", header_format=(0, 1), index_col=None
):
    """
    Read excel database file as a pandas dataframe

    Input
    ----------
    database: string
        The type of the database to be loaded. Accepted values are:
        'default': to load default database,
        'user-defined': to load user-defined database.
    sheet_name: string
        The excel sheet to be loaded
    header_format: tuple of numbers
        The header format of data table in the sheet

    Output
    ----------
    dataframe: pandas dataframe object

    """
    if database.lower() == "default":
        file_name = "Material_database.xls"
    elif database.lower() == "user-defined":
        file_name = "Material_database_user-defined.xls"
    else:
        raise ValueError("undefined database.")

    datadir = os.path.dirname(__file__)
    database_file = os.path.join(datadir, file_name)
    dataframe = pd.read_excel(
        database_file, header=header_format, index_col=index_col
    )
    return dataframe


def get_material_properties(name):
    """
    Extract material properties from all databases.

    Input
    ----------
    name : string
        The name of material in the database.

    Output
    ----------
    dictionary of material properties

    """
    if not isinstance(name, str):
        raise TypeError("Name of material should be a string.")
    else:
        name = name.lower()

    dataframe_default = get_excel_database(database="default")
    dataframe_user = get_excel_database(database="user-defined")
    dataframe = pd.concat(
        [dataframe_default, dataframe_user], axis=0, sort=False
    )

    data_mask = dataframe[("Tissue", "Name")].str.lower().isin([name])
    if not data_mask.any():
        raise ValueError(
            "the material: \033[1m" + name + "\033[0m  is not in the database."
        )
    else:
        material = dataframe.loc[data_mask]
        density = material[("Density (kg/m3)", "Average")].get_values()[0]
        speed_of_sound = material[
            ("Speed of Sound [m/s]", "Average")
        ].get_values()[0]
        attenuation_coeff_a = material[
            ("Attenuation Constant", "a [Np/m/MHz]")
        ].get_values()[0]
        attenuation_pow_b = material[
            ("Attenuation Constant", "b")
        ].get_values()[0]
        output = {
            "name": name,
            "density": density,
            "speed_of_sound": speed_of_sound,
            "attenuation_coeff_a": attenuation_coeff_a,
            "attenuation_pow_b": attenuation_pow_b,
        }
        return output


def write_material_database(properties):

    """
    Write a pandas dataframe of user-defined properties to the user-defined
    material database Excel file.

    Input
    ----------
    properties: dict
        The dictionary of material properties defined by the user

    Output
    ----------
    None

    """

    user_database_file = "Material_database_user-defined.xls"
    dataframe_default = get_excel_database(database="default")
    dataframe_user = get_excel_database(
        database="user-defined", header_format=[0, 1], index_col=0
    )
    dataframe_both = pd.concat(
        [dataframe_default, dataframe_user], axis=0, sort=False
    )

    name = properties["name"].lower()
    data_mask = dataframe_both[("Tissue", "Name")].str.lower().isin([name])
    if data_mask.any():
        raise ValueError(
            "A material with the name: \033[1m"
            + name
            + "\033[0m  already EXISTs in the database files "
            + "(either default or user-defined)."
        )
    else:
        cols = [
            ("Tissue", "Name"),
            ("Density (kg/m3)", "Average"),
            ("Speed of Sound [m/s]", "Average"),
            ("Nonlinearity Prameter B/A", "Average"),
            ("Attenuation Constant", "a [Np/m/MHz]"),
            ("Attenuation Constant", "b"),
            ("Heat Capacity (J/kg/°C)", "Average"),
            ("Thermal Conductivity (W/m/°C)", "Average"),
            ("Heat Transfer Rate (ml/min/kg)", "Average"),
            ("Heat Generation Rate (W/kg)", "Average"),
        ]

        dataframe_tmp = pd.DataFrame(
            {
                ("Tissue", "Name"): properties["name"],
                ("Density (kg/m3)", "Average"): properties["density"],
                ("Speed of Sound [m/s]", "Average"): properties[
                    "speed_of_sound"
                ],
                ("Attenuation Constant", "a [Np/m/MHz]"): properties[
                    "attenuation_coeff_a"
                ],
                ("Attenuation Constant", "b"): properties["attenuation_pow_b"],
            },
            index=[0],
            columns=cols,
        )

        datadir = os.path.dirname(__file__)
        database_file = os.path.join(datadir, user_database_file)
        writer = pd.ExcelWriter(database_file, engine="xlsxwriter")
        dataframe_user.append(
            dataframe_tmp, ignore_index=True, sort=False
        ).to_excel(writer, sheet_name="user-defined")
        writer.save()


class Material:
    def __init__(self, properties):
        """
        Object for the physical properties of a material.

        Input argument
        ----------
        properties : dict
            A dictionary of the material properties with the keys
            as defined below.

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

        self.name = properties["name"]
        self.density = properties["density"]
        self.speed_of_sound = properties["speed_of_sound"]
        self.attenuation_coeff_a = properties["attenuation_coeff_a"]
        self.attenuation_pow_b = properties["attenuation_pow_b"]
        self.properties = properties

    def compute_wavenumber(self, frequency):
        """
        Calculate the wavenumber for the specified frequency.

        Input
        ----------
        frequency: float/int
            The wave frequency

        Output
        ----------
        complex wavenumber
        """
        return (
            2 * _np.pi * frequency / self.speed_of_sound
            + 1j * self.compute_attenuation(frequency)
        )

    def compute_wavelength(self, frequency):
        """
        Calculate the wave length for the specified frequency.

        Input
        ----------
        frequency: float/int
            The wave frequency

        Output
        ----------
        wavelength
        """
        return self.speed_of_sound / frequency

    def compute_attenuation(self, frequency):
        """
        Calculate the power law attenuation coefficient (alpha) for the
        specified frequency.

        Input
        ----------
        frequency: float/int
            The wave frequency

        Output
        ----------
        attenuation coefficient
        """
        return (
            self.attenuation_coeff_a
            * (frequency * 1e-6) ** self.attenuation_pow_b
        )

    def print(self):
        cols = [
            "name",
            "density",
            "speed_of_sound",
            "attenuation_coeff_a",
            "attenuation_pow_b",
        ]
        print(
            pd.DataFrame(self.properties, columns=cols, index=[0]).to_string(
                index=False
            )
        )
