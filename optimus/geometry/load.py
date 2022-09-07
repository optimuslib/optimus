"""Load geometry shapes."""

import bempp.api as _bempp
from .common import ImportedGeometry as _ImportedGeometry


def import_grid(filename, label="imported_grid"):
    """Import a grid from a file as geometry.

    Parameters
    ----------
    filename : str
        The full file name, including path and extension.
    label : str
        The label of the goemetry.

    Returns
    -------
    geometry : optimus.geometry.Geometry
        The geometry based on the imported grid.
    """
    grid = _bempp.import_grid(filename)
    return _ImportedGeometry(grid, label, filename)


def bempp_grid(grid, label="bempp_grid"):
    """Import a BEMPP grid as geometry.

    Parameters
    ----------
    grid : bempp.api.Grid
        The BEMPP grid.
    label : str
        The label of the goemetry.

    Returns
    -------
    geometry : optimus.geometry.Geometry
        The geometry based on the imported grid.
    """

    return _ImportedGeometry(grid, label)
