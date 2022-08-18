"""Global initialization for Optimus."""

from . import geometry
from . import material
from . import model
from . import source
from . import utils
from . import postprocess


def _get_version():
    """Get version string."""
    from optimus import version

    return version.__version__


__version__ = _get_version()

global_parameters = utils.parameters.DefaultParameters()
