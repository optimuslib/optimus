"""Set the global parameters."""

import numpy as _np


class DefaultParameters:
    def __init__(self):
        """
        Initialize the default global parameters
        """

        from bempp.api import global_parameters as bempp_parameters

        self.bempp_parameters = bempp_parameters

        self.verbosity = False

        self.linalg = LinalgParameters()
        self.preconditioning = PreconditioningParameters()

    def print(self):
        """
        Print all parameters.
        """

        print("Verbosity:", self.verbosity)
        print("")
        print("Linear algebra.")
        self.linalg.print(prefix=" ")
        print("")
        print("Preconditioning.")
        self.preconditioning.print(prefix=" ")


class LinalgParameters:
    def __init__(self):
        """
        Initialize the default parameters for linear algebra routines.
        """
        self.tol = 1e-5
        self.maxiter = 1000
        self.restart = 1000

    def print(self, prefix=""):
        """
        Print all parameters.
        """
        print(prefix + "Tolerance:", self.tol)
        print(prefix + "Maximum number of iterations:", self.maxiter)
        print(prefix + "Number of iterations before restart:", self.restart)


class PreconditioningParameters:
    def __init__(self):
        """
        Initialize the default parameters for the preconditioners.
        """
        self.osrc = OsrcParameters()

    def print(self, prefix=""):
        """
        Print all parameters.
        """
        print("")
        print(prefix + "OSRC preconditioner.")
        self.osrc.print(prefix=prefix + " ")


class OsrcParameters:
    def __init__(self):
        """
        Initialize the default parameters for the OSRC preconditioner.
        """
        self.npade = 4
        self.theta = _np.pi / 3
        self.wavenumber = "int"
        self.damped_wavenumber = None

    def print(self, prefix=""):
        """
        Print all parameters.
        """
        print(prefix + "Number of Padé expansion terms:", self.npade)
        print(prefix + "Branch cut angle for Padé series:", self.theta)
        print(prefix + "Wavenumber:", self.wavenumber)
        print(prefix + "Damped wavenumber:", self.damped_wavenumber)
