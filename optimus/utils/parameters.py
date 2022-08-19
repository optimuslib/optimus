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
        self.postprocessing = PostProcessingParameters()

    def print(self):
        """
        Print all parameters.
        """
        from .generic import bold_ul_red_text

        print("\n" + bold_ul_red_text("Verbosity parameter:"), self.verbosity)
        print("\n" + bold_ul_red_text("Linear algebra parameters:"))
        self.linalg.print(prefix=" ")
        print("\n" + bold_ul_red_text("Preconditioning parameters:"))
        self.preconditioning.print(prefix=" ")
        print("\n" + bold_ul_red_text("Postprocessing parameters:"))
        self.postprocessing.print(prefix=" ")


class LinalgParameters:
    def __init__(self):
        """
        Initialize the default parameters for linear algebra routines.
        """
        self.linsolver = "gmres"
        self.tol = 1e-5
        self.maxiter = 1000
        self.restart = 1000

    def print(self, prefix=""):
        """
        Print all parameters.
        """
        print(prefix + "Linear solver:", self.linsolver)
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


class PostProcessingParameters:
    def __init__(self):
        """
        Initialize the default parameters for postprocessing routines.
        """
        self.hmat_eps = 1.0e-8
        self.hmat_max_rank = 10000
        self.hmat_max_block_size = 10000
        self.assembly_type = "h-matrix"
        self.solid_angle_tolerance = 0.1
        self.quadrature_order = 4
        self.concave_hull_alpha = 0.1

    def print(self, prefix=""):
        """
        Print all parameters.
        """
        print(prefix + "Potential operator assembly type is: ", self.assembly_type)
        if self.assembly_type.lower() in [
            "h-matrix",
            "hmat",
            "h-mat",
            "h_mat",
            "h_matrix",
        ]:
            print(
                prefix + "H-matrix epsilon for postprocessing operators:", self.hmat_eps
            )
            print(
                prefix + "H-matrix maximum rank for postprocessing operators:",
                self.hmat_max_rank,
            )
            print(
                prefix + "H-matrix maximum block size for postprocessing operators:",
                self.hmat_max_block_size,
            )

        print(prefix + "Solid angle tolerance is: ", self.solid_angle_tolerance)
        print(
            prefix + "Concave Hull threshold parameter (alpha) is: ",
            self.concave_hull_alpha,
        )
