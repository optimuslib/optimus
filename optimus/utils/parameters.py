import numpy as _np


class DefaultParameters:
    def __init__(self):
        """Initialize the default global parameters"""

        self.verbosity = False

        self.bem = BemParameters()
        self.linalg = LinalgParameters()
        self.incident_field_parallelisation = (
            IncidentFieldParallelProcessingParameters()
        )
        self.preconditioning = PreconditioningParameters()
        self.postprocessing = PostProcessingParameters()

    def print(self):
        """Print all parameters."""

        from .generic import bold_ul_red_text

        print("\n" + bold_ul_red_text("Verbosity parameter:"), self.verbosity)
        print("\n" + bold_ul_red_text("BEM parameters:"))
        self.bem.print(prefix=" ")
        print("\n" + bold_ul_red_text("Linear algebra parameters:"))
        self.linalg.print(prefix=" ")
        print("\n" + bold_ul_red_text("Incident field parallel processing parameters:"))
        self.incident_field_parallelisation.print(prefix=" ")
        print("\n" + bold_ul_red_text("Preconditioning parameters:"))
        self.preconditioning.print(prefix=" ")
        print("\n" + bold_ul_red_text("Postprocessing parameters:"))
        self.postprocessing.print(prefix=" ")


class LinalgParameters:
    def __init__(self):
        """Initialize the default parameters for linear algebra routines."""

        self.linsolver = "gmres"
        self.tol = 1e-5
        self.maxiter = 1000
        self.restart = 1000

    def print(self, prefix=""):
        """Print all parameters."""

        print(prefix + "Linear solver:", self.linsolver)
        print(prefix + "Tolerance:", self.tol)
        print(prefix + "Maximum number of iterations:", self.maxiter)
        print(prefix + "Number of iterations before restart:", self.restart)


class IncidentFieldParallelProcessingParameters:
    def __init__(self):
        """Initialize the default parameters for incident field parallelisation."""

        import multiprocessing as _mp

        self.cpu_count = _mp.cpu_count()
        self.mem_per_core = 1.108895e8
        self.parallelisation_method = "numba"

    def print(self, prefix=""):
        """Print all parameters."""

        print(prefix + "Parallelisation method:", self.parallelisation_method)
        if self.parallelisation_method.lower() in [
            "multiprocessing",
            "mp",
            "multi-processing",
        ]:
            print(prefix + "Number of CPU used in parallelisation:", self.cpu_count)
            print(
                prefix + "Memory allocation per core [MB]:",
                int(self.mem_per_core / 1e6),
            )


class PreconditioningParameters:
    def __init__(self):
        """Initialize the default parameters for the preconditioners."""

        self.osrc = OsrcParameters()
        self.calderon = CalderonParameters()

    def print(self, prefix=""):
        """Print all parameters."""

        print(prefix + "OSRC preconditioner.")
        self.osrc.print(prefix=prefix + " ")

        print(prefix + "Calderón preconditioner.")
        self.calderon.print(prefix=prefix + " ")


class OsrcParameters:
    def __init__(self):
        """Initialize the default parameters for the OSRC preconditioner."""

        self.npade = 4
        self.theta = _np.pi / 3
        self.wavenumber = "int"
        self.damped_wavenumber = None

    def print(self, prefix=""):
        """Print all parameters."""

        print(prefix + "Number of Padé expansion terms:", self.npade)
        print(prefix + "Branch cut angle for Padé series:", self.theta)
        print(prefix + "Wavenumber:", self.wavenumber)
        print(prefix + "Damped wavenumber:", self.damped_wavenumber)


class CalderonParameters:
    def __init__(self):
        """Initialize the default parameters for the Calderón preconditioner."""

        self.domain = "exterior"

    def print(self, prefix=""):
        """Print all parameters."""

        print(prefix + "Domain for operators:", self.domain)


class PostProcessingParameters:
    def __init__(self):
        """Initialize the default parameters for postprocessing routines."""

        self.solid_angle_tolerance = 0.1

    def print(self, prefix=""):
        """Print all parameters."""

        print(prefix + "Solid angle tolerance is: ", self.solid_angle_tolerance)


class BemParameters:
    def __init__(self):
        """Initialize the default parameters for BEM routines."""

        from bempp.api import global_parameters as bempp_parameters

        self.bempp_parameters = bempp_parameters

        self._matrix_assembly_type = "hmat"
        self._field_assembly_type = "dense"

        self._matrix_hmat_eps = 1.0e-3
        self._matrix_hmat_max_rank = 30
        self._matrix_hmat_max_block_size = 1000000

        self._field_hmat_eps = 1.0e-8
        self._field_hmat_max_rank = 10000
        self._field_hmat_max_block_size = 10000

        self.update_hmat_parameters("boundary")

    def print(self, prefix=""):
        """Print all parameters."""

        print(prefix + "Matrix assembly type: ", self._matrix_assembly_type)
        if self._matrix_assembly_type == "hmat":
            print(
                prefix + " H-matrix epsilon for boundary operators:",
                self._matrix_hmat_eps,
            )
            print(
                prefix + " H-matrix maximum rank for boundary operators:",
                self._matrix_hmat_max_rank,
            )
            print(
                prefix + " H-matrix maximum block size for boundary operators:",
                self._matrix_hmat_max_block_size,
            )

        print(prefix + "Field assembly type: ", self._field_assembly_type)
        if self._field_assembly_type == "hmat":
            print(
                prefix + " H-matrix epsilon for potential operators:",
                self._field_hmat_eps,
            )
            print(
                prefix + " H-matrix maximum rank for potential operators:",
                self._field_hmat_max_rank,
            )
            print(
                prefix + " H-matrix maximum block size for potential operators:",
                self._field_hmat_max_block_size,
            )

        print(prefix + "Numerical quadrature order.")
        print(prefix + " Double integration for boundary operators in the matrix.")
        print(
            prefix + "  Self interaction:",
            self.bempp_parameters.quadrature.double_singular,
        )
        print(
            prefix + "  Near interaction:",
            self.bempp_parameters.quadrature.near.double_order,
        )
        print(
            prefix + "  Medium interaction:",
            self.bempp_parameters.quadrature.medium.double_order,
        )
        print(
            prefix + "  Far interaction:",
            self.bempp_parameters.quadrature.far.double_order,
        )
        print(prefix + " Single integration for potential operators for the field.")
        print(
            prefix + "  Near interaction:",
            self.bempp_parameters.quadrature.near.single_order,
        )
        print(
            prefix + "  Medium interaction:",
            self.bempp_parameters.quadrature.medium.single_order,
        )
        print(
            prefix + "  Far interaction:",
            self.bempp_parameters.quadrature.far.single_order,
        )

    def print_current_hmat_parameters(self, prefix=""):
        """Print the current H-matrix parameters."""

        print(prefix + " H-matrix epsilon:", self.bempp_parameters.hmat.eps)
        print(
            prefix + " H-matrix maximum rank:",
            self.bempp_parameters.hmat.max_rank,
        )
        print(
            prefix + " H-matrix maximum block size:",
            self.bempp_parameters.hmat.max_block_size,
        )

    def set_matrix_assembly_type(self, assembly_type):
        """
        Set the assembly type of the boundary integral operators in BEMPP.

        Parameters
        ----------
        assembly_type : str
            The type of operator assembly: either "dense" or "hmat"
        """
        assembly_type = self._process_assembly_type(assembly_type)
        self._matrix_assembly_type = assembly_type
        self.bempp_parameters.assembly.boundary_operator_assembly_type = assembly_type

    def set_field_assembly_type(self, assembly_type):
        """
        Set the assembly type of the potential integral operators in BEMPP.

        Parameters
        ----------
        assembly_type : str
            The type of operator assembly: either "dense" or "hmat"
        """
        assembly_type = self._process_assembly_type(assembly_type)
        self._field_assembly_type = assembly_type
        self.bempp_parameters.assembly.potential_operator_assembly_type = assembly_type

    @staticmethod
    def _process_assembly_type(assembly_type):
        """Process assembly type input"""
        if assembly_type.lower() in (
            "hmat",
            "h-mat",
            "h_mat",
            "h-matrix",
            "h_matrix",
        ):
            return "hmat"
        elif assembly_type.lower() == "dense":
            return "dense"
        else:
            raise ValueError("Assembly type has to be 'dense' or 'hmat'.")

    def set_matrix_hmat(self, eps=None, max_rank=None, max_block_size=None):
        """
        Set the assembly type of the boundary integral operators in BEMPP.

        Parameters
        ----------
        eps : float
            The precision of the H-matrix compression.
        max_rank : int
            The maximum rank of the H-matrix blocks.
        max_block_size : int
            The maximum size of the H-matrix blocks.
        """
        from optimus.utils.conversions import convert_to_positive_float
        from optimus.utils.conversions import convert_to_positive_int

        if eps is not None:
            self._matrix_hmat_eps = convert_to_positive_float(eps, "hmat eps")
        if max_rank is not None:
            self._matrix_hmat_max_rank = convert_to_positive_int(
                max_rank, "hmat max rank"
            )
        if max_block_size is not None:
            self._matrix_hmat_max_block_size = convert_to_positive_int(
                max_block_size, "hmat max block size"
            )

    def set_field_hmat(self, eps=None, max_rank=None, max_block_size=None):
        """
        Set the assembly type of the potential integral operators in BEMPP.

        Parameters
        ----------
        eps : float
            The precision of the H-matrix compression.
        max_rank : int
            The maximum rank of the H-matrix blocks.
        max_block_size : int
            The maximum size of the H-matrix blocks.
        """
        from optimus.utils.conversions import convert_to_positive_float
        from optimus.utils.conversions import convert_to_positive_int

        if eps is not None:
            self._field_hmat_eps = convert_to_positive_float(eps, "hmat eps")
        if max_rank is not None:
            self._field_hmat_max_rank = convert_to_positive_int(
                max_rank, "hmat max rank"
            )
        if max_block_size is not None:
            self._field_hmat_max_block_size = convert_to_positive_int(
                max_block_size, "hmat max block size"
            )

    def update_hmat_parameters(self, operator_type):
        """
        Update the H-matrix parameters of BEMPP for matrix or field calculations.

        Remember that BEMPP has only one set of global parameter for the
        hierarchical matrix compression, which is used for both the boundary
        and potential operators.

        Parameters
        ----------
        operator_type : str
            The type of operator: either 'boundary' or 'potential'.
        """
        if operator_type.lower() in ("boundary", "matrix"):
            self.bempp_parameters.hmat.eps = self._matrix_hmat_eps
            self.bempp_parameters.hmat.max_rank = self._matrix_hmat_max_rank
            self.bempp_parameters.hmat.max_block_size = self._matrix_hmat_max_block_size
        elif operator_type.lower() in ("potential", "field"):
            self.bempp_parameters.hmat.eps = self._field_hmat_eps
            self.bempp_parameters.hmat.max_rank = self._field_hmat_max_rank
            self.bempp_parameters.hmat.max_block_size = self._field_hmat_max_block_size
        else:
            raise ValueError("Operator type has to be 'boundary' or 'potential'.")

    def set_quadrature_order(self, operator_type, region, order):
        """
        Set the quadrature order for the numerical integration scheme.

        Parameters
        ----------
        operator_type : str
            The operator type, either 'boundary', 'potential' or 'all'.
        region : str
            The integration region, either 'singular', 'near', 'medium', 'far' or 'all'.
        order : int
            The integration order.
        """
        from optimus.utils.conversions import convert_to_positive_int

        if operator_type.lower() in ("boundary", "matrix"):
            operator_type = "boundary"
        elif operator_type.lower() in ("potential", "field"):
            operator_type = "potential"
        elif operator_type.lower() in ("all", "both"):
            operator_type = "all"
        else:
            raise ValueError(
                "Operator type has to be 'boundary', 'potential' or 'all'."
            )

        if region.lower() in ("singular", "self"):
            region = "singular"
        elif region.lower() in ("near", "medium", "far", "all"):
            region = region.lower()
        else:
            raise ValueError(
                "Region has to be 'singular', 'near', 'medium', 'far' or 'all."
            )

        order = convert_to_positive_int(order, "quadrature order")

        if operator_type in ("boundary", "all"):
            if region in ("singular", "all"):
                self.bempp_parameters.quadrature.double_singular = order
            if region in ("near", "all"):
                self.bempp_parameters.quadrature.near.double_order = order
            if region in ("medium", "all"):
                self.bempp_parameters.quadrature.medium.double_order = order
            if region in ("far", "all"):
                self.bempp_parameters.quadrature.far.double_order = order
        if operator_type in ("potential", "all"):
            if region in ("near", "all"):
                self.bempp_parameters.quadrature.near.single_order = order
            if region in ("medium", "all"):
                self.bempp_parameters.quadrature.medium.single_order = order
            if region in ("far", "all"):
                self.bempp_parameters.quadrature.far.single_order = order
