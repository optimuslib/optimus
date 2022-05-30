"""Linear algebra methods."""


def linear_solve(
    lhs_system,
    rhs_system,
    return_iteration_count=False,
):
    """
    Solve a system of linear equations.

    Parameters
    ----------
    lhs_system : Numpy array or bempp.api.discreteBoundaryOperator
        The system of linear equations.
    rhs_system : Numpy array
        The right-hand-side vector.
    return_iteration_count : bool
        Return the number of iterations of the linear solver.
        Default: False

    Returns
    ----------
    solution : Numpy array
        The solution vector.
    it_count : int
        The number of iterations (optional).
    """

    solver = GmresSolver(lhs_system, rhs_system)
    solution = solver.solve()

    if return_iteration_count:
        return solution, solver.it_count
    else:
        return solution


class GmresSolver:
    def __init__(self, lhs_system, rhs_system):
        """
        Create a GMRES linear solver.

        Parameters
        ----------
        lhs_system : Numpy array or bempp.api.discreteBoundaryOperator
            The system of linear equations.
        rhs_system : Numpy array
            The right-hand-side vector of the system.
        """
        self.lhs_matrix = lhs_system
        self.rhs_vector = rhs_system
        self.it_count = None

    def solve(self):
        """
        Solve the linear system with GMRES.
        """
        from scipy.sparse.linalg import gmres
        import optimus

        global_params_linalg = optimus.global_parameters.linalg

        self.it_count = 0

        def callback_fct(residual):
            self.it_count += 1

        solution, info = gmres(
            self.lhs_matrix,
            self.rhs_vector,
            tol=global_params_linalg.tol,
            maxiter=global_params_linalg.maxiter,
            restart=min(global_params_linalg.maxiter, global_params_linalg.restart),
            callback=callback_fct,
        )

        if self.it_count == global_params_linalg.maxiter:
            import warnings

            warnings.warn(
                "The GMRES solver stopped at the maximum number of "
                + str(self.it_count)
                + " iterations.",
                RuntimeWarning,
            )

        return solution
