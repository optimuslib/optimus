"""Linear algebra methods."""

import optimus


def linear_solve(
    lhs_system,
    rhs_system,
    verbosity=False,
):
    """
    Solve a system of linear equations.

    Parameters
    ----------
    lhs_system : scipy.sparse.linalg.LinearOperator
        The system of linear equations.
    rhs_system : numpy.ndarray
        The right-hand-side vector.
    verbosity : bool
        Display the iterations of the linear solver.
        Default: False

    Returns
    -------
    solution : numpy.ndarray
        The solution vector.
    linear_solve_it_count : int
        The number of iterations.
    linear_solve_residual_error : list
        The error at each iterations.
    total_time : int
        The total time for solver (in seconds).
    time_per_it : int
        The average time per iterations for GMRES.
    """

    if optimus.global_parameters.linalg.linsolver == "gmres":
        solver = GmresSolver(lhs_system, rhs_system, verbosity)
    else:
        raise ValueError(
            "Linear solver "
            + optimus.global_parameters.linalg.linsolver
            + " is not known."
        )
    solution = solver.solve()

    # if verbosity:
    return (
        solution,
        solver.linear_solve_it_count[-1],
        solver.linear_solve_residual_error,
        solver.total_time,
        solver.time_per_it,
    )
    # else:
    #     return solution


class GmresSolver:
    def __init__(self, lhs_system, rhs_system, verbosity=False):
        """
        Create a GMRES linear solver.

        Parameters
        ----------
        lhs_system : scipy.sparse.linalg.LinearOperator
            The system of linear equations.
        rhs_system : numpy.ndarray
            The right-hand-side vector of the system.
        """

        self.lhs_matrix = lhs_system
        self.rhs_vector = rhs_system
        self.verbosity = verbosity
        self.linear_solve_it_count = []
        self.linear_solve_residual_error = []

        if optimus.global_parameters.verbosity:
            self.verbosity = True

    def solve(self):
        """
        Solve the linear system with GMRES.
        """

        from scipy.sparse.linalg import gmres
        from time import time as _time

        global_params_linalg = optimus.global_parameters.linalg

        self._iter_count_tmp = 0

        def callback_fct(rk=None):
            """Iteration counter function for gmres"""
            self._iter_count_tmp += 1
            self.linear_solve_residual_error.append(rk)
            self.linear_solve_it_count.append(self._iter_count_tmp)
            if self.verbosity:
                print("iter %3i\trk = %s" % (self._iter_count_tmp, str(rk)))

            return

        t_start = _time()
        solution, INFO = gmres(
            self.lhs_matrix,
            self.rhs_vector,
            tol=global_params_linalg.tol,
            maxiter=global_params_linalg.maxiter,
            restart=min(global_params_linalg.maxiter, global_params_linalg.restart),
            callback=callback_fct,
        )
        t_end = _time()
        total_time = t_end - t_start
        self.total_time = total_time  # _timedelta(seconds=total_time)
        self.time_per_it = total_time / self._iter_count_tmp
        if self.verbosity:
            print(
                "\n",
                70 * "*",
                "\n The linear system was solved in: \n {0} iterations, \n time: {1} ( = {2} secs) ".format(
                    self._iter_count_tmp, str(self.total_time), total_time
                ),
            )
            print(
                "\n",
                70 * "*",
                "\nThe average time per iteration is: {0:.3f} secs".format(
                    self.time_per_it
                ),
            )

        if self.linear_solve_it_count[-1] == global_params_linalg.maxiter:
            import warnings

            warnings.warn(
                "The GMRES solver stopped at the maximum number of "
                + str(self.linear_solve_it_count[-1])
                + " iterations.",
                RuntimeWarning,
            )

        return solution
