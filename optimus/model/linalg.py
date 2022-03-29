"""Linear algebra methods."""


def linear_solve(
    lhs_system,
    rhs_system,
):
    """
    Solve a system of linear equations.

    Parameters
    ----------
    lhs_system : Numpy array or bempp.api.discreteBoundaryOperator
        The system of linear equations.
    rhs_system : Numpy array
        The right-hand-side vector.

    Returns
    ----------
    solution : Numpy array
        The solution vector.
    """

    from scipy.sparse.linalg import gmres

    solution, info = gmres(
        lhs_system,
        rhs_system,
        tol=1e-5,
        restart=1000,
        maxiter=1000,
    )

    return solution
