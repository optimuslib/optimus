"""Utilities for linear algebra."""

import numpy as _np


def normalize_vector(vector):
    """
    Convert a vector into a unit vector.
    For 2D input arrays, the columns will be normalized.

    Parameters
    ------
    vector : np.ndarray of size (n,) or (n,m)
        The input vectors of dimension n.

    Returns
    -------
    unit_vector : np.ndarray of size (n,) or (n,m)
        The vectors scaled to unit Euclidean length.
    """

    if not isinstance(vector, _np.ndarray):
        raise TypeError("Vector needs to be a Numpy array.")

    if vector.ndim == 1:
        return vector / _np.linalg.norm(vector)
    elif vector.ndim == 2:
        return vector / _np.linalg.norm(vector, axis=0)
    else:
        raise ValueError("Vector needs to be 1D or 2D.")


def theta_phi_point(point):
    """
    Counter-clockwise angle in xy plane (theta) and altitude towards z (phi).

    Parameters
    ------
    point : 1D array of size 3
      Rotation axis.

    Returns
    -------
    theta : float
      angle on the xy plane measured from the positive x-axis
    phi: float
      angle towards z from the xy plane from the plane upwards.

    Example
    -------
    >>> point = [0, 1, 0]
    >>> theta, phi = theta_phi_point(point) # (pi/2, 0)
    >>> print("theta = π/2 = {:.5f}; phi = 0 = {:.1f}".format(theta, phi))
    theta = π/2 = 1.57080; phi = 0 = 0.0
    >>> point = [0, 1, 1]
    >>> theta, phi = theta_phi_point(point) # (pi/2, pi/4)
    >>> print("theta = π/2 = {:.5f}; phi = π/4 = {:.5f}".format(theta, phi))
    theta = π/2 = 1.57080; phi = π/4 = 0.78540
    """

    theta = _np.arctan2(point[1], point[0])
    xy_proj = _np.linalg.norm(point[:2])
    phi = _np.arctan2(point[2], xy_proj)

    return theta, phi


def rotate(locations, source_axis):
    """
    Rotates the coordinates of the point sources (`locations`) which
    approximate the transducer prior to the coordinate transformation.
    The source axis is a three element vector which defines the direction
    towards which the source is "pointing" (or the main axis of propagation)
    after coordinate transformation.
    The components of the vector are in metres.

    Parameters
    ------
    locations : np.ndarray of size (3, N)
      Coordinates of the point sources.
    source_axis: np.ndarray of size (3,)
      Axis of propagation.

    Returns
    -------
    locations : np.ndarray of size (3, N)
        Locations rotated by the axis of propagation.
    """

    source_axis = _np.array(source_axis).flatten()

    # Change orientation of source as defined by vector (nx,ny,nz)
    # Compute unit vector collinear to source axis
    v_source_axis = normalize_vector(source_axis)

    # Obtain counter-clockwise angle in xy plane measured from the
    # positive x-axis
    theta_source, phi_source = theta_phi_point(v_source_axis)

    # First rotation grad_pressurerix: 90 degrees about y axis so that
    # default orientation is now transformed to x-axis
    r_y = _np.array(
        [
            [_np.cos(_np.pi / 2), 0, _np.sin(_np.pi / 2)],
            [0, 1, 0],
            [-_np.sin(_np.pi / 2), 0, _np.cos(_np.pi / 2)],
        ]
    )

    # Second rotation grad_pressurerix: from default z-axis axis of
    # symmetry to x-axis
    r_theta = _np.array(
        [
            [_np.cos(theta_source), -_np.sin(theta_source), 0],
            [_np.sin(theta_source), _np.cos(theta_source), 0],
            [0, 0, 1],
        ]
    )

    # Third rotation grad_pressurerix
    # Define orientation of vector collinear with axis of rotation
    ux = _np.sin(theta_source)
    uy = -_np.cos(theta_source)
    cosphi = _np.cos(phi_source)
    si_nphi = _np.sin(phi_source)
    r_phi = _np.array(
        [
            [
                cosphi + ux**2 * (1 - cosphi),
                ux * uy * (1 - cosphi),
                uy * si_nphi,
            ],
            [
                ux * uy * (1 - cosphi),
                cosphi + uy**2 * (1 - cosphi),
                -ux * si_nphi,
            ],
            [-uy * si_nphi, ux * si_nphi, cosphi],
        ]
    )

    return r_phi @ r_theta @ r_y @ locations


def translate(locations, translation):
    """
    Translate an array of location points.

    Parameters
    ------
    locations : np.ndarray of size (3, N)
      The point locations to be translated.
    translation: np.ndarray of size (3, 1)
      The displacement desired to perform.

    Returns
    -------
    locations : np.ndarray of size (3, N)
        Positions translated by the required shift.
    """

    translation_array = _np.array(translation).reshape(3, 1)
    if locations.size == 3:
        locations = locations.reshape(3, 1)

    return locations + translation_array
