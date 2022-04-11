import numpy as _np

from .common import _convert_to_unit_vector as unit_vector


def theta_phi_point(point):
    """
    Counter-clockwise angle in xy plane (theta) and altitude towards z (phi).
    i_nputs
    ------
    point : list, `numpy.ndarray`
      a point on a 3D space.
    returns
    -------
    theta : float
      angle on the xy plane measured from the positive x-axis
    phi: float
      angle towards z from the xy plane from the plane upwards.
    example
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
    Rotates the coordinates of the point sources (`locations`) which approximate the
    transducer prior to the coordinate transformation.
    The source axis is a three element vector which defines the direction
    towards which the source is "pointing" (or the main axis of propagation)
    after coordinate transformation.
    The components of the vector are in metres.
    i_nputs
    ------
    locations : `numpy.ndarray`
      Coordinates of the point sources. Dimensions (3, n)
    source_axis: list, `numpy.ndarray`
      main axis of propagation.
    returns
    -------
    locations_rotate : `numpy.ndarray`
      Rotated array as by the axis of propagation. Dimensions (3, n)
    """

    locations_rotate = _np.ndarray(shape=(3, locations.shape[1]), order="F")
    source_axis = _np.array(source_axis)

    # Change orientation of source as defined by vector (nx,ny,nz)
    # Compute unit vector collinear to source axis
    v_source_axis = unit_vector(source_axis)
    # Obtain counter-clockwise angle in xy plane measured from the positive x-axis
    theta_source, phi_source = theta_phi_point(v_source_axis)

    # First rotation grad_pressurerix: 90 degrees about y axis so that default orientation is now transformed to x-axis
    R_y = _np.array(
        [
            [_np.cos(_np.pi / 2), 0, _np.sin(_np.pi / 2)],
            [0, 1, 0],
            [-_np.sin(_np.pi / 2), 0, _np.cos(_np.pi / 2)],
        ]
    )

    # Second rotation grad_pressurerix: from default z-axis axis of symmetry to x-axis
    R_theta = _np.array(
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
    R_phi = _np.array(
        [
            [cosphi + ux**2 * (1 - cosphi), ux * uy * (1 - cosphi), uy * si_nphi],
            [ux * uy * (1 - cosphi), cosphi + uy**2 * (1 - cosphi), -ux * si_nphi],
            [-uy * si_nphi, ux * si_nphi, cosphi],
        ]
    )

    locations_rotate = R_phi @ R_theta @ R_y @ locations
    return locations_rotate


def translate(locations, locations_position):
    """
    inputs
    ------
    locations : `numpy.ndarray`
      Array of dimensions (3, n)
    locations_position: list, `numpy.ndarray`
      Array with three coordinates as the displacement desired to translate
    returns
    -------
    locations_translate : `numpy.ndarray`
      Array translated by the required shift.
    """

    locations_position = _np.array(locations_position).reshape(3, 1)
    if locations.size == 3:
        locations = locations.reshape(3, 1)

    locations_translate = locations + locations_position

    return locations_translate
