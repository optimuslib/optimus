"""Utilities for data type checks and conversions."""

import numpy as _np


def convert_to_positive_int(value, label="variable"):
    """
    Check if the input value can be converted into a positive integer.

    Parameters
    ----------
    value : Any
        The input value to be converted into a positive integer.
    label : str
        The name of the variable.

    Returns
    -------
    value : int
        The output value.
    """

    if not isinstance(value, (float, int)):
        raise TypeError(label + " needs to be a float or int, not " + str(type(value)))

    if value < 0:
        raise ValueError(label + " needs to be positive, not " + str(value))

    return int(value)


def convert_to_float(value, label="variable"):
    """
    Check if the input value can be converted into a float.

    Parameters
    ----------
    value : Any
        The input value to be converted into a float.
    label : str
        The name of the variable.

    Returns
    -------
    value : float
        The output value.
    """

    if not isinstance(value, (float, int)):
        raise TypeError(label + " needs to be a float or int, not " + str(type(value)))

    return float(value)


def convert_to_array(vector, shape=None, label="variable"):
    """
    Check if the input vector can be converted into an array for the specified
    shape, and perform the conversion.

    Parameters
    ----------
    vector : Any
        The input vector to be converted into a Numpy array.
    shape : tuple
        The output shape of the vector.
    label : str
        The name of the variable.

    Returns
    -------
    array : np.ndarray
        The output array with the specified shape.
    """

    if not isinstance(vector, (list, tuple, _np.ndarray)):
        raise TypeError(label + " needs to be an array type, not " + str(type(vector)))

    array = _np.array(vector)

    if array.dtype not in (float, int):
        raise TypeError(
            label + " needs to be of type float or int, not " + str(array.dtype)
        )

    if shape is not None:
        size = _np.prod(shape)
        if array.size != size:
            raise ValueError(
                label + " needs to have size " + str(size) + ", not " + str(array.size)
            )
        return _np.reshape(array, shape)
    else:
        return array


def convert_to_3n_array(array, label="variable"):
    """
    Convert the input array into a 3xN Numpy array, if possible.

    Parameters
    ----------
    array : list, tuple, array
        The input vector to be converted into a 3xN Numpy array.
    label : str
        The name of the variable.

    Returns
    -------
    array : np.ndarray
        The output array with the shape of 3xN.
    """

    array_np = convert_to_array(array)

    if array_np.ndim == 1:

        if array_np.size == 3:
            return array_np.reshape([3, 1])
        else:
            raise ValueError(label + " needs to be three dimensional.")

    elif array_np.ndim == 2:

        if array_np.shape[0] == 3:
            return array_np
        elif array_np.shape[1] == 3:
            return array_np.transpose()
        else:
            raise ValueError(label + " needs to be three dimensional.")

    else:

        raise ValueError(label + " needs to be three dimensional.")
