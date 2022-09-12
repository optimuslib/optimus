"""Unit tests."""

import optimus
import numpy as np
import pytest


def test_unit_sphere_radius():
    """Test of radius of a unit sphere bempp grid."""

    geometry = optimus.geometry.shapes.Sphere(element_size=0.4)

    vertices = geometry.grid.leaf_view.vertices

    actual = np.mean(
        np.sqrt(vertices[0, :] ** 2 + vertices[1, :] ** 2 + vertices[2, :] ** 2)
    )

    expected = 1.1

    np.testing.assert_almost_equal(actual, expected, decimal=5)