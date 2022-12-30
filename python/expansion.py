"""Implement some polynomial approximations for general functions.
"""
import numpy as np
from numpy.testing import assert_almost_equal

from concept import Expansion
import polynomial


class Lagrange(Expansion):
    """The Lagrange expansion of a general function.
    """

    def __init__(self, points, length: float) -> None:
        super().__init__()
        n_point = len(points)
        assert n_point >= 1
        self._n_point = n_point
        if n_point > 1:
            assert_almost_equal(length, points[-1] - points[0])
        assert length > 0
        self._jacobian = length / 2.0  # linear coordinate transform
        self._basis = polynomial.Lagrange(n_point)
        self._global_coords = points.copy()
        self._sample_values = np.zeros(n_point)

    def jacobian(self, x_global: float):
        """Get the Jacobian value at a given point.
        """
        return self._jacobian

    def get_sample_points(self):
        """Get the global coordinates of all sample points."""
        return self._global_coords

    def local_to_global(self, x_local):
        """Coordinate transform from local to global.
        """
        return self._global_coords[0] + (self._jacobian * (x_local + 1.0))

    def global_to_local(self, x_global):
        """Coordinate transform from global to local.
        """
        return (x_global - self._global_coords[0]) / self._jacobian - 1.0

    def set_coeff(self, values):
        """Set values at sample points.

        Users are responsible to ensure (for each i) values[i] is sampled at self._global_coords[i].
        """
        assert len(values) == self._n_point
        for i in range(self._n_point):
            self._sample_values[i] = values[i]

    def get_coeff(self):
        return self._sample_values

    def approximate(self, function):
        for i in range(self._n_point):
            self._sample_values[i] = function(self._global_coords[i])

    def get_function_value(self, x_global):
        x_local = self.global_to_local(x_global)
        values = self._basis.get_function_value(x_local)
        value = values.dot(self._sample_values)
        return value

    def get_gradient_value(self, x_global):
        x_local = self.global_to_local(x_global)
        values = self._basis.get_gradient_value(x_local)
        values /= self._jacobian
        value = values.dot(self._sample_values)
        return value


if __name__ == '__main__':
    pass
