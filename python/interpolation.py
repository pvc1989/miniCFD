"""Implement some polynomial approximations for general functions.
"""
import numpy as np

from concept import Interpolation
import polynomial


class Lagrange(Interpolation):
    """The Lagrange interpolation of a general function.
    """

    def __init__(self, points) -> None:
        super().__init__()
        n_point = len(points)
        assert n_point >= 2
        self._n_point = n_point
        self._jacobian = (points[-1] - points[0]) / 2.0  # linear coordinate transform
        self._basis = polynomial.Lagrange(n_point)
        self._global_coords = points.copy()
        self._sample_values = np.zeros(n_point)

    def local_to_global(self, x_local):
        """Coordinate transform from local to global.
        """
        return self._global_coords[0] + (self._jacobian * (x_local + 1.0))

    def global_to_local(self, x_global):
        """Coordinate transform from global to local.
        """
        return (x_global - self._global_coords[0]) / self._jacobian - 1.0

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
