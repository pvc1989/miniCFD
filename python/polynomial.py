"""Implement some polynomial approximations for general functions.
"""
import numpy as np

from concept import Polynomial


class Lagrange(Polynomial):
    """The Lagrange interpolation of a general function.
    """

    def __init__(self, points) -> None:
        super().__init__()
        n_point = len(points)
        assert n_point >= 2
        self._n_point = n_point
        self._jacobian = (points[-1] - points[0]) / 2.0  # linear coordinate transform
        self._local_coords = np.linspace(-1.0, 1.0, n_point)
        self._global_coords = points.copy()
        self._sample_values = np.zeros(n_point)

    def _local_to_global(self, x_local):
        return self._global_coords[0] + (self._jacobian * (x_local + 1.0))

    def _global_to_local(self, x_global):
        return (x_global - self._global_coords[0]) / self._jacobian - 1.0

    def _get_basis_values(self, x_local):
        values = np.zeros(self._n_point)
        for i in range(self._n_point):
            product = 1.0
            for j in range(self._n_point):
                if j == i:
                    continue
                dividend = x_local - self._local_coords[j]
                divisor = self._local_coords[i] - self._local_coords[j]
                product *= (dividend / divisor)
            values[i] = product
        return values

    def _get_basis_gradients(self, x_local):
        values = np.zeros(self._n_point)
        for i in range(self._n_point):
            for j in range(self._n_point):
                if j == i:
                    continue
                dividend = 1.0
                divisor = self._local_coords[i] - self._local_coords[j]
                for k in range(self._n_point):
                    if k in (i, j):
                        continue
                    dividend *= x_local - self._local_coords[k]
                    divisor *= self._local_coords[i] - self._local_coords[k]
                values[i] += dividend / divisor
        return values

    def approximate(self, function):
        for i in range(self._n_point):
            self._sample_values[i] = function(self._global_coords[i])

    def get_function_value(self, x_global):
        x_local = self._global_to_local(x_global)
        values = self._get_basis_values(x_local)
        value = values.dot(self._sample_values)
        return value

    def get_gradient_value(self, x_global):
        x_local = self._global_to_local(x_global)
        values = self._get_basis_gradients(x_local)
        values /= self._jacobian
        value = values.dot(self._sample_values)
        return value


if __name__ == '__main__':
    pass
