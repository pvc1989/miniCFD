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
        self._n_point = n_point
        self._local_coords = np.linspace(-1.0, 1.0, n_point)
        self._global_coords = points.copy()
        self._sample_values = np.zeros(n_point)

    def _local_to_global(self, x_local):
        pass

    def _global_to_local(self, x_global):
        pass

    def _get_basis_value(self, x_local):
        return np.random.rand(self._n_point)

    def approximate(self, function):
        pass

    def get_function_value(self, x_global):
        x_local = self._global_to_local(x_global)
        return self._get_basis_value(x_local).dot(self._sample_values)

    def get_gradient_value(self, x_global):
        pass


if __name__ == '__main__':
    pass
