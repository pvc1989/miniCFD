"""Implement some polynomial approximations for general functions.
"""
import numpy as np

from concept import Expansion
import polynomial


class Lagrange(Expansion):
    """The Lagrange expansion of a general function.
    """

    def __init__(self, points: np.ndarray, x_left: float, x_right: float) -> None:
        super().__init__()
        n_point = len(points)
        assert n_point >= 1
        assert x_left <= points[0] <= points[-1] <= x_right
        self._sample_values = np.zeros(n_point)
        self._sample_points = points
        if n_point == 1:  # use the centroid rather than the left end
            self._sample_points[0] = (x_left + x_right) / 2.0
        # linear coordinate transform:
        self._x_left = x_left
        self._jacobian = (x_right - x_left) / 2.0
        local_points = np.ndarray(n_point)
        for i in range(n_point):
            local_points[i] = self.global_to_local(points[i])
        self._basis = polynomial.LagrangeBasis(local_points)

    def n_term(self):
        return self._basis.n_term()

    def degree(self):
        return self.n_term() - 1

    def jacobian(self, x_global: float):
        """Get the Jacobian value at a given point.
        """
        return self._jacobian

    def get_sample_points(self):
        """Get the global coordinates of all sample points."""
        return self._sample_points

    def local_to_global(self, x_local):
        """Coordinate transform from local to global.
        """
        return self._x_left + (self._jacobian * (x_local + 1.0))

    def global_to_local(self, x_global):
        """Coordinate transform from global to local.
        """
        return (x_global - self._x_left) / self._jacobian - 1.0

    def set_coeff(self, values):
        """Set values at sample points.

        Users are responsible to ensure (for each i) values[i] is sampled at self._sample_points[i].
        """
        assert len(values) == self._basis.n_term()
        for i in range(len(values)):
            self._sample_values[i] = values[i]

    def get_coeff(self):
        return self._sample_values

    def approximate(self, function):
        for i in range(self._basis.n_term()):
            self._sample_values[i] = function(self._sample_points[i])

    def get_basis_values(self, x_global):
        x_local = self.global_to_local(x_global)
        values = self._basis.get_function_value(x_local)
        return values

    def get_basis_gradients(self, x_global):
        x_local = self.global_to_local(x_global)
        values = self._basis.get_gradient_value(x_local)
        return values

    def get_function_value(self, x_global):
        values = self.get_basis_values(x_global)
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
