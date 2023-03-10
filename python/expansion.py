"""Implement some polynomial approximations for general functions.
"""
import numpy as np
from scipy import special

from concept import Expansion
import polynomial
import integrate


class Lagrange(Expansion):
    """The Lagrange expansion of a general function.
    """

    def __init__(self, points: np.ndarray, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__()
        n_point = len(points)
        assert n_point >= 1
        assert x_left <= points[0] <= points[-1] <= x_right
        self._sample_values = np.ndarray(n_point, value_type)
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
        values /= self._jacobian
        return values

    def get_function_value(self, x_global):
        values = self.get_basis_values(x_global)
        value = values.dot(self._sample_values)
        return value

    def get_gradient_value(self, x_global):
        basis_grad = self.get_basis_gradients(x_global)
        value = basis_grad.dot(self._sample_values)
        return value


class Legendre(Expansion):
    """Approximate a general function based on Legendre polynomials.
    """

    def __init__(self, degree: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__()
        self._n_term = degree + 1
        # linear coordinate transform:
        self._x_left = x_left
        self._jacobian = (x_right - x_left) / 2.0
        self._mode_coeffs = np.ndarray(self._n_term, value_type)
        self._mode_weights = np.ndarray(self._n_term, float)
        for k in range(self._n_term):
            self._mode_weights[k] = self._jacobian * integrate.fixed_quad_local(
                lambda x: (special.eval_legendre(k, x))**2, n_point=k+1)

    def get_mode_weight(self, k):
        """Get the inner-product of the kth basis with itself.
        """
        return self._mode_weights[k]

    def n_term(self):
        return self._n_term

    def degree(self):
        return self._n_term + 1

    def jacobian(self, x_global: float):
        """Get the Jacobian value at a given point.
        """
        return self._jacobian

    def local_to_global(self, x_local):
        """Coordinate transform from local to global.
        """
        return self._x_left + (self._jacobian * (x_local + 1.0))

    def global_to_local(self, x_global):
        """Coordinate transform from global to local.
        """
        return (x_global - self._x_left) / self._jacobian - 1.0

    def set_coeff(self, coeffs):
        """Set coefficient for each mode.
        """
        assert len(coeffs) == self.n_term()
        for i in range(len(coeffs)):
            self._mode_coeffs[i] = coeffs[i]

    def get_coeff(self):
        return self._mode_coeffs

    def approximate(self, function):
        for k in range(self.n_term()):
            def integrand(x_local):
                value = special.eval_legendre(k, x_local)
                value *= function(self.local_to_global(x_local))
                return value
            self._mode_coeffs[k] = integrate.fixed_quad_local(integrand,
                self.n_term()) * self._jacobian
            self._mode_coeffs[k] /= self._mode_weights[k]

    def get_basis_values(self, x_global):
        x_local = self.global_to_local(x_global)
        values = np.ndarray(self.n_term())
        for k in range(self.n_term()):
            values[k] = special.eval_legendre(k, x_local)
        return values

    def get_basis_gradients(self, x_global):
        x_local = self.global_to_local(x_global)
        values = np.ndarray(self.n_term())
        values[0] = 0.0
        for k in range(1, self.n_term()):
            values[k] = (k * special.eval_legendre(k-1, x_local)
                + x_local * values[k-1])
        values /= self._jacobian
        return values

    def get_function_value(self, x_global):
        values = self.get_basis_values(x_global)
        value = values.dot(self._mode_coeffs)
        return value

    def get_gradient_value(self, x_global):
        basis_grad = self.get_basis_gradients(x_global)
        value = basis_grad.dot(self._mode_coeffs)
        return value


if __name__ == '__main__':
    pass
