"""Implement some element for spatial discretization.
"""
import numpy as np

from interpolation import Lagrange
from polynomial import Radau


class FluxReconstruction(object):
    """Element for implement flux reconstruction schemes.
    """

    def __init__(self, equation, degree: int, x_min, x_max) -> None:
        self._equation = equation
        self._degree = degree
        self._x_left = x_min
        self._x_right = x_max
        self._solution_points = np.linspace(x_min, x_max, degree + 1)
        self._solution_lagrange = Lagrange(self._solution_points)
        self._flux_lagrange = Lagrange(self._solution_points)
        self._radau = Radau(degree + 1)

    def approximate(self, function):
        """Approximate a general function.
        """
        n_point = self._degree + 1
        # Sample points strictly inside the element to avoid continuous at boundaries.
        delta = (self._x_right - self._x_left) / 10
        lagrange = Lagrange(np.linspace(self._x_left + delta,
            self._x_right - delta, n_point))
        lagrange.approximate(function)
        solution_values = np.zeros(n_point)
        flux_values = np.zeros(n_point)
        for i in range(n_point):
            solution_values[i] = lagrange.get_function_value(
                self._solution_points[i])
            flux_values[i] = self._equation.F(solution_values[i])
        self._solution_lagrange.set_sample_values(solution_values)
        self._flux_lagrange.set_sample_values(flux_values)

    def get_solution_value(self, x_global):
        """Get the value of U at a given point.
        """
        return self._solution_lagrange.get_function_value(x_global)

    def get_discontinuous_flux(self, x_global):
        """Get the value of the discontinuous flux at a given point.
        """
        return self._flux_lagrange.get_function_value(x_global)

    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the value of the reconstructed continuous flux at a given point.
        """
        flux = self.get_discontinuous_flux(x_global)
        x_local = self._solution_lagrange.global_to_local(x_global)
        radau_left, radau_right = self._radau.get_function_value(x_local)
        flux += radau_right * (upwind_flux_left
            - self.get_discontinuous_flux(self._x_left))
        flux += radau_left * (upwind_flux_right
            - self.get_discontinuous_flux(self._x_right))
        return flux

    def get_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right):
        """
        """
        return 0.0


if __name__ == '__main__':
    pass
