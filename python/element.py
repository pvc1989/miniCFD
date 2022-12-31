"""Implement elements for spatial discretization.
"""
import numpy as np

from concept import Element, Equation
from expansion import Lagrange
from polynomial import Radau


class FluxReconstruction(Element):
    """Element for implement flux reconstruction schemes.
    """

    def __init__(self, equation: Equation, degree: int,
            x_min: float, x_max: float) -> None:
        self._equation = equation
        self._degree = degree
        self._boundaries = (x_min, x_max)
        length = x_max - x_min
        self._coord_lagrange = Lagrange(np.linspace(x_min, x_max, degree + 1), length)
        # Sample points evenly distributed in the element.
        delta = (self.x_right() - self.x_left()) / 10
        length -= delta * 2
        self._solution_points = np.linspace(x_min + delta, x_max - delta, degree + 1)
        if degree == 0:
            self._solution_points[0] = (x_min + x_max) / 2
        self._solution_lagrange = Lagrange(self._solution_points, length)
        self._flux_lagrange = Lagrange(self._solution_points, length)
        self._radau = Radau(degree + 1)

    def x_left(self):
        return self._boundaries[0]

    def x_right(self):
        return self._boundaries[-1]

    def approximate(self, function):
        n_point = self._degree + 1
        coeff = np.ndarray(n_point)
        for i in range(n_point):
            coeff[i] = function(self._solution_points[i])
        self.set_solution_coeff(coeff)

    def set_solution_coeff(self, coeff):
        self._solution_lagrange.set_coeff(coeff)
        # update the corresponding fluxes
        n_point = self._degree + 1
        flux_values = np.zeros(n_point)
        for i in range(n_point):
            flux_values[i] = self._equation.get_convective_flux(coeff[i])
        self._flux_lagrange.set_coeff(flux_values)

    def get_solution_column(self):
        # In current case (scalar problem), no conversion is needed.
        return self._solution_lagrange.get_coeff()

    def get_solution_value(self, x_global):
        return self._solution_lagrange.get_function_value(x_global)

    def get_discontinuous_flux(self, x_global):
        """Get the value of the discontinuous flux at a given point.
        """
        return self._flux_lagrange.get_function_value(x_global)

    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the value of the reconstructed continuous flux at a given point.
        """
        flux = self.get_discontinuous_flux(x_global)
        x_local = self._coord_lagrange.global_to_local(x_global)
        radau_left, radau_right = self._radau.get_function_value(x_local)
        flux += radau_right * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        flux += radau_left * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return flux

    def get_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the gradient value of the reconstructed continuous flux at a given point.
        """
        gradient = self._flux_lagrange.get_gradient_value(x_global)
        x_local = self._coord_lagrange.global_to_local(x_global)
        radau_left, radau_right = self._radau.get_gradient_value(x_local)
        radau_left /= self._coord_lagrange.jacobian(x_global)
        radau_right /= self._coord_lagrange.jacobian(x_global)
        gradient += radau_right * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        gradient += radau_left * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return gradient

    def get_flux_gradients(self, upwind_flux_left, upwind_flux_right):
        """Get the gradients of the continuous flux at allsolution points.
        """
        values = np.ndarray(len(self._solution_points))
        for i in range(len(values)):
            point_i = self._solution_points[i]
            values[i] = self.get_flux_gradient(point_i,
                upwind_flux_left, upwind_flux_right)
        return values


if __name__ == '__main__':
    pass
