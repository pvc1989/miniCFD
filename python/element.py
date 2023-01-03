"""Implement elements for spatial discretization.
"""
import numpy as np
from scipy import integrate

from concept import Element, Equation
from expansion import Lagrange
from polynomial import Radau


class LagrangeDG(Element):
    """Element for implement the DG scheme using a Lagrange expansion.
    """

    def __init__(self, equation: Equation, degree: int,
            x_left: float, x_right: float) -> None:
        super().__init__(x_left, x_right)
        self._equation = equation
        self._n_point = degree + 1
        length = x_right - x_left
        self._coord_lagrange = Lagrange(np.linspace(x_left, x_right, degree + 1), length)
        # Sample points evenly distributed in the element.
        delta = length / 10
        length -= delta * 2
        self._solution_points = np.linspace(x_left + delta, x_right - delta, self._n_point)
        if degree == 0:
            self._solution_points[0] = (x_left + x_right) / 2
        self._solution_lagrange = Lagrange(self._solution_points, length)
        self._flux_lagrange = Lagrange(self._solution_points, length)
        self._radau = Radau(degree + 1)

    def degree(self):
        return self._solution_lagrange.degree()

    def n_dof(self):
        return self._n_point  # * self_equation.n_component()

    def approximate(self, function):
        coeff = np.ndarray(self._n_point)
        for i in range(self._n_point):
            coeff[i] = function(self._solution_points[i])
        self.set_solution_coeff(coeff)

    def get_basis_values(self, x_global):
        return self._solution_lagrange.get_basis_values(x_global)

    def get_basis_gradients(self, x_global):
        return self._solution_lagrange.get_basis_gradients(x_global)

    def build_mass_matrix(self):
        def integrand(points):
            n_row = self._n_point**2
            n_col = len(points)
            values = np.ndarray((n_row, n_col))
            for c in range(n_col):
                column = self.get_basis_values(points[c])
                matrix = np.tensordot(column, column, axes=0)
                values[:,c] = matrix.reshape(n_row)
            return values
        mass_matrix, _ = integrate.fixed_quad(
            integrand, self.x_left(), self.x_right(), n=5)
        assert self._n_point == self.n_dof()
        # Otherwise, it should be spanned to a block diagonal matrix.
        return mass_matrix.reshape(self.n_dof(), self.n_dof())

    def set_solution_coeff(self, coeff):
        self._solution_lagrange.set_coeff(coeff)
        # update the corresponding fluxes
        flux_values = np.zeros(self._n_point)
        for i in range(self._n_point):
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


class LagrangeFR(LagrangeDG):
    """Element for implement the FR scheme using a Lagrange expansion.
    """

    def __init__(self, equation: Equation, degree: int,
            x_left: float, x_right: float) -> None:
        super().__init__(equation, degree, x_left, x_right)
        length = x_right - x_left
        self._coord_lagrange = Lagrange(np.linspace(x_left, x_right, degree + 1), length)
        length = self._solution_points[-1] - self._solution_points[0]
        self._flux_lagrange = Lagrange(self._solution_points, length)
        self._radau = Radau(degree + 1)

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
