"""Implement elements for spatial scheme.
"""
import numpy as np
from scipy import integrate
from scipy import special

from concept import Element, Equation
from polynomial import Vincent
import expansion


class LagrangeDG(Element):
    """Element for implement the DG scheme using a Lagrange expansion.
    """

    def __init__(self, equation: Equation, degree: int,
            x_left: float, x_right: float, value_type=float) -> None:
        super().__init__(x_left, x_right)
        self._equation = equation
        # Sample points evenly distributed in the element.
        delta = (x_right - x_left) / 10
        self._solution_points = np.linspace(x_left + delta, x_right - delta,
            degree + 1)
        # Or, use zeros of special polynomials.
        roots, _ = special.roots_legendre(degree + 1)
        x_center = (x_left + x_right) / 2
        jacobian = (x_right - x_left) / 2
        self._solution_points = x_center + roots * jacobian
        assert len(self._solution_points) == degree + 1
        self._solution_lagrange = expansion.Lagrange(self._solution_points,
            x_left, x_right, value_type)
        self._flux_lagrange = expansion.Lagrange(self._solution_points,
            x_left, x_right, value_type)  # TODO: use self._solution_lagrange
        self._value_type = value_type

    def degree(self):
        return self._solution_lagrange.degree()

    def n_term(self):
        return self._solution_lagrange.n_term()

    def n_dof(self):
        return self.n_term()  # * self_equation.n_component()

    def approximate(self, function):
        coeff = np.ndarray(self.n_term(), self._value_type)
        for i in range(self.n_term()):
            coeff[i] = function(self._solution_points[i])
        self.set_solution_coeff(coeff)

    def get_basis_values(self, x_global):
        return self._solution_lagrange.get_basis_values(x_global)

    def get_basis_gradients(self, x_global):
        return self._solution_lagrange.get_basis_gradients(x_global)

    def build_mass_matrix(self):
        def integrand(points):
            n_row = self.n_term()**2
            n_col = len(points)
            values = np.ndarray((n_row, n_col))
            for c in range(n_col):
                column = self.get_basis_values(points[c])
                matrix = np.tensordot(column, column, axes=0)
                values[:,c] = matrix.reshape(n_row)
            return values
        mass_matrix, _ = integrate.fixed_quad(integrand,
            self.x_left(), self.x_right(), n=self.degree()+1)
        assert self.n_term() == self.n_dof()
        # Otherwise, it should be spanned to a block diagonal matrix.
        return mass_matrix.reshape(self.n_dof(), self.n_dof())

    def set_solution_coeff(self, coeff):
        self._solution_lagrange.set_coeff(coeff)
        # update the corresponding fluxes
        flux_values = np.ndarray(self.n_term(), self._value_type)
        for i in range(self.n_term()):
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
            x_left: float, x_right: float, value_type=float) -> None:
        super().__init__(equation, degree, x_left, x_right, value_type)
        # self._radau = Radau(degree + 1)
        self._correction = Vincent(degree, Vincent.huyhn_lump_lobatto)

    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the value of the reconstructed continuous flux at a given point.
        """
        flux = self.get_discontinuous_flux(x_global)
        x_local = self._solution_lagrange.global_to_local(x_global)
        left, right = self._correction.get_function_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return flux

    def get_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the gradient value of the reconstructed continuous flux at a given point.
        """
        gradient = self._flux_lagrange.get_gradient_value(x_global)
        x_local = self._solution_lagrange.global_to_local(x_global)
        left, right = self._correction.get_gradient_value(x_local)
        left /= self._solution_lagrange.jacobian(x_global)
        right /= self._solution_lagrange.jacobian(x_global)
        gradient += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        gradient += right * (upwind_flux_right
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
