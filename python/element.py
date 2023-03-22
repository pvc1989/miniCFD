"""Implement elements for spatial scheme.
"""
import numpy as np

from concept import Element, Equation
from polynomial import Vincent
import expansion
import integrate


class LagrangeDG(Element):
    """Element for implement the DG scheme using a Lagrange expansion.
    """

    def __init__(self, equation: Equation, degree: int,
            x_left: float, x_right: float, value_type=float) -> None:
        super().__init__(equation, x_left, x_right, value_type)
        self._u_approx = expansion.Lagrange(degree,
            x_left, x_right, value_type)
        self._mass_matrix = self._build_mass_matrix()

    def degree(self):
        return self._u_approx.degree()

    def n_term(self):
        return self._u_approx.n_term()

    def n_dof(self):
        return self.n_term()  # * self_equation.n_component()

    def approximate(self, function):
        self._u_approx.approximate(function)

    def get_basis_values(self, x_global):
        return self._u_approx.get_basis_values(x_global)

    def get_basis_gradients(self, x_global):
        return self._u_approx.get_basis_gradients(x_global)

    def _build_mass_matrix(self):
        def integrand(x_global):
            column = self.get_basis_values(x_global)
            matrix = np.tensordot(column, column, axes=0)
            return matrix
        mass_matrix = integrate.fixed_quad_global(integrand,
            self.x_left(), self.x_right(), self.n_term())
        assert self.n_term() == self.n_dof()
        # Otherwise, it should be spanned to a block diagonal matrix.
        return mass_matrix

    def divide_mass_matrix(self, column: np.ndarray):
        return np.linalg.solve(self._mass_matrix, column)

    def set_solution_coeff(self, coeff):
        self._u_approx.set_coeff(coeff)

    def get_solution_column(self):
        # In current case (scalar problem), no conversion is needed.
        return self._u_approx.get_coeff()

    def get_solution_value(self, x_global):
        return self._u_approx.get_function_value(x_global)

    def get_discontinuous_flux(self, x_global):
        """Get the value of the discontinuous flux at a given point.
        """
        u_approx = self.get_solution_value(x_global)
        return self._equation.get_convective_flux(u_approx)


class LegendreDG(Element):
    """Element for implement the DG scheme based on Legendre polynomials.

    Since flux is not explicitly approximated, it's just a wrapper of the
    expansion.Legendre class.
    """

    def __init__(self, equation: Equation, degree: int,
            x_left: float, x_right: float, value_type=float) -> None:
        super().__init__(equation, x_left, x_right, value_type)
        self._u_approx = expansion.Legendre(degree, x_left, x_right,
            value_type)

    def degree(self):
        return self._u_approx.degree()

    def n_term(self):
        return self._u_approx.n_term()

    def n_dof(self):
        return self.n_term()  # * self_equation.n_component()

    def approximate(self, function):
        self._u_approx.approximate(function)

    def get_basis_values(self, x_global):
        return self._u_approx.get_basis_values(x_global)

    def get_basis_gradients(self, x_global):
        return self._u_approx.get_basis_gradients(x_global)

    def set_solution_coeff(self, coeff):
        self._u_approx.set_coeff(coeff)

    def get_solution_column(self):
        # In current case (scalar problem), no conversion is needed.
        return self._u_approx.get_coeff()

    def get_solution_value(self, x_global):
        return self._u_approx.get_function_value(x_global)

    def get_discontinuous_flux(self, x_global):
        """Get the value of the discontinuous flux at a given point.
        """
        u_approx = self.get_solution_value(x_global)
        return self._equation.get_convective_flux(u_approx)

    def divide_mass_matrix(self, column: np.ndarray):
        for k in range(self.n_term()):
            column[k] /= self._u_approx.get_mode_weight(k)
        return column


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
        x_local = self._u_approx.global_to_local(x_global)
        left, right = self._correction.get_function_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return flux

    def get_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the gradient value of the reconstructed continuous flux at a given point.
        """
        u_approx = self._u_approx.get_function_value(x_global)
        a_approx = self._equation.get_convective_jacobian(u_approx)
        gradient = self._u_approx.get_gradient_value(x_global)
        gradient = a_approx * gradient
        x_local = self._u_approx.global_to_local(x_global)
        left, right = self._correction.get_gradient_value(x_local)
        left /= self._u_approx.jacobian(x_global)
        right /= self._u_approx.jacobian(x_global)
        gradient += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        gradient += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return gradient

    def get_flux_gradients(self, upwind_flux_left, upwind_flux_right):
        """Get the gradients of the continuous flux at all nodes.
        """
        nodes = self._u_approx.get_sample_points()
        values = np.ndarray(len(nodes), self._value_type)
        for i in range(len(nodes)):
            values[i] = self.get_flux_gradient(nodes[i],
                upwind_flux_left, upwind_flux_right)
        return values


class LegendreFR(LegendreDG):
    """Element for implement the FR scheme using a Legendre expansion.
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
        x_local = self._u_approx.global_to_local(x_global)
        left, right = self._correction.get_function_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return flux

    def get_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the gradient value of the reconstructed continuous flux at a given point.
        """
        u_approx = self._u_approx.get_function_value(x_global)
        a_approx = self._equation.get_convective_jacobian(u_approx)
        gradient = a_approx * self._u_approx.get_gradient_value(x_global)
        x_local = self._u_approx.global_to_local(x_global)
        left, right = self._correction.get_gradient_value(x_local)
        left /= self._u_approx.jacobian(x_global)
        right /= self._u_approx.jacobian(x_global)
        gradient += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        gradient += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return gradient

    def get_flux_gradients(self, upwind_flux_left, upwind_flux_right):
        """Get the gradients of the continuous flux at all modes.
        """
        # TODO: project grad-correction on the Legendre basis in init.
        def integrand(x_global):
            column = self.get_basis_values(x_global) * self.get_flux_gradient(
                x_global, upwind_flux_left, upwind_flux_right)
            column = self.divide_mass_matrix(column)
            return column
        values = integrate.fixed_quad_global(integrand,
            self.x_left(), self.x_right(), self.n_term())
        return values


if __name__ == '__main__':
    pass
