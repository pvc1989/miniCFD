"""Implement elements for spatial scheme.
"""
import numpy as np

from concept import Element, Equation, Coordinate
from polynomial import Vincent
import expansion


class LagrangeDG(Element):
    """Element for implement the DG scheme using a Lagrange expansion.
    """

    def __init__(self, equation: Equation, degree: int,
            coordinate: Coordinate, value_type=float) -> None:
        Element.__init__(self, equation, coordinate,
            expansion.Lagrange(degree, coordinate, value_type), value_type)
        self._mass_matrix = self._build_mass_matrix()

    def divide_mass_matrix(self, column: np.ndarray):
        return np.linalg.solve(self._mass_matrix, column)

    def get_sample_points(self) -> np.ndarray:
        assert isinstance(self.expansion, expansion.Lagrange)
        return self.expansion.get_sample_points()


class LegendreDG(Element):
    """Element for implement the DG scheme based on Legendre polynomials.

    Since flux is not explicitly approximated, it's just a wrapper of the
    expansion.Legendre class.
    """

    def __init__(self, equation: Equation, degree: int,
            coordinate: Coordinate, value_type=float) -> None:
        Element.__init__(self, equation, coordinate,
            expansion.Legendre(degree, coordinate, value_type), value_type)

    def divide_mass_matrix(self, column: np.ndarray):
        assert isinstance(self.expansion, expansion.Legendre)
        for k in range(self.n_term()):
            column[k] /= self.expansion.get_mode_weight(k)
        return column


class LagrangeFR(LagrangeDG):
    """Element for implement the FR scheme using a Lagrange expansion.
    """

    def __init__(self, equation: Equation, degree: int,
            coordinate: Coordinate, value_type=float) -> None:
        LagrangeDG.__init__(self, equation, degree, coordinate, value_type)
        self._correction = Vincent(degree, Vincent.huyhn_lump_lobatto)

    def get_correction_gradients(self, x_global):
        x_local = self.coordinate.global_to_local(x_global)
        left, right = self._correction.local_to_gradient(x_local)
        jacobian = self.coordinate.local_to_jacobian(x_local)
        left /= jacobian
        right /= jacobian
        return left, right

    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscous=0.0):
        """Get the value of the reconstructed continuous flux at a given point.
        """
        flux = self.get_discontinuous_flux(x_global)
        x_local = self.coordinate.global_to_local(x_global)
        left, right = self._correction.local_to_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return flux

    def _get_flux_gradient(self, x_global, extra_viscous):
        """Get the gradient value of the discontinuous flux at a given point.
        """
        basis_gradients = self.get_basis_gradients(x_global)
        flux_gradient = 0.0
        i_sample = 0
        assert isinstance(self.expansion, expansion.Lagrange)
        for x_sample in self.expansion.get_sample_points():
            u_sample = self.get_solution_value(x_sample)
            du_sample = self.get_solution_gradient(x_sample)
            f_sample = self._equation.get_convective_flux(u_sample)
            f_sample -= self._equation.get_diffusive_flux(u_sample, du_sample)
            f_sample -= extra_viscous * du_sample
            flux_gradient += f_sample * basis_gradients[i_sample]
            i_sample += 1
        return flux_gradient

    def get_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscous=0.0):
        """Get the gradient value of the reconstructed continuous flux at a given point.
        """
        gradient = self._get_flux_gradient(x_global, extra_viscous)
        left, right = self.get_correction_gradients(x_global)
        gradient += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left(), extra_viscous))
        gradient += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right(), extra_viscous))
        return gradient

    def get_flux_gradients(self, upwind_flux_left, upwind_flux_right,
            extra_viscous=0.0):
        """Get the gradients of the continuous flux at all nodes.
        """
        assert isinstance(self.expansion, expansion.Lagrange)
        nodes = self.expansion.get_sample_points()
        values = np.ndarray(len(nodes), self._value_type)
        for i in range(len(nodes)):
            values[i] = self.get_flux_gradient(nodes[i],
                upwind_flux_left, upwind_flux_right, extra_viscous)
        return values


class LegendreFR(LegendreDG):
    """Element for implement the FR scheme using a Legendre expansion.
    """

    def __init__(self, equation: Equation, degree: int,
            coordinate: Coordinate, value_type=float) -> None:
        LegendreDG.__init__(self, equation, degree, coordinate, value_type)
        self._correction = Vincent(degree, Vincent.huyhn_lump_lobatto)

    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscous=0.0):
        """Get the value of the reconstructed continuous flux at a given point.
        """
        flux = self.get_discontinuous_flux(x_global)
        x_local = self.coordinate.global_to_local(x_global)
        left, right = self._correction.local_to_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return flux

    def get_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscous=0.0):
        """Get the gradient value of the reconstructed continuous flux at a given point.
        """
        u_approx = self.get_solution_value(x_global)
        a_approx = self._equation.get_convective_speed(u_approx)
        gradient = a_approx * self.expansion.global_to_gradient(x_global)
        x_local = self.coordinate.global_to_local(x_global)
        left, right = self._correction.global_to_gradient(x_local)
        left /= self.coordinate.global_to_jacobian(x_global)
        right /= self.coordinate.global_to_jacobian(x_global)
        gradient += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        gradient += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return gradient

    def get_flux_gradients(self, upwind_flux_left, upwind_flux_right,
            extra_viscous=0.0):
        """Get the gradients of the continuous flux at all modes.
        """
        # TODO: project grad-correction on the Legendre basis in init.
        def integrand(x_global):
            column = self.get_basis_values(x_global) * self.get_flux_gradient(
                x_global, upwind_flux_left, upwind_flux_right, extra_viscous)
            column = self.divide_mass_matrix(column)
            return column
        values = self.expansion._integrator.fixed_quad_global(
            integrand, self.n_term())
        return values


if __name__ == '__main__':
    pass
