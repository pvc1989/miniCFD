"""Implement elements for spatial scheme.
"""
import abc
import numpy as np

from concept import Expansion, Element, RiemannSolver, Coordinate
from polynomial import Vincent
from expansion import Lagrange as LagrangeExpansion
from expansion import GaussLagrange as GaussLagrangeExpansion
from expansion import Legendre as LegendreExpansion


class DiscontinuousGalerkin(Element):
    """Element for implement DG schemes.
    """

    # See Cockburn and Shu, "Runge–Kutta Discontinuous Galerkin Methods for Convection-Dominated Problems", Journal of Scientific Computing 16, 3 (2001), pp. 173--261.
    _cfl = np.ones((9, 5))
    _cfl[:, 2] = (1.000, 0.333, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033)
    _cfl[:, 3] = (1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033)
    _cfl[:, 4] = (1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037)

    def __init__(self, r: RiemannSolver, e: Expansion) -> None:
        Element.__init__(self, r, e)

    def suggest_cfl(self, rk_order: int) -> float:
        return DiscontinuousGalerkin._cfl[self.degree()][rk_order]

    def get_interior_residual(self, extra_viscosity):
        """Get the residual column by evaluating the internal integral.
        """
        def integrand(x_global):
            return np.tensordot(
                self.get_basis_gradients(x_global),
                self.get_discontinuous_flux(x_global, extra_viscosity),
                0)
        return self.fixed_quad_global(integrand, self.degree())

    def add_interface_residual(self, extra_viscosity, left_flux, right_flux, residual):
        residual += np.tensordot(
            self.get_basis_values(self.x_left()), left_flux, 0)
        residual -= np.tensordot(
            self.get_basis_values(self.x_right()), right_flux, 0)

    def add_interface_correction(self, left_jump, right_jump, residual: np.ndarray):
        correction = self.get_interface_correction(left_jump, right_jump)
        residual -= correction


class LagrangeDG(DiscontinuousGalerkin):
    """Element for implement the DG scheme using a Lagrange expansion.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LagrangeExpansion(degree, coordinate, riemann.value_type())
        DiscontinuousGalerkin.__init__(self, riemann, e)
        self._mass_matrix = self._build_mass_matrix()

    def expansion(self) -> LagrangeExpansion:
        return Element.expansion(self)

    def divide_mass_matrix(self, column: np.ndarray):
        return np.linalg.solve(self._mass_matrix, column)

    def get_sample_points(self) -> np.ndarray:
        return self.expansion().get_sample_points()

    def get_sample_values(self) -> np.ndarray:
        return self.expansion().get_sample_values()


class GaussLagrangeDG(LagrangeDG):
    """Specialized LagrangeDG element using Gaussian quadrature points as nodes.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = GaussLagrangeExpansion(degree, coordinate, riemann.value_type())
        DiscontinuousGalerkin.__init__(self, riemann, e)
        self._mass_matrix_diag = np.ndarray(self.n_term())
        self._basis_gradients = np.ndarray(self.n_term(), np.ndarray)
        points = self.get_sample_points()
        for k in range(self.n_term()):
            self._mass_matrix_diag[k] = self.expansion().get_sample_weight(k)
            self._basis_gradients[k] = self.get_basis_gradients(points[k])

    def expansion(self) -> GaussLagrangeExpansion:
        return Element.expansion(self)

    def divide_mass_matrix(self, column: np.ndarray):
        for k in range(self.n_term()):
            column[k] /= self._mass_matrix_diag[k]
        return column

    def get_interior_residual(self, extra_viscosity):
        """Get the residual column by evaluating the internal integral.

        For GaussLagrangeDG, which is a spetral element scheme, the integral can be reduced to a weighted sum of nodal values.
        """
        points = self.get_sample_points()
        values = self.get_sample_values()
        gauss = self.expansion()
        residual = np.tensordot(self._basis_gradients[0],
            self.get_discontinuous_flux(points[0], extra_viscosity,
                values[0], values.dot(self._basis_gradients[0])),
            0) * gauss.get_sample_weight(0)
        for k in range(1, self.n_term()):
            residual += np.tensordot(self._basis_gradients[k],
                self.get_discontinuous_flux(points[k], extra_viscosity,
                    values[k], values.dot(self._basis_gradients[k])),
                0) * gauss.get_sample_weight(k)
        return residual


class LegendreDG(DiscontinuousGalerkin):
    """Element for implement the DG scheme based on Legendre polynomials.

    Since flux is not explicitly approximated, it's just a wrapper of the
    LegendreExpansion class.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LegendreExpansion(degree, coordinate, riemann.value_type())
        DiscontinuousGalerkin.__init__(self, riemann, e)

    def expansion(self) -> LegendreExpansion:
        return Element.expansion(self)

    def divide_mass_matrix(self, column: np.ndarray):
        for k in range(self.n_term()):
            column[k] /= self.expansion().get_mode_weight(k)
        return column


class FluxReconstruction(Element):

    def __init__(self, r: RiemannSolver, e: Expansion) -> None:
        Element.__init__(self, r, e)
        self._correction = Vincent(e.degree(), Vincent.huyhn_lump_lobatto)

    def suggest_cfl(self, rk_order: int) -> float:
        return 2 * DiscontinuousGalerkin.suggest_cfl(self, rk_order)

    def get_correction_gradients(self, x_global):
        """Get the gradients of the correction functions at a given point.
        """
        x_local = self.coordinate().global_to_local(x_global)
        left, right = self._correction.local_to_gradient(x_local)
        jacobian = self.coordinate().local_to_jacobian(x_local)
        left /= jacobian
        right /= jacobian
        return left, right

    @abc.abstractmethod
    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscosity=0.0):
        """Get the value of the reconstructed continuous flux at a given point.
        """

    @abc.abstractmethod
    def get_continuous_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscosity=0.0):
        """Get the gradient value of the reconstructed continuous flux at a given point.
        """


class LagrangeFR(FluxReconstruction):
    """Element for implement the FR scheme using a Lagrange expansion.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LagrangeExpansion(degree, coordinate, riemann.value_type())
        FluxReconstruction.__init__(self, riemann, e)
        self._disspation_matrices = None

    def expansion(self) -> LagrangeExpansion:
        return Element.expansion(self)

    def get_sample_points(self) -> np.ndarray:
        return LagrangeDG.get_sample_points(self)

    def divide_mass_matrix(self, column: np.ndarray):
        return column

    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscosity=0.0):
        flux = self.get_discontinuous_flux(x_global, extra_viscosity)
        x_local = self.coordinate().global_to_local(x_global)
        left, right = self._correction.local_to_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left(), extra_viscosity))
        flux += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right(), extra_viscosity))
        return flux

    def get_discontinuous_flux_gradient(self, x_global, extra_viscosity):
        """Get the gradient value of the discontinuous flux at a given point.
        """
        basis_gradients = self.get_basis_gradients(x_global)
        flux_gradient = 0.0
        i_sample = 0
        for x_sample in self.expansion().get_sample_points():
            u_sample = self.get_solution_value(x_sample)
            du_sample = self.get_solution_gradient(x_sample)
            f_sample = self.equation().get_convective_flux(u_sample)
            f_sample -= self.equation().get_diffusive_flux(u_sample, du_sample)
            if callable(extra_viscosity):
                f_sample -= du_sample * extra_viscosity(x_sample)
            else:
                f_sample -= du_sample * extra_viscosity
            flux_gradient += f_sample * basis_gradients[i_sample]
            i_sample += 1
        return flux_gradient

    def get_interior_residual(self, extra_viscosity=0) -> np.ndarray:
        residual = np.ndarray(self.n_term(), self.value_type())
        points = self.expansion().get_sample_points()
        for i_sample in range(self.n_term()):
            x_sample = points[i_sample]
            residual[i_sample] = \
                -self.get_discontinuous_flux_gradient(x_sample, extra_viscosity)
        return residual

    def get_continuous_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscosity=0.0):
        left_flux_gap = upwind_flux_left - \
            self.get_discontinuous_flux(self.x_left(), extra_viscosity)
        right_flux_gap = upwind_flux_right - \
            self.get_discontinuous_flux(self.x_right(), extra_viscosity)
        gradient = self.get_discontinuous_flux_gradient(x_global, extra_viscosity)
        left_grad, right_grad = self.get_correction_gradients(x_global)
        gradient += left_grad * left_flux_gap
        gradient += right_grad * right_flux_gap
        return gradient

    def add_interface_residual(self, extra_viscosity, upwind_flux_left, upwind_flux_right, residual: np.ndarray):
        left_flux_gap = upwind_flux_left - \
            self.get_discontinuous_flux(self.x_left(), extra_viscosity)
        right_flux_gap = upwind_flux_right - \
            self.get_discontinuous_flux(self.x_right(), extra_viscosity)
        points = self.expansion().get_sample_points()
        for i_sample in range(self.n_term()):
            x_global = points[i_sample]
            left_grad, right_grad = self.get_correction_gradients(x_global)
            residual[i_sample] -= left_grad * left_flux_gap
            residual[i_sample] -= right_grad * right_flux_gap
        return residual

    def get_dissipation_matrices(self):
        left, right = self.neighbor_expansions()
        n_term = self.n_term()
        shape = (n_term, n_term)
        points = self.get_sample_points()
        mat_a = np.ndarray(shape)
        for i in range(n_term):
            mat_a[i] = self.get_basis_gradients(points[i])
        mat_b = np.zeros(shape)
        for k in range(n_term):
            for l in range(n_term):
                mat_b[k] += mat_a[k][l] * mat_a[l]
        # print('B =\n', mat_b)
        basis_values_curr_left = self.get_basis_values(self.x_left())
        basis_values_curr_right = self.get_basis_values(self.x_right())
        correction_gradients_left = np.ndarray(n_term)
        correction_gradients_right = np.ndarray(n_term)
        for i in range(n_term):
            correction_gradients_left[i], correction_gradients_right[i] \
                = self.get_correction_gradients(points[i])
        mat_c = np.zeros(shape)
        for k in range(n_term):
            for l in range(n_term):
                a = correction_gradients_right[k] * basis_values_curr_right[l]
                a += correction_gradients_left[k] * basis_values_curr_left[l]
                mat_c[k] += a * mat_a[l]
        # print('C =\n', mat_c)
        d_minus_left = 0.0
        if left:
            d_minus_left = self._riemann.get_interface_gradient(
                left.length(), self.length(),
                u_jump = +basis_values_curr_left,
                du_mean = 0.5 * self.get_basis_gradients(self.x_left()),
                ddu_jump = +self.get_basis_hessians(self.x_left()))
        d_minus_right = 0.0
        if right:
            d_minus_right = self._riemann.get_interface_gradient(
                self.length(), right.length(),
                u_jump = -basis_values_curr_right,
                du_mean = 0.5 * self.get_basis_gradients(self.x_right()),
                ddu_jump = -self.get_basis_hessians(self.x_right()))
        mat_d = np.ndarray(shape)
        for k in range(n_term):
            mat_d[k] = correction_gradients_right[k] * d_minus_right
            mat_d[k] += correction_gradients_left[k] * d_minus_left
        # print('D- =\n', mat_d)
        # print('D =\n', mat_b - mat_c + mat_d)
        d_plus_left = 0.0
        if left:
            d_plus_left = self._riemann.get_interface_gradient(
                left.length(), self.length(),
                u_jump = -left.get_basis_values(left.x_right()),
                du_mean = 0.5 * left.get_basis_gradients(left.x_right()),
                ddu_jump = -left.get_basis_hessians(left.x_right()))
        d_plus_right = 0.0
        if right:
            d_plus_right = self._riemann.get_interface_gradient(
                self.length(), right.length(),
                u_jump = +right.get_basis_values(right.x_left()),
                du_mean = 0.5 * right.get_basis_gradients(right.x_left()),
                ddu_jump = +right.get_basis_hessians(right.x_left()))
        mat_e = np.ndarray(shape)
        mat_f = np.ndarray(shape)
        for k in range(n_term):
            mat_e[k] = correction_gradients_left[k] * d_plus_left
            mat_f[k] = correction_gradients_right[k] * d_plus_right
        # print('E =\n', mat_e)
        # print('F =\n', mat_f)
        return mat_b, mat_c, mat_d, mat_e, mat_f


class GaussLagrangeFR(LagrangeFR):
    """Element for implement the FR scheme using a GaussLagrange expansion.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = GaussLagrangeExpansion(degree, coordinate, riemann.value_type())
        FluxReconstruction.__init__(self, riemann, e)
        self._disspation_matrices = None
        self._basis_gradients = np.ndarray(self.n_term(), np.ndarray)
        self._correction_gradients = np.ndarray(self.n_term(), tuple)
        points = self.expansion().get_sample_points()
        i_sample = 0
        for x_global in points:
            self._basis_gradients[i_sample] = self.get_basis_gradients(x_global)
            self._correction_gradients[i_sample] = \
                self.get_correction_gradients(x_global)
            i_sample += 1
        assert i_sample == self.n_term()

    def expansion(self) -> GaussLagrangeExpansion:
        return Element.expansion(self)

    def get_interior_residual(self, extra_viscosity):
        residual = np.ndarray(self.n_term(), self.value_type())
        u_samples = self.expansion().get_coeff_ref()
        # build gradients of u at sample points
        du_samples = np.ndarray(self.n_term(), self.value_type())
        for i in range(self.n_term()):
            du_samples[i] = u_samples.dot(self._basis_gradients[i])
        x_samples = self.expansion().get_sample_points()
        # build values of f at sample points
        f_samples = np.ndarray(self.n_term(), self.value_type())
        for i in range(self.n_term()):
            f_samples[i] = self.get_discontinuous_flux(
                x_samples[i], extra_viscosity, u_samples[i], du_samples[i])
        # build gradients of f at sample points
        for i in range(self.n_term()):
            residual[i] = -f_samples.dot(self._basis_gradients[i])
        return residual

    def add_interface_residual(self, extra_viscosity, upwind_flux_left, upwind_flux_right, residual: np.ndarray):
        left_flux_gap = upwind_flux_left - \
            self.get_discontinuous_flux(self.x_left(), extra_viscosity)
        right_flux_gap = upwind_flux_right - \
            self.get_discontinuous_flux(self.x_right(), extra_viscosity)
        for i_sample in range(self.n_term()):
            left, right = self._correction_gradients[i_sample]
            residual[i_sample] -= left * left_flux_gap
            residual[i_sample] -= right * right_flux_gap
        return residual

    def add_interface_correction(self, left_jump, right_jump, residual: np.ndarray):
        return
        correction = self.get_interface_correction(left_jump, right_jump)
        e = self.expansion()
        for k in range(self.n_term()):
            residual[k] -= correction[k] / e.get_sample_weight(k)

    def get_dissipation_rate(self):
        if not self._disspation_matrices:
            mat_b, mat_c, mat_d, mat_e, mat_f = self.get_dissipation_matrices()
            mat_d += mat_b - mat_c
            mat_w = np.eye(self.n_term())
            for k in range(self.n_term()):
                mat_w[k][k] = self.expansion().get_sample_weight(k)
            self._disspation_matrices = (mat_w@mat_d, mat_w@mat_e, mat_w@mat_f)
        mat_d, mat_e, mat_f = self._disspation_matrices
        curr_column = self.get_solution_column()
        dissipation = mat_d @ curr_column
        # left, right = self.neighbor_expansions()
        # dissipation += mat_e @ left.get_coeff_ref()
        # dissipation += mat_f @ right.get_coeff_ref()
        dissipation = curr_column.transpose() @ dissipation
        assert dissipation < 0
        # print(f'dissipation = {dissipation:.2e}')
        return dissipation


class LegendreFR(FluxReconstruction):
    """Element for implement the FR scheme using a Legendre expansion.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LegendreExpansion(degree, coordinate, riemann.value_type())
        FluxReconstruction.__init__(self, riemann, e)

    def expansion(self) -> LegendreExpansion:
        return Element.expansion(self)

    def divide_mass_matrix(self, column: np.ndarray):
        return LegendreDG.divide_mass_matrix(self, column)

    def get_continuous_flux(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscosity=0.0):
        flux = self.get_discontinuous_flux(x_global)
        x_local = self.coordinate().global_to_local(x_global)
        left, right = self._correction.local_to_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return flux

    def get_continuous_flux_gradient(self, x_global, upwind_flux_left, upwind_flux_right,
            extra_viscosity=0.0):
        u_approx = self.get_solution_value(x_global)
        a_approx = self.equation().get_convective_jacobian(u_approx)
        gradient = a_approx @ self.expansion().global_to_gradient(x_global)
        x_local = self.coordinate().global_to_local(x_global)
        left, right = self._correction.local_to_gradient(x_local)
        left /= self.coordinate().global_to_jacobian(x_global)
        right /= self.coordinate().global_to_jacobian(x_global)
        gradient += left * (upwind_flux_left
            - self.get_discontinuous_flux(self.x_left()))
        gradient += right * (upwind_flux_right
            - self.get_discontinuous_flux(self.x_right()))
        return gradient

    def get_flux_gradients(self, upwind_flux_left, upwind_flux_right,
            extra_viscosity=0.0):
        # TODO: project grad-correction on the Legendre basis in init.
        def integrand(x_global):
            column = self.get_basis_values(x_global) * self.get_flux_gradient(
                x_global, upwind_flux_left, upwind_flux_right, extra_viscosity)
            column = self.divide_mass_matrix(column)
            return column
        values = self.expansion().integrator().fixed_quad_global(
            integrand, self.n_term())
        return values


if __name__ == '__main__':
    pass
