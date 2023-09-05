"""Implement elements for spatial scheme.
"""
import abc
import numpy as np

from concept import Expansion, Element, RiemannSolver, Coordinate
from polynomial import Vincent
from expansion import LagrangeOnUniformRoots
from expansion import LagrangeOnLegendreRoots
from expansion import Legendre as LegendreExpansion
import expansion


class DiscontinuousGalerkin(Element):
    """Element for implement DG schemes.

    See [Cockburn and Shu, "Runge–Kutta Discontinuous Galerkin Methods for Convection-Dominated Problems", Journal of Scientific Computing 16, 3 (2001), pp. 173--261](https://doi.org/10.1023/a:1012873910884) for details.
    """
    _cfl = np.ones((9, 5))
    _cfl[:, 2] = (1.000, 0.333, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033)
    _cfl[:, 3] = (1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033)
    _cfl[:, 4] = (1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037)

    def __init__(self, r: RiemannSolver, e: Expansion) -> None:
        Element.__init__(self, r, e)

    def suggest_cfl(self, rk_order: int) -> float:
        return DiscontinuousGalerkin._cfl[self.degree()][rk_order]

    def get_interior_residual(self):
        """Get the residual column by evaluating the internal integral.
        """
        def integrand(x_global):
            return np.tensordot(
                self.get_basis_gradients(x_global),
                self.get_dg_flux(x_global),
                0)
        return self.fixed_quad_global(integrand, self.degree())

    def add_interface_residual(self, left_flux, right_flux, residual):
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

    def __init__(self, r: RiemannSolver, e: expansion.Lagrange) -> None:
        DiscontinuousGalerkin.__init__(self, r, e)
        self._mass_matrix = self._build_mass_matrix()

    def expansion(self) -> expansion.Lagrange:
        return Element.expansion(self)

    def divide_mass_matrix(self, column: np.ndarray):
        return np.linalg.solve(self._mass_matrix, column)

    def get_sample_points(self) -> np.ndarray:
        return self.expansion().get_sample_points()

    def get_sample_values(self) -> np.ndarray:
        return self.expansion().get_sample_values()


class DGonUniformRoots(DiscontinuousGalerkin):
    """Specialized LagrangeDG using uniform (internal) roots as solution points.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LagrangeOnUniformRoots(degree, coordinate, riemann.value_type())
        LagrangeDG.__init__(self, riemann, e)

    def expansion(self) -> LagrangeOnUniformRoots:
        return Element.expansion(self)


class DGonLegendreRoots(LagrangeDG):
    """Specialized LagrangeDG using Legendre roots as solution points.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LagrangeOnLegendreRoots(degree, coordinate, riemann.value_type())
        DiscontinuousGalerkin.__init__(self, riemann, e)
        self._mass_matrix_diag = np.ndarray(self.n_term())
        self._basis_gradients = np.ndarray(self.n_term(), np.ndarray)
        points = self.get_sample_points()
        for k in range(self.n_term()):
            self._mass_matrix_diag[k] = self.expansion().get_sample_weight(k)
            self._basis_gradients[k] = self.get_basis_gradients(points[k])

    def expansion(self) -> LagrangeOnLegendreRoots:
        return Element.expansion(self)

    def divide_mass_matrix(self, column: np.ndarray):
        for k in range(self.n_term()):
            column[k] /= self._mass_matrix_diag[k]
        return column

    def get_interior_residual(self):
        """Get the residual column by evaluating the internal integral.

        For DGonLegendreRoots, which is a spetral element scheme, the integral can be reduced to a weighted sum of nodal values.
        """
        points = self.get_sample_points()
        values = self.get_sample_values()
        gauss = self.expansion()
        residual = np.tensordot(self._basis_gradients[0],
            self.get_dg_flux(points[0],
                values[0], values.dot(self._basis_gradients[0])),
            0) * gauss.get_sample_weight(0)
        for k in range(1, self.n_term()):
            residual += np.tensordot(self._basis_gradients[k],
                self.get_dg_flux(points[k],
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
    """Element for implement FR schemes.

    See [Huynh, "A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods", in 18th AIAA Computational Fluid Dynamics Conference (Miami, Florida, USA: American Institute of Aeronautics and Astronautics, 2007)](https://doi.org/10.2514/6.2007-4079) and [Vincent and Castonguay and Jameson, "A New Class of High-Order Energy Stable Flux Reconstruction Schemes", Journal of Scientific Computing 47, 1 (2010), pp. 50--72.](https://doi.org/10.1007/s10915-010-9420-z) for details.
    """

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
    def get_fr_flux(self, x_global, upwind_flux_left, upwind_flux_right):
        """Get the value of the reconstructed continuous flux at a given point.
        """

    @abc.abstractmethod
    def get_fr_flux_gradient(self, x_global,
            upwind_flux_left, upwind_flux_right):
        """Get the gradient of the reconstructed continuous flux at a given point.
        """


class LagrangeFR(FluxReconstruction):
    """Element for implement the FR scheme using a Lagrange expansion.
    """

    def __init__(self, r: RiemannSolver, e: expansion.Lagrange) -> None:
        FluxReconstruction.__init__(self, r, e)
        self._disspation_matrices = None

    def expansion(self) -> expansion.Lagrange:
        return Element.expansion(self)

    def get_sample_points(self) -> np.ndarray:
        return LagrangeDG.get_sample_points(self)

    def divide_mass_matrix(self, column: np.ndarray):
        return column

    def get_fr_flux(self, x_global,
            upwind_flux_left, upwind_flux_right):
        flux = self.get_dg_flux(x_global)
        x_local = self.coordinate().global_to_local(x_global)
        left, right = self._correction.local_to_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_dg_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_dg_flux(self.x_right()))
        return flux

    def get_dg_flux_gradient(self, x_global):
        """Get the gradient value of the discontinuous flux at a given point.
        """
        basis_gradients = self.get_basis_gradients(x_global)
        flux_gradient = 0.0
        i_sample = 0
        for x_sample in self.expansion().get_sample_points():
            f_sample = self.get_dg_flux(x_sample,
                u_given=self.get_solution_value(x_sample),
                du_given=self.get_solution_gradient(x_sample))
            flux_gradient += f_sample * basis_gradients[i_sample]
            i_sample += 1
        return flux_gradient

    def get_interior_residual(self) -> np.ndarray:
        residual = np.ndarray(self.n_term(), self.value_type())
        points = self.expansion().get_sample_points()
        for i_sample in range(self.n_term()):
            x_sample = points[i_sample]
            residual[i_sample] = \
                -self.get_dg_flux_gradient(x_sample)
        return residual

    def get_fr_flux_gradient(self, x_global,
            upwind_flux_left, upwind_flux_right):
        left_flux_gap = upwind_flux_left - \
            self.get_dg_flux(self.x_left())
        right_flux_gap = upwind_flux_right - \
            self.get_dg_flux(self.x_right())
        gradient = self.get_dg_flux_gradient(x_global)
        left_grad, right_grad = self.get_correction_gradients(x_global)
        gradient += left_grad * left_flux_gap
        gradient += right_grad * right_flux_gap
        return gradient

    def add_interface_residual(self, upwind_flux_left, upwind_flux_right,
            residual: np.ndarray):
        left_flux_gap = upwind_flux_left - \
            self.get_dg_flux(self.x_left())
        right_flux_gap = upwind_flux_right - \
            self.get_dg_flux(self.x_right())
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


class FRonUniformRoots(LagrangeFR):
    """Specialized LagrangeFR using uniform (internal) roots as solution points.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LagrangeOnUniformRoots(degree, coordinate, riemann.value_type())
        LagrangeFR.__init__(self, riemann, e)
        self._disspation_matrices = None

    def expansion(self) -> LagrangeOnUniformRoots:
        return Element.expansion(self)


class FRonLegendreRoots(LagrangeFR):
    """Specialized LagrangeFR using Legendre roots as solution points.
    """

    def __init__(self, riemann: RiemannSolver, degree: int,
            coordinate: Coordinate) -> None:
        e = LagrangeOnLegendreRoots(degree, coordinate, riemann.value_type())
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

    def expansion(self) -> LagrangeOnLegendreRoots:
        return Element.expansion(self)

    def get_interior_residual(self):
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
            f_samples[i] = self.get_dg_flux(
                x_samples[i], u_samples[i], du_samples[i])
        # build gradients of f at sample points
        for i in range(self.n_term()):
            residual[i] = -f_samples.dot(self._basis_gradients[i])
        return residual

    def add_interface_residual(self, upwind_flux_left, upwind_flux_right,
            residual: np.ndarray):
        left_flux_gap = upwind_flux_left - self.get_dg_flux(self.x_left())
        right_flux_gap = upwind_flux_right - self.get_dg_flux(self.x_right())
        for i_sample in range(self.n_term()):
            left, right = self._correction_gradients[i_sample]
            residual[i_sample] -= left * left_flux_gap
            residual[i_sample] -= right * right_flux_gap
        return residual

    def add_interface_correction(self, left_jump, right_jump,
            residual: np.ndarray):
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
        n_component = self.equation().n_component()
        if n_component == 1:
            column = self.get_solution_column()
            dissipation = mat_d @ column
            # left, right = self.neighbor_expansions()
            # dissipation += mat_e @ left.get_coeff_ref()
            # dissipation += mat_f @ right.get_coeff_ref()
            dissipation = column.transpose() @ dissipation
            assert dissipation < 0, dissipation
        else:
            L, _ = self.get_convective_eigmats()
            U = self.expansion().get_sample_values()
            V = np.ndarray((self.n_term(), n_component))
            for i_node in range(self.n_term()):
                V[i_node, :] = L @ U[i_node]
            dissipation = np.ndarray(n_component)
            for i_comp in range(n_component):
                column = V[:, i_comp]
                dissipation_i = column.transpose() @ (mat_d @ column)
                assert dissipation_i < 0, dissipation_i
                dissipation[i_comp] = dissipation_i
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

    def get_fr_flux(self, x_global,
            upwind_flux_left, upwind_flux_right):
        flux = self.get_dg_flux(x_global)
        x_local = self.coordinate().global_to_local(x_global)
        left, right = self._correction.local_to_value(x_local)
        flux += left * (upwind_flux_left
            - self.get_dg_flux(self.x_left()))
        flux += right * (upwind_flux_right
            - self.get_dg_flux(self.x_right()))
        return flux

    def get_fr_flux_gradient(self, x_global,
            upwind_flux_left, upwind_flux_right):
        u_approx = self.get_solution_value(x_global)
        a_approx = self.equation().get_convective_jacobian(u_approx)
        gradient = a_approx @ self.expansion().global_to_gradient(x_global)
        x_local = self.coordinate().global_to_local(x_global)
        left, right = self._correction.local_to_gradient(x_local)
        left /= self.coordinate().global_to_jacobian(x_global)
        right /= self.coordinate().global_to_jacobian(x_global)
        gradient += left * (upwind_flux_left
            - self.get_dg_flux(self.x_left()))
        gradient += right * (upwind_flux_right
            - self.get_dg_flux(self.x_right()))
        return gradient

    def get_flux_gradients(self, upwind_flux_left, upwind_flux_right):
        # TODO: project grad-correction on the Legendre basis in init.
        def integrand(x_global):
            column = self.get_basis_values(x_global) * self.get_flux_gradient(
                x_global, upwind_flux_left, upwind_flux_right)
            column = self.divide_mass_matrix(column)
            return column
        values = self.expansion().integrator().fixed_quad_global(
            integrand, self.n_term())
        return values


if __name__ == '__main__':
    pass
