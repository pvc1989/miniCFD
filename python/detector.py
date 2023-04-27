"""Concrete implementations of jump detectors.
"""
import numpy as np
import abc

import concept
from spatial import FiniteElement
import integrate
import expansion


class Dummy(concept.JumpDetector):
    """A jump detector that reports no cell as troubled.
    """

    def name(self, verbose=False):
        return 'Dummy'

    def get_troubled_cell_indices(self, scheme: FiniteElement):
        return []


class ReportAll(concept.JumpDetector):
    """A jump detector that reports all cells as troubled.
    """

    def name(self, verbose=False):
        return 'ReportAll'

    def get_troubled_cell_indices(self, scheme: FiniteElement):
        troubled_cell_indices = np.arange(scheme.n_element(), dtype=int)
        return troubled_cell_indices


class SmoothnessBased(concept.JumpDetector):

    @abc.abstractmethod
    def get_smoothness_values(self, scheme: FiniteElement) -> np.ndarray:
        """Get the values of smoothness for each element.
        """

    def max_smoothness(self, scheme: FiniteElement) -> float:
        """Get the max value of the smoothness of a smooth cell.
        """
        return 1.0

    def get_troubled_cell_indices(self, scheme: FiniteElement):
        smoothness_values = self.get_smoothness_values(scheme)
        troubled_cell_indices = []
        for i_cell in range(len(smoothness_values)):
            if smoothness_values[i_cell] > self.max_smoothness(scheme):
                troubled_cell_indices.append(i_cell)
        return troubled_cell_indices


class Krivodonova2004(SmoothnessBased):
    """A jump detector for high-order DG schemes.

    See Krivodonova et al., "Shock detection and limiting with discontinuous Galerkin methods for hyperbolic conservation laws", Applied Numerical Mathematics 48, 3-4 (2004), pp. 323--338.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Krivodonova et al. (2004)'
        else:
            return 'KXCRF (2004)'

    def get_smoothness_values(self, scheme: FiniteElement) -> np.ndarray:
        n_cell = scheme.n_element()
        norms = np.ndarray(n_cell)
        for i_cell in range(n_cell):
            cell = scheme.get_element_by_index(i_cell)
            def function(x_global):
                return cell.get_solution_value(x_global)
            norms[i_cell] = integrate.norm_infty(function, cell) + 1.0
        ratio = scheme.delta_x()**((cell.degree() + 1) / 2)
        smoothness = np.ndarray(n_cell)
        # always report boundary cells as troubled
        smoothness[0] = smoothness[-1] = 100.0
        for i_curr in range(1, n_cell-1):
            curr = scheme.get_element_by_index(i_curr)
            i_prev = i_curr - 1
            prev = scheme.get_element_by_index(i_prev)
            i_next = (i_curr + 1) % n_cell
            next = scheme.get_element_by_index(i_next)
            # evaluate smoothness
            dividend = 0.0
            if curr.get_convective_jacobian(curr.x_left()) > 0:
                dividend += (curr.get_solution_value(curr.x_left())
                    - prev.get_solution_value(prev.x_right()))
            if curr.get_convective_jacobian(curr.x_right()) < 0:
                dividend += (next.get_solution_value(next.x_left())
                    - curr.get_solution_value(curr.x_right()))
            dividend = np.abs(dividend)
            divisor = ratio * norms[i_curr]
            smoothness[i_curr] = dividend / divisor
        return smoothness


class LiRen2011(SmoothnessBased):
    """A jump detector for high-order finite volume schemes.

    See Li and Ren, "High-order k-exact WENO finite volume schemes for solving gas dynamic Euler equations on unstructured grids", International Journal for Numerical Methods in Fluids 70, 6 (2011), pp. 742--763.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Li–Ren (2011)'
        else:
            return 'Li (2011)'

    def get_smoothness_values(self, scheme: FiniteElement) -> np.ndarray:
        n_cell = scheme.n_element()
        averages = np.ndarray(n_cell)  # could be easier for some schemes
        for i_cell in range(n_cell):
            cell = scheme.get_element_by_index(i_cell)
            def function(x_global):
                return cell.get_solution_value(x_global)
            average = integrate.average(function, cell)
            averages[i_cell] = np.abs(average) + cell.length()
        ratio = 2 * scheme.delta_x()**((cell.degree() + 1) / 2)
        smoothness = np.ndarray(n_cell)
        # always report boundary cells as troubled
        smoothness[0] = smoothness[-1] = 100.0
        for i_curr in range(1, n_cell-1):
            curr = scheme.get_element_by_index(i_curr)
            curr_solution = curr.get_solution_value(curr.x_center())
            i_prev = i_curr - 1
            prev = scheme.get_element_by_index(i_prev)
            prev_solution = prev.get_solution_value(curr.x_center()
                + scheme.length() * (i_curr == 0))
            i_next = (i_curr + 1) % n_cell
            next = scheme.get_element_by_index(i_next)
            next_solution = next.get_solution_value(curr.x_center()
                - scheme.length() * (i_next == 0))
            # evaluate smoothness
            dividend = (np.abs(curr_solution - next_solution)
                + np.abs(curr_solution - prev_solution))
            divisor = ratio * max(averages[i_curr], averages[i_prev],
                averages[i_next])
            # print(dividend, divisor)
            smoothness[i_curr] = dividend / divisor
        return smoothness

    def max_smoothness(self, scheme: FiniteElement):
        return 1 + 2*(scheme.degree() > 2)


class ZhuShuQiu2021(SmoothnessBased):
    """A jump detector for high-order DG schemes.

    See Zhu and Shu and Qiu, "High-order Runge-Kutta discontinuous Galerkin methods with multi-resolution WENO limiters for solving steady-state problems", Applied Numerical Mathematics 165 (2021), pp. 482--499.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Zhu–Shu–Qiu (2011)'
        else:
            return 'Zhu (2021)'

    def get_smoothness_values(self, scheme: FiniteElement) -> np.ndarray:
        n_cell = scheme.n_element()
        norms = np.ndarray(n_cell)
        for i_cell in range(n_cell):
            cell = scheme.get_element_by_index(i_cell)
            def function(x_global):
                return cell.get_solution_value(x_global)
            norms[i_cell] = integrate.norm_1(function, cell) + cell.length()
        smoothness = np.ndarray(n_cell)
        # always report boundary cells as troubled
        smoothness[0] = smoothness[-1] = 100.0
        for i_curr in range(1, n_cell-1):
            curr = scheme.get_element_by_index(i_curr)
            def curr_solution(x_global):
                return curr.get_solution_value(x_global)
            i_prev = i_curr - 1
            prev = scheme.get_element_by_index(i_prev)
            def prev_solution(x_global):
                return prev.get_solution_value(x_global
                    + scheme.length() * (i_curr == 0))
            i_next = (i_curr + 1) % n_cell
            next = scheme.get_element_by_index(i_next)
            def next_solution(x_global):
                return next.get_solution_value(x_global
                    - scheme.length() * (i_next == 0))
            # evaluate smoothness
            dividend = max(
                self._integrate(curr_solution, prev_solution, curr),
                self._integrate(curr_solution, next_solution, curr))
            divisor = min(norms[i_curr], norms[i_prev],
                norms[i_next]) * scheme.delta_x()
            # print(dividend, divisor)
            smoothness[i_curr] = dividend / divisor
        return smoothness

    def _integrate(self, f, g, cell):
        value = integrate.fixed_quad_global(lambda x: f(x) - g(x),
            cell.x_left(), cell.x_right(), cell.degree())
        return np.abs(value)


class LiRen2022(concept.JumpDetector):
    """A jump detector for high-order FD schemes.

    See Li Yanhui, Chen Congwei, and Ren Yu-Xin, "A class of high-order finite difference schemes with minimized dispersion and adaptive dissipation for solving compressible flo…", Journal of Computational Physics 448 (2022), pp. 110770.
    """

    def __init__(self) -> None:
        self._xi = 0.01
        self._psi_c = 0.4
        self._epsilon = (0.9 * self._psi_c) / (1 - 0.9 * self._psi_c)
        self._epsilon *= self._xi**2

    def name(self, verbose=False):
        if verbose:
            return 'Li–Ren (2022)'
        else:
            return 'Li (2022)'

    def get_troubled_cell_indices(self, scheme: FiniteElement) -> np.ndarray:
        troubled_cell_indices = []
        n_cell = scheme.n_element()
        averages = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            expansion = scheme.get_element_by_index(i_curr).get_expansion()
            averages[i_curr] = expansion.get_average()
        psi_values = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            a = np.abs(averages[i_curr] - averages[i_curr-1])
            a += np.abs(averages[i_curr] - 2 * averages[i_curr-1]
                + averages[i_curr-2])
            b = np.abs(averages[i_curr] - averages[(i_curr+1)%n_cell])
            b += np.abs(averages[i_curr] - 2 * averages[(i_curr+1)%n_cell]
                + averages[(i_curr+2)%n_cell])
            psi_values[i_curr] = (2 * a * b + self._epsilon) / (
                a**2 + b**2 + self._epsilon)
        for i_curr in range(n_cell):
            left = min(psi_values[i_curr], psi_values[i_curr-1])
            right = min(psi_values[i_curr], psi_values[(i_curr+1)%n_cell])
            if min(left, right) < self._psi_c:
                troubled_cell_indices.append(i_curr)
        return troubled_cell_indices

    def get_smoothness_values(self, scheme: FiniteElement) -> np.ndarray:
        troubled_cell_indices = self.get_troubled_cell_indices(scheme)
        n_cell = scheme.n_element()
        smoothness_values = np.zeros(n_cell) + 1e-2
        for i_cell in troubled_cell_indices:
            smoothness_values[i_cell] = 1e2
        return smoothness_values


class Persson2006(SmoothnessBased):
    """A jump detector based on artificial viscosity.

    See Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006).
    """

    def name(self, verbose=False):
        return 'Persson (2006)'

    def get_smoothness_value(self, u_approx: expansion.Taylor):
        if isinstance(u_approx, expansion.Legendre):
            pth_mode_energy = u_approx.get_mode_energy(-1)
            all_modes_energy = 0.0
            for k in range(u_approx.n_term()):
                all_modes_energy += u_approx.get_mode_energy(k)
        else:
            assert isinstance(u_approx, expansion.Taylor)
            def u(x):
                return u_approx.get_function_value(x)
            all_modes_energy = integrate.inner_product(u, u, u_approx)
            legendre = expansion.Legendre(u_approx.degree(),
                u_approx.x_left(), u_approx.x_right())
            pth_basis = legendre.get_basis(u_approx.degree())
            pth_mode_energy = integrate.inner_product(u, pth_basis, u_approx)**2
            pth_mode_energy /= legendre.get_mode_weight(u_approx.degree())
        return pth_mode_energy / all_modes_energy

    def get_smoothness_values(self, scheme: FiniteElement) -> np.ndarray:
        n_cell = scheme.n_element()
        smoothness = np.ndarray(n_cell)
        sensor_ref = scheme.degree()**(-3)
        for i_cell in range(n_cell):
            cell = scheme.get_element_by_index(i_cell)
            u_approx = cell.get_expansion()
            sensor = self.get_smoothness_value(u_approx)
            smoothness[i_cell] = sensor / sensor_ref
        return smoothness


if __name__ == '__main__':
    pass
