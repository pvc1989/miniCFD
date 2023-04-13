"""Concrete implementations of jump detectors.
"""
import numpy as np

from concept import JumpDetector
from spatial import FiniteElement
import integrate


class ReportAll(JumpDetector):
    """A jump detector that reports all cells as troubled.
    """

    def name(self, verbose=False):
        return 'ReportAll'

    def get_smoothness_values(self, scheme: FiniteElement) -> np.ndarray:
        smoothness = np.ones(scheme.n_element()) * np.Inf
        return smoothness

    def get_troubled_cell_indices(self, scheme: FiniteElement):
        troubled_cell_indices = np.arange(scheme.n_element(), dtype=int)
        return troubled_cell_indices


class Krivodonova2004(JumpDetector):
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

    def get_troubled_cell_indices(self, scheme: FiniteElement):
        smoothness_values = self.get_smoothness_values(scheme)
        troubled_cell_indices = []
        for i_cell in range(len(smoothness_values)):
            if smoothness_values[i_cell] > 1:
                troubled_cell_indices.append(i_cell)
        return troubled_cell_indices


class LiRen2011(JumpDetector):
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

    def get_troubled_cell_indices(self, scheme: FiniteElement):
        smoothness_values = self.get_smoothness_values(scheme)
        troubled_cell_indices = []
        for i_cell in range(len(smoothness_values)):
            cell = scheme.get_element_by_index(i_cell)
            if smoothness_values[i_cell] > (1 + 2*(cell.degree() > 2)):
                troubled_cell_indices.append(i_cell)
        return troubled_cell_indices


class ZhuShuQiu2021(JumpDetector):
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

    def get_troubled_cell_indices(self, scheme: FiniteElement):
        smoothness_values = self.get_smoothness_values(scheme)
        troubled_cell_indices = []
        for i_cell in range(len(smoothness_values)):
            if smoothness_values[i_cell] > 1:
                troubled_cell_indices.append(i_cell)
        return troubled_cell_indices

    def _integrate(self, f, g, cell):
        value = integrate.fixed_quad_global(lambda x: f(x) - g(x),
            cell.x_left(), cell.x_right(), cell.degree())
        return np.abs(value)


if __name__ == '__main__':
    pass
