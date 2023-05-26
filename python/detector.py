"""Concrete implementations of jump detectors.
"""
import numpy as np
import abc

import concept
from spatial import FiniteElement
import expansion


class Off(concept.Detector):
    """A jump detector that reports no cell as troubled.
    """

    def name(self, verbose=False):
        return 'Off'

    def get_troubled_cell_indices(self, elements, periodic: bool):
        return []


class All(concept.Detector):
    """A jump detector that reports all cells as troubled.
    """

    def name(self, verbose=False):
        return 'All'

    def get_troubled_cell_indices(self, elements, periodic: bool):
        return range(len(elements))


class SmoothnessBased(concept.Detector):

    @abc.abstractmethod
    def get_smoothness_values(self, elements, periodic: bool) -> np.ndarray:
        """Get the values of smoothness for each element.
        """

    def max_smoothness(self, curr: concept.Element) -> float:
        """Get the max value of the smoothness of a smooth cell.
        """
        return 1.0

    def get_troubled_cell_indices(self, elements, periodic: bool) -> list:
        troubled_cell_indices = []
        smoothness_values = self.get_smoothness_values(elements, periodic)
        for i in range(len(elements)):
            if smoothness_values[i] > self.max_smoothness(elements[i]):
                troubled_cell_indices.append(i)
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

    def get_smoothness_values(self, elements, periodic: bool) -> np.ndarray:
        n_cell = len(elements)
        smoothness_values = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = elements[i_curr]
            assert isinstance(curr, concept.Element)
            def function(x_global):
                return curr.get_solution_value(x_global)
            curr_norm = curr.integrator().norm_infty(function, curr.n_term())
            curr_norm += 1e-8  # avoid division-by-zero
            ratio = curr.length()**((curr.degree() + 1) / 2)
            left, right = None, None
            if periodic or i_curr > 0:
                left = elements[i_curr - 1]
            if periodic or i_curr + 1 < n_cell:
                right = elements[(i_curr + 1) % n_cell]
            dividend = 0.0
            if (curr.get_convective_speed(curr.x_left()) > 0 and
                    isinstance(left, concept.Element)):
                dividend += np.abs(curr.get_solution_value(curr.x_left())
                    - left.get_solution_value(left.x_right()))
            if (curr.get_convective_speed(curr.x_right()) < 0 and
                    isinstance(right, concept.Element)):
                dividend += np.abs(curr.get_solution_value(curr.x_right())
                    - right.get_solution_value(right.x_left()))
            divisor = ratio * curr_norm
            smoothness_values[i_curr] = dividend / divisor
        return smoothness_values


class LiRen2011(SmoothnessBased):
    """A jump detector for high-order finite volume schemes.

    See Li and Ren, "High-order k-exact WENO finite volume schemes for solving gas dynamic Euler equations on unstructured grids", International Journal for Numerical Methods in Fluids 70, 6 (2011), pp. 742--763.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Li–Ren (2011)'
        else:
            return 'Li (2011)'

    def get_smoothness_values(self, elements, periodic: bool) -> np.ndarray:
        n_cell = len(elements)
        averages = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = elements[i_curr]
            assert isinstance(curr, concept.Element)
            averages[i_curr] = np.abs(curr.expansion().average())
        # Averaging is trivial for finite-volume and Legendre-based schems,
        # but might be expansive for Lagrange-based schemes.
        smoothness_values = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = elements[i_curr]
            assert isinstance(curr, concept.Element)
            curr_value = curr.get_solution_value(curr.x_center())
            left_value = curr_value
            right_value = curr_value
            max_average = averages[i_curr]
            n_neighbor = 0
            dividend = 0.0
            left, right = None, None
            if periodic or i_curr > 0:
                i_left = i_curr - 1
                left = elements[i_left]
                assert isinstance(left, concept.Element)
                n_neighbor += 1
                max_average = max(max_average, averages[i_left])
                curr_center = left.x_right() + curr.length() / 2
                left_value = left.get_solution_value(curr_center)
                dividend += np.abs(curr_value - left_value)
            if periodic or i_curr + 1 < n_cell:
                i_right = (i_curr + 1) % n_cell
                right = elements[i_right]
                assert isinstance(right, concept.Element)
                n_neighbor += 1
                max_average = max(max_average, averages[i_right])
                curr_center = right.x_left() - curr.length() / 2
                right_value = right.get_solution_value(curr_center)
                dividend += np.abs(curr_value - right_value)
            ratio = n_neighbor * curr.length()**((curr.degree() + 1) / 2)
            max_average += 1e-8  # avoid division-by-zero
            divisor = ratio * max_average
            smoothness_values[i_curr] = dividend / divisor
        return smoothness_values

    def max_smoothness(self, curr: concept.Element):
        return 1 + 2*(curr.degree() > 2)


class ZhuShuQiu2021(SmoothnessBased):
    """A jump detector for high-order DG schemes.

    See Zhu and Shu and Qiu, "High-order Runge-Kutta discontinuous Galerkin methods with multi-resolution WENO limiters for solving steady-state problems", Applied Numerical Mathematics 165 (2021), pp. 482--499.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Zhu–Shu–Qiu (2011)'
        else:
            return 'Zhu (2021)'

    def get_smoothness_values(self, elements, periodic: bool) -> np.ndarray:
        n_cell = len(elements)
        norms = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = elements[i_curr]
            assert isinstance(curr, concept.Element)
            def function(x_global):
                return curr.get_solution_value(x_global)
            norms[i_curr] = curr.integrator().norm_1(function, curr.n_term())
        smoothness_values = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = elements[i_curr]
            assert isinstance(curr, concept.Element)
            dividend = 0.0
            min_norm = norms[i_curr]
            if periodic or i_curr > 0:
                i_left = i_curr - 1
                integral = self._integrate(curr, elements[i_left], True)
                dividend = max(dividend, integral)
                min_norm = min(min_norm, norms[i_left])
            if periodic or i_curr + 1 < n_cell:
                i_right = (i_curr + 1) % n_cell
                integral = self._integrate(curr, elements[i_right], False)
                dividend = max(dividend, integral)
                min_norm = min(min_norm, norms[i_right])
            min_norm += 1e-8  # avoid division-by-zero
            divisor = min_norm * curr.length()
            smoothness_values[i_curr] = dividend / divisor
        return smoothness_values

    def _integrate(self, curr: concept.Element, that: concept.Element,
            left: bool):
        def integrand(x_curr):
            value = curr.get_solution_value(x_curr)
            if left:
                x_left = that.x_right() + (x_curr - curr.x_left())
                value -= that.get_solution_value(x_left)
                return value
            else:
                x_right = that.x_left() - (curr.x_right() - x_curr)
                value -= that.get_solution_value(x_right)
            return value
        integral = curr.integrator().fixed_quad_global(integrand,
            curr.n_term())
        return np.abs(integral)


class LiRen2022(concept.Detector):
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

    def get_troubled_cell_indices(self, elements, periodic: bool) -> np.ndarray:
        troubled_cell_indices = []
        n_cell = len(elements)
        n_node_per_cell = elements[0].n_term()
        n_node = n_cell * n_node_per_cell
        u_values = np.ndarray(n_node)
        i_node = 0
        for i_cell in range(n_cell):
            curr_element = elements[i_cell]
            assert isinstance(curr_element, concept.Element)
            if isinstance(curr_element.expansion(), expansion.Lagrange):
                for u_value in curr_element.get_solution_column():
                    u_values[i_node] = u_value
                    i_node += 1
            else:
                delta = curr_element.length() / curr_element.n_term() / 2
                points = np.linspace(curr_element.x_left() + delta,
                    curr_element.x_right() - delta, curr_element.n_term())
                for x in points:
                    u_values[i_node] = curr_element.get_solution_value(x)
                    i_node += 1
        assert i_node == n_node
        psi_values = np.ndarray(n_node)
        for i_node in range(n_node):
            a = np.abs(u_values[i_node] - u_values[i_node-1])
            a += np.abs(u_values[i_node] - 2 * u_values[i_node-1]
                + u_values[i_node-2])
            b = np.abs(u_values[i_node] - u_values[(i_node+1)%n_node])
            b += np.abs(u_values[i_node] - 2 * u_values[(i_node+1)%n_node]
                + u_values[(i_node+2)%n_node])
            psi_values[i_node] = (2 * a * b + self._epsilon) / (
                a**2 + b**2 + self._epsilon)
        for i_cell in range(n_cell):
            i_node_min = n_node_per_cell * i_cell
            i_node_max = n_node_per_cell + i_node_min
            for i_node in range(i_node_min, i_node_max):
                left = min(psi_values[i_node], psi_values[i_node-1])
                right = min(psi_values[i_node], psi_values[(i_node+1)%n_cell])
                if min(left, right) < self._psi_c:
                    troubled_cell_indices.append(i_cell)
                    break
        return troubled_cell_indices

    def get_smoothness_values(self, elements, periodic: bool) -> np.ndarray:
        """Not necessary for detecting, just for comparison.
        """
        troubled_cell_indices = self.get_troubled_cell_indices(elements, periodic)
        smoothness_values = np.zeros(len(elements)) + 1e-2
        for i_cell in troubled_cell_indices:
            smoothness_values[i_cell] = 1e2
        return smoothness_values


class Persson2006(SmoothnessBased):
    """A jump detector based on artificial viscosity.

    See Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006).
    """

    def name(self, verbose=False):
        return 'Persson (2006)'

    @staticmethod
    def get_smoothness_value(u_approx: expansion.Taylor):
        if isinstance(u_approx, expansion.Legendre):
            pth_mode_energy = u_approx.get_mode_energy(-1)
            all_modes_energy = 0.0
            for k in range(u_approx.n_term()):
                all_modes_energy += u_approx.get_mode_energy(k)
        else:
            assert isinstance(u_approx, expansion.Taylor)
            def u(x):
                return u_approx.global_to_value(x)
            all_modes_energy = u_approx.integrator().inner_product(
                u, u, u_approx.n_term())
            legendre = expansion.Legendre(u_approx.degree(),
                u_approx.coordinate())
            pth_basis = legendre.get_basis(u_approx.degree())
            pth_mode_energy = u_approx.integrator().inner_product(
                u, pth_basis, u_approx.n_term())**2
            pth_mode_energy /= legendre.get_mode_weight(u_approx.degree())
        return pth_mode_energy / all_modes_energy

    def get_smoothness_values(self, elements, periodic: bool) -> np.ndarray:
        n_cell = len(elements)
        smoothness_values = np.ndarray(n_cell)
        for i_cell in range(n_cell):
            cell = elements[i_cell]
            assert isinstance(cell, concept.Element)
            sensor_ref = cell.degree()**(-3)
            u_approx = cell.expansion()
            sensor = Persson2006.get_smoothness_value(u_approx)
            smoothness_values[i_cell] = sensor / sensor_ref
        return smoothness_values


if __name__ == '__main__':
    pass
