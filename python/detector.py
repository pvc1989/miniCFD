"""Concrete implementations of jump detectors.
"""
import numpy as np
import abc

import concept
import expansion
import equation
from scipy import stats


class All(concept.Detector):
    """A jump detector that reports all cells as troubled.
    """

    def name(self, verbose=False):
        return 'All'

    def get_troubled_cell_indices(self, grid: concept.Grid):
        return range(grid.n_element())


class SmoothnessBased(concept.Detector):

    @abc.abstractmethod
    def get_smoothness_values(self, grid: concept.Grid) -> np.ndarray:
        """Get the values of smoothness for each element.
        """

    def max_smoothness(self, curr: concept.Element) -> float:
        """Get the max value of the smoothness of a smooth cell.
        """
        return 1.0

    def get_scalar_smoothness(self, cell: concept.Element, dividend, divisor):
        if cell.is_scalar():
            return dividend / divisor
        else:
            return max(dividend / divisor)

    def get_troubled_cell_indices(self, grid: concept.Grid) -> list:
        troubled_cell_indices = []
        smoothness_values = self.get_smoothness_values(grid)
        for i in range(grid.n_element()):
            cell_i = grid.get_element_by_index(i)
            if smoothness_values[i] > self.max_smoothness(cell_i):
                troubled_cell_indices.append(i)
        return troubled_cell_indices


class Krivodonova2004(SmoothnessBased):
    """A jump detector for high-order DG schemes.

    See [Krivodonova et al., "Shock detection and limiting with discontinuous Galerkin methods for hyperbolic conservation laws", Applied Numerical Mathematics 48, 3-4 (2004), pp. 323--338](https://doi.org/10.1016/j.apnum.2003.11.002) for details.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Krivodonova et al. (2004)'
        else:
            return 'KXRCF'

    @staticmethod
    def _get_normal_velocity(cell: concept.Element, value) -> float:
        eq = cell.equation()
        if isinstance(eq, concept.ScalarEquation):
            return eq.get_convective_speed(value)
        elif isinstance(eq, equation.Euler):
            # _, u, _ = eq.conservative_to_primitive(value)
            return value[1] / value[0]
        else:
            assert False

    def get_smoothness_values(self, grid: concept.Grid) -> np.ndarray:
        n_cell = grid.n_element()
        smoothness_values = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            cell_i = grid.get_element_by_index(i_curr)
            curr = cell_i.expansion()
            # def function(x_global):
            #     return curr.global_to_value(x_global)
            # curr_norm = curr.integrator().norm_infty(function, curr.n_term())
            # averaging is faster if cached
            curr_norm = np.abs(curr.average())
            curr_norm += 1e-8  # avoid division-by-zero
            ratio = curr.length()**((curr.degree() + 1) / 2)
            left, right = cell_i.neighbor_expansions()
            if np.isscalar(curr_norm):
                dividend = 0.0
            else:
                dividend = np.zeros(len(curr_norm))
            if left:
                value = curr.get_boundary_derivatives(0, True, False)
                u_n = Krivodonova2004._get_normal_velocity(cell_i, value)
                if u_n >= 0:
                    dividend += np.abs(value
                        - left.get_boundary_derivatives(0, False, False))
            if right:
                value = curr.get_boundary_derivatives(0, False, False)
                u_n = Krivodonova2004._get_normal_velocity(cell_i, value)
                if u_n <= 0:
                    dividend += np.abs(value
                        - right.get_boundary_derivatives(0, True, False))
            divisor = ratio * curr_norm
            smoothness_values[i_curr] = \
                self.get_scalar_smoothness(cell_i, dividend, divisor)
        return smoothness_values


class LiWanAi2011(SmoothnessBased):
    """A jump detector for high-order finite volume schemes.

    See [Li and Ren, "High-order k-exact WENO finite volume schemes for solving gas dynamic Euler equations on unstructured grids", International Journal for Numerical Methods in Fluids 70, 6 (2011), pp. 742--763](https://doi.org/10.1002/fld.2710) for details.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Li Wan-Ai (2011)'
        else:
            return 'LWA'

    def get_smoothness_values(self, grid: concept.Grid) -> np.ndarray:
        n_cell = grid.n_element()
        averages = np.ndarray(n_cell, grid.get_element_by_index(0).value_type())
        for i_curr in range(n_cell):
            curr = grid.get_element_by_index(i_curr)
            averages[i_curr] = np.abs(curr.expansion().average())
        # Averaging is trivial for finite-volume and Legendre-based schems,
        # but might be expansive for Lagrange-based schemes.
        smoothness_values = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = grid.get_element_by_index(i_curr)
            curr_value = curr.get_solution_value(curr.x_center())
            curr_norm = averages[i_curr]
            n_neighbor = 0
            dividend = 0.0
            left, right = curr.neighbor_expansions()
            if left:
                i_left = i_curr - 1
                n_neighbor += 1
                curr_norm = np.minimum(curr_norm, averages[i_left])
                left_value = left.global_to_value(curr.x_center())
                dividend += np.abs(curr_value - left_value)
            if right:
                i_right = (i_curr + 1) % n_cell
                n_neighbor += 1
                curr_norm = np.minimum(curr_norm, averages[i_right])
                right_value = right.global_to_value(curr.x_center())
                dividend += np.abs(curr_value - right_value)
            ratio = n_neighbor * curr.length()**((curr.degree() + 1) / 2)
            curr_norm += 1e-8  # avoid division-by-zero
            divisor = ratio * curr_norm
            smoothness_values[i_curr] = \
                self.get_scalar_smoothness(curr, dividend, divisor)
        return smoothness_values

    def max_smoothness(self, curr: concept.Element):
        return 1 + 2*(curr.degree() > 2)


class ZhuJun2021(SmoothnessBased):
    """A jump detector for high-order DG schemes.

    See [Zhu and Shu and Qiu, "High-order Runge-Kutta discontinuous Galerkin methods with multi-resolution WENO limiters for solving steady-state problems", Applied Numerical Mathematics 165 (2021), pp. 482--499](https://doi.org/10.1016/j.apnum.2021.03.011) for details.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Zhu Jun (2021)'
        else:
            return 'ZJ'

    def get_smoothness_values(self, grid: concept.Grid) -> np.ndarray:
        n_cell = grid.n_element()
        value_type = grid.get_element_by_index(0).value_type()
        norms = np.ndarray(n_cell, value_type)
        for i_curr in range(n_cell):
            curr = grid.get_element_by_index(i_curr)
            def function(x_global):
                return curr.get_solution_value(x_global)
            norms[i_curr] = curr.integrator().norm_1(function)
        smoothness_values = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = grid.get_element_by_index(i_curr)
            left, right = curr.neighbor_expansions()
            dividend = 0.0
            min_norm = norms[i_curr]
            if left:
                i_left = i_curr - 1
                integral = self._integrate(curr, left)
                dividend = np.maximum(dividend, integral)
                min_norm = np.minimum(min_norm, norms[i_left])
            if right:
                i_right = (i_curr + 1) % n_cell
                integral = self._integrate(curr, right)
                dividend = np.maximum(dividend, integral)
                min_norm = np.minimum(min_norm, norms[i_right])
            min_norm += 1e-8  # avoid division-by-zero
            divisor = min_norm * curr.length()
            smoothness_values[i_curr] = \
                self.get_scalar_smoothness(curr, dividend, divisor)
        return smoothness_values

    def _integrate(self, curr: concept.Element, that: concept.Expansion):
        integral = curr.integrator().integrate(
            lambda x: curr.get_solution_value(x) - that.global_to_value(x))
        return np.abs(integral)


class LiYanHui2022(concept.Detector):
    """A jump detector for high-order FD schemes.

    See [Li Yanhui, Chen Congwei, and Ren Yu-Xin, "A class of high-order finite difference schemes with minimized dispersion and adaptive dissipation for solving compressible flo…", Journal of Computational Physics 448 (2022), pp. 110770](https://doi.org/10.1016/j.jcp.2021.110770) for details.
    """

    def __init__(self) -> None:
        self._xi = 0.01
        self._psi_c = 0.4
        self._epsilon = (0.9 * self._psi_c) / (1 - 0.9 * self._psi_c)
        self._epsilon *= self._xi**2
        self._sample_points = []
        self._sample_values = None
        self._grid = None

    def name(self, verbose=False):
        if verbose:
            return 'Li Yan-Hui (2022)'
        else:
            return 'LYH'

    def get_troubled_cell_indices(self, grid: concept.Grid) -> np.ndarray:
        # TODO: support nonuniform grid
        troubled_cell_indices = []
        n_cell = grid.n_element()
        value_type = grid.get_element_by_index(0).value_type()
        n_node_per_cell = grid.get_element_by_index(0).n_term()
        n_node_per_cell = max(n_node_per_cell, 3)
        n_node = n_cell * n_node_per_cell
        if grid is not self._grid:
            self._grid = grid
            self._sample_points = []
            self._sample_values = np.ndarray(n_node, value_type)
        u_values = self._sample_values
        i_node = 0
        for i_cell in range(n_cell):
            curr_element = grid.get_element_by_index(i_cell)
            assert isinstance(curr_element, concept.Element)
            if len(self._sample_points) == i_cell:
                delta = curr_element.length() / n_node_per_cell / 2
                points = np.linspace(curr_element.x_left() + delta,
                    curr_element.x_right() - delta, n_node_per_cell)
                self._sample_points.append(points)
            for x in self._sample_points[i_cell]:
                u_values[i_node] = curr_element.get_solution_value(x)
                i_node += 1
        assert i_node == n_node, (i_node, n_node)
        psi_values = np.ndarray(n_node, value_type)
        for i_node in range(n_node):
            a = np.abs(u_values[i_node] - u_values[i_node-1])
            a += np.abs(u_values[i_node] - 2 * u_values[i_node-1]
                + u_values[i_node-2])
            b = np.abs(u_values[i_node] - u_values[(i_node+1)%n_node])
            b += np.abs(u_values[i_node] - 2 * u_values[(i_node+1)%n_node]
                + u_values[(i_node+2)%n_node])
            psi_values[i_node] = (2 * a * b + self._epsilon) / (
                a**2 + b**2 + self._epsilon)
        if not grid.is_periodic():
            psi_values[0] = psi_values[1] = self._psi_c
            psi_values[-1] = psi_values[-2] = self._psi_c
        for i_cell in range(n_cell):
            i_node_min = n_node_per_cell * i_cell
            i_node_max = n_node_per_cell + i_node_min
            for i_node in range(i_node_min, i_node_max):
                i_left = i_node - 1
                i_right = (i_node + 1) % n_node
                psi_min = np.minimum(psi_values[i_node], psi_values[i_left])
                psi_min = np.minimum(psi_min, psi_values[i_right])
                if (psi_min < self._psi_c).any():
                    troubled_cell_indices.append(i_cell)
                    break
        return troubled_cell_indices

    def get_smoothness_values(self, grid: concept.Grid) -> np.ndarray:
        """Not necessary for detecting, just for comparison.
        """
        troubled_cell_indices = self.get_troubled_cell_indices(grid)
        smoothness_values = np.zeros(grid.n_element()) + 1e-2
        for i_cell in troubled_cell_indices:
            smoothness_values[i_cell] = 1e2
        return smoothness_values


class Persson2006(SmoothnessBased):
    """A jump detector based on decay of modal coefficients.

    See [Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006)](https://doi.org/10.2514/6.2006-112) for details.
    """

    def name(self, verbose=False):
        if verbose:
            return 'Persson (2006)'
        else:
            return 'Persson'

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
                u_approx.coordinate(), u_approx.value_type())
            pth_basis = legendre.get_basis(u_approx.degree())
            pth_mode_energy = u_approx.integrator().inner_product(
                u, pth_basis, u_approx.n_term())**2
            pth_mode_energy /= legendre.get_mode_weight(u_approx.degree())
        all_modes_energy += 1e-8
        return pth_mode_energy / all_modes_energy

    def get_smoothness_values(self, grid: concept.Grid) -> np.ndarray:
        n_cell = grid.n_element()
        smoothness_values = np.ndarray(n_cell)
        for i_cell in range(n_cell):
            cell = grid.get_element_by_index(i_cell)
            sensor_ref = cell.degree()**(-3)
            u_approx = cell.expansion()
            sensor = Persson2006.get_smoothness_value(u_approx)
            smoothness_values[i_cell] = \
                self.get_scalar_smoothness(cell, sensor, sensor_ref)
        return smoothness_values


class Kloeckner2011(SmoothnessBased):
    """A jump detector based on decay of modal coefficients.

    See [Klöckner and Warburton and Hesthaven, "Viscous Shock Capturing in a Time-Explicit Discontinuous Galerkin Method", Mathematical Modelling of Natural Phenomena 6, 3 (2011), pp. 57--83](https://doi.org/10.1051/mmnp/20116303) for details.
    """

    def __init__(self, degree: int) -> None:
        self._degree = degree
        decay = np.ndarray(degree)
        sum = 0
        for n in range(degree):
            decay[n] = (n + 1)**(-2 * degree)
            sum += decay[n]
        decay /= sum
        self._modal_decay = decay

    def name(self, verbose=False):
        if verbose:
            return 'Klöckner (2011)'
        else:
            return 'Klöckner'

    def _b2(self, degree: int):
        assert 1 <= degree <= self._degree
        return self._modal_decay[degree - 1]

    def add_modal_decay(self, energy_array: np.ndarray) -> np.ndarray:
        assert self._degree + 1 == len(energy_array)
        energy_sum = np.sum(energy_array) + 1e-8
        for n in range(self._degree, 0, -1):
            energy_array[n] += energy_sum * self._b2(n)
        return energy_array

    def apply_skyline(self, energy_array: np.ndarray) -> np.ndarray:
        assert self._degree + 1 == len(energy_array)
        energy_array[-1] = energy_array[-2] = \
            np.maximum(np.abs(energy_array[-1]), np.abs(energy_array[-2]))
        for n in range(len(energy_array) - 3, 0, -1):
            energy_array[n] = np.maximum(energy_array[n], energy_array[n + 1])
        return energy_array

    def get_least_square_slope(self, energy_array: np.ndarray):
        assert self._degree + 1 == len(energy_array)
        p = len(energy_array) - 1
        x_sum = 0
        y_sum = np.log10(energy_array[1])
        xx_sum = 0
        xy_sum = 0
        for k in range(1, p):
            x_k = -2 * np.log10(k + 1)
            y_k = np.log10(energy_array[k + 1])
            x_sum += x_k
            y_sum += y_k
            xx_sum += x_k * x_k
            xy_sum += x_k * y_k
        return (xy_sum - x_sum * y_sum / p) / (xx_sum - x_sum * x_sum / p)

    def get_smoothness_value(self, u_approx: expansion.Taylor):
        if isinstance(u_approx, expansion.Legendre):
            legendre = u_approx
        else:
            assert isinstance(u_approx, expansion.Taylor)
            legendre = expansion.Legendre(u_approx.degree(),
                u_approx.coordinate(), u_approx.value_type())
        energy_array = np.ndarray(legendre.n_term(), legendre.value_type())
        for k in range(legendre.n_term()):
            energy_array[k] += legendre.get_mode_energy(k)
        self.add_modal_decay(energy_array)
        self.apply_skyline(energy_array)
        return self.get_least_square_slope(energy_array)

    def get_smoothness_values(self, grid: concept.Grid) -> np.ndarray:
        n_cell = grid.n_element()
        smoothness_values = np.ndarray(n_cell)
        for i_cell in range(n_cell):
            cell = grid.get_element_by_index(i_cell)
            values = self.get_smoothness_value(cell.expansion())
            smoothness_values[i_cell] = \
                self.get_scalar_smoothness(cell, 1.5, values)
        return smoothness_values


if __name__ == '__main__':
    pass
