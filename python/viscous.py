import numpy as np

import concept
import expansion
import element
import detector
from coordinate import LinearCoordinate
from integrator import GaussLegendre

class Constant(concept.Viscous):

    def __init__(self, const=0.0) -> None:
        super().__init__()
        self._const = const

    def name(self, verbose=False) -> str:
        if verbose:
            return 'Constant(' + r'$\nu=$' + f'{self._const})'
        else:
            return 'Constant'

    def generate(self, troubled_cell_indices, grid: concept.Grid):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._const
            # print(f'nu[{i_cell}] = {coeff}')
            self._index_to_coeff[i_cell] = coeff


class Persson2006(concept.Viscous):
    """Artificial viscosity for DG and FR schemes.

    See Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006).
    """

    def __init__(self, kappa=2.0) -> None:
        super().__init__()
        self._kappa = kappa

    def name(self, verbose=False) -> str:
        if verbose:
            return 'Persson(2006, ' + r'$\kappa=$' + f'{self._kappa})'
        else:
            return 'Persson'

    def _get_constant_coeff(self, cell: concept.Element):
        u_approx = cell.expansion()
        s_0 = -4 * np.log10(u_approx.degree())
        smoothness = detector.Persson2006.get_smoothness_value(u_approx)
        s_gap = np.log10(smoothness) - s_0
        # print(smoothness, s_gap)
        nu = u_approx.length() / u_approx.degree()
        if s_gap > self._kappa:
            pass
        elif s_gap > -self._kappa:
            nu *= 0.5 * (1 + np.sin(s_gap / self._kappa * np.pi / 2))
        else:
            nu = 0.0
        return nu

    def generate(self, troubled_cell_indices, grid: concept.Grid):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._get_constant_coeff(grid.get_element_by_index(i_cell))
            self._index_to_coeff[i_cell] = coeff


class Energy(concept.Viscous):

    def __init__(self, tau=0.01) -> None:
        super().__init__()
        self._tau = tau
        self._index_to_nu = dict()
        self._index_to_a_inv = dict()

    def name(self, verbose=False) -> str:
        if verbose:
            return 'Energy(' + r'$\tau=$' + f'{self._tau})'
        else:
            return 'Energy'

    @staticmethod
    def _jumps_to_energy(jumps: np.ndarray, curr: expansion.GaussLagrange,
            indices=None):
        energy = 0.0
        if not indices:
            indices = range(len(jumps))
        for k in indices:
            energy += curr.get_sample_weight(k) * jumps[k]**2 / 2
        return energy

    def _get_high_order_energy(self, cell: element.GaussLagrangeFR,
            points: np.ndarray, values: np.ndarray):
        """Compare with polynomials on neighbors.
        """
        curr = cell.expansion()
        left, right = cell.neighbor_expansions()
        left_jumps = np.ndarray(len(points))
        right_jumps = np.ndarray(len(points))
        for i in range(len(points)):
            x_curr = points[i]
            if left:
                left_jumps[i] = values[i] - left.global_to_value(x_curr)
            else:
                left_jumps[i] = np.infty
            if right:
                right_jumps[i] = values[i] - right.global_to_value(x_curr)
            else:
                right_jumps[i] = np.infty
        return min(Energy._jumps_to_energy(left_jumps, curr),
                   Energy._jumps_to_energy(right_jumps, curr))

    def _get_low_order_energy(self, cell: element.GaussLagrangeFR,
            points: np.ndarray, values: np.ndarray):
        """Compare with p=1 polynomials borrowed from neighbors.
        """
        curr = cell.expansion()
        left, right = cell.neighbor_expansions()
        # build left_jumps
        left_jumps = np.ndarray(len(points))
        if left:
            left_low = expansion.Legendre(1, left.coordinate())
            left_low.approximate(lambda x: left.global_to_value(x))
            for i in range(len(points)):
                left_jumps[i] = values[i] - left_low.global_to_value(points[i])
        else:
            for i in range(len(points)):
                left_jumps[i] = np.infty
        # build right_jumps
        right_jumps = np.ndarray(len(points))
        if right:
            right_low = expansion.Legendre(1, right.coordinate())
            right_low.approximate(lambda x: right.global_to_value(x))
            for i in range(len(points)):
                right_jumps[i] = values[i] - right_low.global_to_value(points[i])
        else:
            for i in range(len(points)):
                right_jumps[i] = np.infty
        return min(Energy._jumps_to_energy(left_jumps, curr),
                   Energy._jumps_to_energy(right_jumps, curr))

    def _get_half_by_half_energy(self, cell: element.GaussLagrangeFR,
            points: np.ndarray, values: np.ndarray):
        """Compare with p=k and p=1 extensions from neighbors in the closer half using the integrator on cell.
        """
        curr = cell.expansion()
        left, right = cell.neighbor_expansions()
        low_jumps = np.zeros(len(points))
        high_jumps = np.zeros(len(points))
        # build left_energy
        left_energy = np.infty
        if left:
            left_low = expansion.Legendre(1, left.coordinate())
            left_low.approximate(lambda x: left.global_to_value(x))
            indices = range((1 + len(points)) // 2)
            for i in indices:
                low_jumps[i] = values[i] - left_low.global_to_value(points[i])
                high_jumps[i] = values[i] - left.global_to_value(points[i])
            if len(points) % 2:
                low_jumps[indices[-1]] /= np.sqrt(2)
                high_jumps[indices[-1]] /= np.sqrt(2)
            left_energy = min(
                Energy._jumps_to_energy(low_jumps, curr, indices),
                Energy._jumps_to_energy(high_jumps, curr, indices))
        # build right_energy
        right_energy = np.infty
        if right:
            right_low = expansion.Legendre(1, right.coordinate())
            right_low.approximate(lambda x: right.global_to_value(x))
            indices = range(len(points) // 2, len(points))
            for i in indices:
                low_jumps[i] = values[i] - right_low.global_to_value(points[i])
                high_jumps[i] = values[i] - right.global_to_value(points[i])
            if len(points) % 2:
                low_jumps[indices[0]] /= np.sqrt(2)
                high_jumps[indices[0]] /= np.sqrt(2)
            right_energy = min(
                Energy._jumps_to_energy(low_jumps, curr, indices),
                Energy._jumps_to_energy(high_jumps, curr, indices))
        return left_energy + right_energy

    def _get_half_exact_energy(self, cell: element.GaussLagrangeFR):
        """Same as _get_half_by_half_energy, but using integrators on subcells.
        """
        def get_energy(coord: concept.Coordinate,
                this: concept.Expansion, that: concept.Expansion):
            degree = max(this.degree(), that.degree()) * 2
            degree += coord.jacobian_degree()
            integrator = GaussLegendre(coord)
            def diff_sq(x):
                value = this.global_to_value(x) - that.global_to_value(x)
                return value**2
            return integrator.fixed_quad_global(diff_sq, degree // 2 + 1) / 2
        curr = cell.expansion()
        left, right = cell.neighbor_expansions()
        # build left_energy
        left_energy = 0
        if left:
            left_coord = LinearCoordinate(curr.x_left(), curr.x_center())
            left_low = expansion.Legendre(1, left_coord)
            left_low.approximate(lambda x: left.global_to_value(x))
            left_energy = min(get_energy(left_coord, curr, left),
                              get_energy(left_coord, curr, left_low))
        # build right_energy
        right_energy = 0
        if right:
            right_coord = LinearCoordinate(curr.x_center(), curr.x_right())
            right_low = expansion.Legendre(1, right_coord)
            right_low.approximate(lambda x: right.global_to_value(x))
            right_energy = min(get_energy(right_coord, curr, right),
                               get_energy(right_coord, curr, right_low))
        return left_energy + right_energy

    def _get_interface_jump_energy(self, cell: element.GaussLagrangeFR):
        """Compute derivative jumps on interfaces.
        """
        curr = cell.expansion()
        left, right = cell.neighbor_expansions()
        weights = (1,)
        def jumps_to_energy(jumps: np.ndarray, distance):
            energy = 0.0
            for k in range(min(len(weights), len(jumps))):
                energy += (jumps[k] * distance**k * weights[k])**2
            return energy
        # build left_energy
        if left:
            jumps = left.global_to_derivatives(left.x_right())
            jumps -= curr.global_to_derivatives(curr.x_left())
            distance = (left.length() + curr.length()) / 2
            left_energy = jumps_to_energy(jumps, distance)
        # build right_energy
        if right:
            jumps = right.global_to_derivatives(right.x_left())
            jumps -= curr.global_to_derivatives(curr.x_right())
            distance = (right.length() + curr.length()) / 2
            right_energy = jumps_to_energy(jumps, distance)
        # build energy
        if left and right:
            return min(left_energy, right_energy)
        elif left:
            return left_energy
        else:
            return right_energy

    def _get_oscillation_energy(self, cell: element.GaussLagrangeFR):
        """Compare with four polynomials borrowed from neighbors.
        """
        points = cell.get_sample_points()
        values = np.ndarray(len(points))
        for i in range(len(points)):
            values[i] = cell.get_solution_value(points[i])
        # return self._get_high_order_energy(cell, points, values)
        # return self._get_low_order_energy(cell, points, values)
        # return self._get_interface_jump_energy(cell, points, values)
        # return min(self._get_low_order_energy(cell, points, values), self._get_high_order_energy(cell, points, values))
        # return min(self._get_low_order_energy(cell, points, values), self._get_high_order_energy(cell, points, values), self._get_interface_jump_energy(cell))
        # return self._get_half_by_half_energy(cell, points, values)
        # return self._get_half_exact_energy(cell)
        return min(self._get_low_order_energy(cell, points, values), self._get_high_order_energy(cell, points, values), self._get_half_by_half_energy(cell, points, values))

    @staticmethod
    def _nu_max(cell: element.LagrangeFR):
        a_max = 0
        u_samples = cell.expansion().get_sample_values()
        for u in u_samples:
            eigvals = cell.equation().get_convective_eigvals(u)
            for val in eigvals:
                a_max = max(a_max, np.abs(val))
        return cell.length() / cell.degree() * a_max

    def _get_constant_coeff(self, grid: concept.Grid, i_curr: int):
        curr = grid.get_element_by_index(i_curr)
        assert isinstance(curr, element.GaussLagrangeFR)
        dissipation = curr.get_dissipation_rate()
        oscillation_energy = self._get_oscillation_energy(curr)
        nu = oscillation_energy / (-dissipation * self._tau)
        assert nu >= 0, (nu)
        return min(nu, Energy._nu_max(curr))

    def _get_nu(self, i_cell: int, n_cell: int):
        assert -1 <= i_cell <= n_cell
        if i_cell < 0:
            i_cell += n_cell
        if i_cell == n_cell:
            i_cell = 0
        if i_cell in self._index_to_nu:
            return self._index_to_nu[i_cell]
        else:
            return 0.0

    def _build_a_on_centers(self, cell: element.GaussLagrangeFR):
        a = np.eye(3)
        a[1][0] = a[2][0] = 1.0
        left, right = cell.neighbor_expansions()
        if left:
            h = cell.x_center() - left.x_center()
            h = cell.length()
        a[1][1] = -h
        a[1][2] = h*h
        if right:
            h = right.x_center() - cell.x_center()
        else:
            h = cell.length()
        a[2][1] = h
        a[2][2] = h*h
        return a

    def _build_a_on_interfaces(self, cell: element.GaussLagrangeFR):
        a = np.eye(3)
        a[1][0] = a[2][0] = 1.0
        h = cell.length() / 2
        a[1][1] = -h
        a[2][1] = h
        a[1][2] = a[2][2] = h*h
        return a

    def _build_a_on_interfaces_with_average(self, cell: element.GaussLagrangeFR):
        a = np.eye(3)
        a[1][0] = a[2][0] = 1.0
        a[0][0] = cell.length()
        a[0][2] = (cell.length() / 2)**3 * 2 / 3
        h = cell.length() / 2
        a[1][1] = -h
        a[2][1] = h
        a[1][2] = a[2][2] = h*h
        return a

    def _build_a_with_averages(self, cell: element.GaussLagrangeFR):
        left, right = cell.neighbor_expansions()
        a = np.ndarray((3,3))
        a[0][0] = cell.length()
        a[0][1] = 0
        a[0][2] = (cell.length() / 2)**3 * 2 / 3
        if left:
            x_left = left.x_left() - cell.x_center()
            x_right = left.x_right() - cell.x_center()
        else:
            x_left = cell.x_left() - cell.length()
            x_right = cell.x_right() - cell.length()
        for p in (0,1,2):
            q = p + 1
            a[1][p] = (x_right**q - x_left**q) / q
        if right:
            x_left = right.x_left() - cell.x_center()
            x_right = right.x_right() - cell.x_center()
        else:
            x_left = cell.x_left() + cell.length()
            x_right = cell.x_right() + cell.length()
        for p in (0,1,2):
            q = p + 1
            a[2][p] = (x_right**q - x_left**q) / q
        return a

    @staticmethod
    def _common(a, b):
        # return (a + b) / 2
        return min(a, b)

    def _get_smooth_coeff(self, grid: concept.Grid, i_cell: int) -> callable:
        cell = grid.get_element_by_index(i_cell)
        n_cell = grid.n_element()
        b = np.ndarray(3)
        assert i_cell in self._index_to_nu
        b[0] = self._index_to_nu[i_cell]
        left, right = cell.neighbor_expansions()
        if left:
            b[1] = Energy._common(b[0], self._get_nu(i_cell - 1, n_cell))
        else:
            b[1] = b[0]
        if right:
            b[2] = Energy._common(b[0], self._get_nu(i_cell + 1, n_cell))
        else:
            b[2] = b[0]
        if b[0] != max(b):  # build linear for non-max
            return lambda x: b[1] + (b[2] - b[1]) * (x - cell.x_left()) / cell.length()
        else:  # build quad for max
            pass
        if i_cell not in self._index_to_a_inv:
            a_inv = np.linalg.inv(self._build_a_on_interfaces(cell))
            self._index_to_a_inv[i_cell] = a_inv
        else:
            a_inv = self._index_to_a_inv[i_cell]
        c = a_inv @ b
        def coeff(x_global: float):
            x = x_global - cell.x_center()
            nu = max(0, c[0] + c[1] * x + c[2] * x * x)
            return min(nu, Energy._nu_max(cell))
        return coeff

    def generate(self, troubled_cell_indices, grid: concept.Grid):
        self._index_to_nu.clear()
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            nu = self._get_constant_coeff(grid, i_cell)
            self._index_to_nu[i_cell] = nu
        for i_cell in troubled_cell_indices:
            coeff = self._get_smooth_coeff(grid, i_cell)
            self._index_to_coeff[i_cell] = coeff


if __name__ == '__main__':
    pass
