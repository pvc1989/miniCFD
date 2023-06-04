import numpy as np

import concept
import expansion
import element
import detector


class Constant(concept.Viscous):

    def __init__(self, const=0.0) -> None:
        super().__init__()
        self._const = const

    def name(self, verbose=False) -> str:
        if verbose:
            return 'Constant (' + r'$\nu=$' + f'{self._const})'
        else:
            return 'Constant'

    def generate(self, troubled_cell_indices, grid: concept.Grid, periodic: bool):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._const
            # print(f'nu[{i_cell}] = {coeff}')
            self._index_to_coeff[i_cell] = coeff


class Persson2006(concept.Viscous):
    """Artificial viscosity for DG and FR schemes.

    See Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006).
    """

    def __init__(self, kappa=0.1) -> None:
        super().__init__()
        self._kappa = kappa

    def name(self, verbose=False) -> str:
        if verbose:
            return "Persson (2006)"
        else:
            return 'Persson'

    def _get_viscous_coeff(self, cell: concept.Element):
        u_approx = cell.expansion()
        s_0 = -4 * np.log10(u_approx.degree())
        smoothness = detector.Persson2006.get_smoothness_value(u_approx)
        s_gap = np.log10(smoothness) - s_0
        print(s_gap)
        nu = u_approx.length() / u_approx.degree()
        if s_gap > self._kappa:
            pass
        elif s_gap > -self._kappa:
            nu *= 0.5 * (1 + np.sin(s_gap / self._kappa * np.pi / 2))
        else:
            nu = 0.0
        return nu

    def generate(self, troubled_cell_indices, grid: concept.Grid, periodic: bool):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._get_viscous_coeff(grid.get_element_by_index(i_cell))
            print(f'nu[{i_cell}] = {coeff}')
            self._index_to_coeff[i_cell] = coeff


class Energy(concept.Viscous):

    def __init__(self, tau=0.01, nu_max=0.1) -> None:
        super().__init__()
        self._tau = tau
        self._nu_max = nu_max
        self._index_to_matrices = dict()

    def name(self, verbose=False) -> str:
        if verbose:
            return 'Energy (' + r'$\tau=$' + f'{self._tau}' \
                + r', $\nu_\max=$' + f'{self._nu_max}' + ')'
        else:
            return 'Energy'

    def _get_oscillation_energy(self, curr: expansion.Lagrange,
            left: expansion.Lagrange, right: expansion.Lagrange):
        """Compare with polynomials on neighbors.
        """
        points = curr.get_sample_points()
        left_jumps = np.ndarray(len(points))
        right_jumps = np.ndarray(len(points))
        left_shift = left.x_right() - curr.x_left()
        right_shift = right.x_left() - curr.x_right()
        for i in range(len(points)):
            x_curr = points[i]
            curr_value = curr.global_to_value(x_curr)
            left_jumps[i] = curr_value - left.global_to_value(
                x_curr + left_shift)
            right_jumps[i] = curr_value - right.global_to_value(
                x_curr + right_shift)
        def jumps_to_energy(jumps: np.ndarray):
            energy = 0.0
            for k in range(curr.n_term()):
                energy += curr.get_node_weight(k) * jumps[k]**2 / 2
            return energy
        return min(jumps_to_energy(left_jumps), jumps_to_energy(right_jumps))

    def _get_oscillation_energy_p1(self, curr: expansion.Lagrange,
            left: expansion.Lagrange, right: expansion.Lagrange):
        """Compare with p=1 polynomials borrowed from neighbors.
        """
        curr_p1 = expansion.Legendre(1, curr.coordinate())
        points = curr.get_sample_points()
        curr_values = np.ndarray(len(points))
        for i in range(len(points)):
            curr_values[i] = curr.global_to_value(points[i])
        # build left_jumps
        left_jumps = np.ndarray(len(points))
        left_shift = left.x_right() - curr.x_left()
        curr_p1.approximate(lambda x_curr:
            left.global_to_value(x_curr + left_shift))
        for i in range(len(points)):
            left_jumps[i] = curr_values[i] - curr_p1.global_to_value(points[i])
        # build right_jumps
        right_jumps = np.ndarray(len(points))
        right_shift = right.x_left() - curr.x_right()
        curr_p1.approximate(lambda x_curr:
            right.global_to_value(x_curr + right_shift))
        for i in range(len(points)):
            right_jumps[i] = curr_values[i] - curr_p1.global_to_value(points[i])
        # build energy
        def jumps_to_energy(jumps: np.ndarray):
            energy = 0.0
            for k in range(curr.n_term()):
                energy += curr.get_node_weight(k) * jumps[k]**2 / 2
            return energy
        return min(jumps_to_energy(left_jumps), jumps_to_energy(right_jumps))

    def _get_oscillation_energy_jumps(self, curr: expansion.Lagrange,
            left: expansion.Lagrange, right: expansion.Lagrange):
        """Compute derivative jumps on interfaces.
        """
        # build left_jumps
        left_jumps = left.get_derivative_values(left.x_right())
        left_jumps -= curr.get_derivative_values(curr.x_left())
        # build right_jumps
        right_jumps = right.get_derivative_values(right.x_left())
        right_jumps -= curr.get_derivative_values(curr.x_right())
        # build energy
        weights = (1, 0.2)
        def jumps_to_energy(jumps: np.ndarray, distance):
            energy = 0.0
            for k in range(min(len(weights), len(jumps))):
                energy += (jumps[k] * distance**k * weights[k])**2
            return energy
        left_distance = (left.length() + curr.length()) / 2
        right_distance = (right.length() + curr.length()) / 2
        return min(jumps_to_energy(left_jumps, left_distance),
            jumps_to_energy(right_jumps, right_distance))

    def _get_viscous_coeff(self, grid: concept.Grid, i_curr):
        i_left = i_curr - 1
        i_right = (i_curr + 1) % grid.n_element()
        curr = grid.get_element_by_index(i_curr)
        left = grid.get_element_by_index(i_left)
        right = grid.get_element_by_index(i_right)
        # TODO: move to element.LagrangeFR
        assert isinstance(curr, element.LagrangeFR)
        assert isinstance(left, element.LagrangeFR)
        assert isinstance(right, element.LagrangeFR)
        if i_curr not in self._index_to_matrices:
            self._index_to_matrices[i_curr] = curr.get_dissipation_matrices(
                left.expansion(), right.expansion())
        mat_d, mat_e, mat_f = self._index_to_matrices[i_curr]
        curr_column = curr.get_solution_column()
        dissipation = mat_d @ curr_column
        # dissipation += mat_e @ left.get_solution_column()
        # dissipation += mat_f @ right.get_solution_column()
        for k in range(curr.n_term()):
            dissipation[k] *= curr.expansion().get_node_weight(k)
        dissipation = curr_column.transpose() @ dissipation
        assert dissipation < 0
        # print(f'dissipation = {dissipation:.2e}')
        oscillation_energy = self._get_oscillation_energy(curr.expansion(),
            left.expansion(), right.expansion())
        return oscillation_energy / (-dissipation * self._tau)

    def generate(self, troubled_cell_indices, grid: concept.Grid, periodic: bool):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._get_viscous_coeff(grid, i_cell)
            coeff = min(coeff, self._nu_max)
            # print(f'ν[{i_cell}] = {coeff:.2e}')
            self._index_to_coeff[i_cell] = coeff


if __name__ == '__main__':
    pass
