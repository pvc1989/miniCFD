import numpy as np

import concept
import expansion
import element
import detector


class Off(concept.Viscous):

    def __init__(self, const) -> None:
        pass

    def name(self, verbose=False) -> str:
        return "Off"

    def generate(self, troubled_cell_indices, elements, periodic: bool):
        pass


class Constant(concept.Viscous):

    def __init__(self, const=0.0) -> None:
        super().__init__()
        self._const = const

    def name(self, verbose=False) -> str:
        return 'Constant (' + r'$\nu=$' + f'{self._const})'

    def generate(self, troubled_cell_indices, elements, periodic: bool):
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
        return "Persson (2006)"

    def _get_viscous_coeff(self, cell: concept.Element):
        u_approx = cell.expansion
        s_0 = -4 * np.log10(u_approx.degree())
        smoothness = detector.Persson2006.get_smoothness_value(u_approx)
        s_gap = np.log10(smoothness) - s_0
        print(s_gap)
        nu = u_approx.coordinate.length() / u_approx.degree()
        if s_gap > self._kappa:
            pass
        elif s_gap > -self._kappa:
            nu *= 0.5 * (1 + np.sin(s_gap / self._kappa * np.pi / 2))
        else:
            nu = 0.0
        return nu

    def generate(self, troubled_cell_indices, elements, periodic: bool):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._get_viscous_coeff(elements[i_cell])
            print(f'nu[{i_cell}] = {coeff}')
            self._index_to_coeff[i_cell] = coeff


class Energy(concept.Viscous):

    def __init__(self, delta_t=0.01) -> None:
        super().__init__()
        self._delta_t = delta_t
        self._index_to_matrices = dict()

    def name(self, verbose=False) -> str:
        return 'Energy (' + r'$\Delta t=$' + f'{self._delta_t})'

    def _build_matrices(self, curr: element.LagrangeFR,
            left: element.LagrangeFR, right: element.LagrangeFR):
        shape = (curr.n_term(), curr.n_term())
        points = curr.get_sample_points()
        mat_a = np.ndarray(shape)
        for i in range(curr.n_term()):
            mat_a[i] = curr.get_basis_gradients(points[i])
        mat_b = np.zeros(shape)
        for k in range(curr.n_term()):
            for l in range(curr.n_term()):
                mat_b[k] += mat_a[k][l] * mat_a[l]
        # print('B =\n', mat_b)
        basis_values_curr_left = curr.get_basis_values(curr.x_left())
        basis_values_curr_right = curr.get_basis_values(curr.x_right())
        correction_gradients_left = np.ndarray(curr.n_term())
        correction_gradients_right = np.ndarray(curr.n_term())
        for i in range(curr.n_term()):
            correction_gradients_left[i], correction_gradients_right[i] \
                = curr.get_correction_gradients(points[i])
        mat_c = np.zeros(shape)
        for k in range(curr.n_term()):
            for l in range(curr.n_term()):
                a = correction_gradients_right[k] * basis_values_curr_right[l]
                a += correction_gradients_left[k] * basis_values_curr_left[l]
                mat_c[k] += a * mat_a[l]
        # print('C =\n', mat_c)
        # TODO: use DDG methods
        beta_0, beta_1 = 3, 1/12
        delta_x = curr.length()
        d_minus_right = 0.5 * curr.get_basis_gradients(curr.x_right()) \
            - beta_0 / delta_x * basis_values_curr_right \
            - beta_1 * delta_x * curr.get_basis_hessians(curr.x_right())
        d_minus_left = 0.5 * curr.get_basis_gradients(curr.x_left()) \
            + beta_0 / delta_x * basis_values_curr_left \
            + beta_1 * delta_x * curr.get_basis_hessians(curr.x_left())
        mat_d = np.ndarray(shape)
        for k in range(curr.n_term()):
            mat_d[k] = correction_gradients_right[k] * d_minus_right
            mat_d[k] += correction_gradients_left[k] * d_minus_left
        # print('D- =\n', mat_d)
        # print('D =\n', mat_b - mat_c + mat_d)
        d_plus_right = 0.5 * right.get_basis_gradients(right.x_left()) \
            + beta_0 / delta_x * right.get_basis_values(right.x_left()) \
            + beta_1 * delta_x * right.get_basis_hessians(right.x_left())
        d_plus_left = 0.5 * left.get_basis_gradients(left.x_right()) \
            - beta_0 / delta_x * left.get_basis_values(left.x_right()) \
            - beta_1 * delta_x * left.get_basis_hessians(left.x_right())
        mat_e = np.ndarray(shape)
        mat_f = np.ndarray(shape)
        for k in range(curr.n_term()):
            mat_e[k] = correction_gradients_left[k] * d_plus_left
            mat_f[k] = correction_gradients_right[k] * d_plus_right
        # print('E =\n', mat_e)
        # print('F =\n', mat_f)
        return (mat_b, mat_c, mat_d, mat_e, mat_f)

    def _get_extra_energy(self, curr: expansion.Lagrange,
            left: expansion.Lagrange, right: expansion.Lagrange):
        points = curr.get_sample_points()
        left_values = np.ndarray(len(points))
        right_values = np.ndarray(len(points))
        left_shift = left.x_right() - curr.x_left()
        right_shift = right.x_left() - curr.x_right()
        for i in range(len(points)):
            x_curr = points[i]
            curr_value = curr.get_function_value(x_curr)
            left_values[i] = curr_value - left.get_function_value(
                x_curr + left_shift)
            right_values[i] = curr_value - right.get_function_value(
                x_curr + right_shift)
        extra_energy = min(np.linalg.norm(left_values),
            np.linalg.norm(right_values))**2 / 2
        # print(left_values, '\n', right_values)
        # print(f'extra_energy = {extra_energy:.2e}')
        return extra_energy

    def _get_viscous_coeff(self, elements, i_curr):
        i_left = i_curr - 1
        i_right = (i_curr + 1) % len(elements)
        curr = elements[i_curr]
        left = elements[i_left]
        right = elements[i_right]
        # TODO: move to element.LagrangeFR
        assert isinstance(curr, element.LagrangeFR)
        assert isinstance(left, element.LagrangeFR)
        assert isinstance(right, element.LagrangeFR)
        if i_curr not in self._index_to_matrices:
            self._index_to_matrices[i_curr] = self._build_matrices(curr, left, right)
        mat_b, mat_c, mat_d, mat_e, mat_f = self._index_to_matrices[i_curr]
        curr_column = curr.get_solution_column()
        dissipation = (mat_b - mat_c + mat_d) @ curr_column
        dissipation += mat_e @ left.get_solution_column()
        dissipation += mat_f @ right.get_solution_column()
        dissipation = curr_column.transpose() @ dissipation
        # assert dissipation < 0
        # print(f'dissipation = {dissipation:.2e}')
        extra_energy = self._get_extra_energy(curr.expansion, left.expansion,
            right.expansion)
        return extra_energy / (-dissipation * self._delta_t)

    def generate(self, troubled_cell_indices, elements, periodic: bool):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._get_viscous_coeff(elements, i_cell)
            print(f'ν[{i_cell}] = {coeff:.2e}')
            self._index_to_coeff[i_cell] = coeff


if __name__ == '__main__':
    pass
