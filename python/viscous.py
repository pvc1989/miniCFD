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
            print(f'nu[{i_cell}] = {coeff}')
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

    def __init__(self, const=0.0) -> None:
        super().__init__()
        self._const = const
        self._index_to_matrices = dict()

    def name(self, verbose=False) -> str:
        return 'Energy'

    def _build_matrices(self, curr: element.LagrangeFR,
            left: element.LagrangeFR, right: element.LagrangeFR):
        shape = (curr.n_term(), curr.n_term())
        mat_b = mat_c = mat_d = mat_e = mat_f = np.zeros(shape)
        return (mat_b, mat_c, mat_d, mat_e, mat_f)

    def _get_viscous_coeff(self, elements, i_curr):
        i_left = i_curr - 1
        i_right = (i_curr + 1) % len(elements)
        curr = elements[i_curr]
        left = elements[i_left]
        right = elements[i_right]
        assert isinstance(curr, element.LagrangeFR)
        assert isinstance(left, element.LagrangeFR)
        assert isinstance(right, element.LagrangeFR)
        if i_curr not in self._index_to_matrices:
            self._index_to_matrices[i_curr] = self._build_matrices(curr, left, right)
        mat_b, mat_c, mat_d, mat_e, mat_f = self._index_to_matrices[i_curr]
        curr_column = curr.get_solution_column()
        dissipation_rate = (mat_b - mat_c + mat_d) @ curr_column
        dissipation_rate += mat_e @ left.get_solution_column()
        dissipation_rate += mat_f @ right.get_solution_column()
        dissipation_rate = curr_column.transpose() @ dissipation_rate
        print('dissipation_rate = ', dissipation_rate)
        return self._const

    def generate(self, troubled_cell_indices, elements, periodic: bool):
        self._index_to_coeff.clear()
        for i_cell in troubled_cell_indices:
            coeff = self._get_viscous_coeff(elements, i_cell)
            print(f'nu[{i_cell}] = {coeff}')
            self._index_to_coeff[i_cell] = coeff


if __name__ == '__main__':
    pass
