import numpy as np

import concept
import expansion
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
        self._const = const

    def name(self, verbose=False) -> str:
        return 'Constant (' + r'$\nu=$' + f'{self._const})'

    def generate(self, troubled_cell_indices, elements, periodic: bool):
        pass

    def get_coeff(self, i_cell: int):
        return self._const


class Persson2006(concept.Viscous):
    """Artificial viscosity for DG and FR schemes.

    See Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006).
    """

    def __init__(self, kappa=0.1) -> None:
        self._kappa = kappa
        self._index_to_coeff = dict()

    def name(self, verbose=False) -> str:
        return "Persson (2006)"

    def _get_viscous_coeff(self, cell: concept.Element):
        u_approx = cell.get_expansion()
        s_0 = -4 * np.log10(u_approx.degree())
        smoothness = detector.Persson2006.get_smoothness_value(u_approx)
        s_gap = np.log10(smoothness) - s_0
        nu = u_approx._coordinate.length() / u_approx.degree()
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
            self._index_to_coeff[i_cell] = coeff

    def get_coeff(self, i_cell: int):
        if i_cell in self._index_to_coeff:
            return self._index_to_coeff[i_cell]
        else:
            return 0.0


if __name__ == '__main__':
    pass
