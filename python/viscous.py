import numpy as np

import concept
import expansion
import detector


class Off(concept.Viscous):

    def __init__(self, const) -> None:
        pass

    def name(self, verbose=False) -> str:
        return "Off"

    def generate(self, scheme: concept.SpatialScheme, troubled_cell_indices):
        pass


class Constant(concept.Viscous):

    def __init__(self, const=0.0) -> None:
        self._const = const

    def name(self, verbose=False) -> str:
        return 'Constant (' + r'$\nu=$' + f'{self._const})'

    def generate(self, scheme: concept.SpatialScheme, troubled_cell_indices):
        pass

    def get_coeff(self, i_cell: int):
        return self._const


class Persson2006(concept.Viscous):
    """Artificial viscosity for DG and FR schemes.

    See Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006).
    """

    def __init__(self, kappa=0.1) -> None:
        self._kappa = kappa
        self._coeffs = np.ndarray(0)

    def name(self, verbose=False) -> str:
        return "Persson (2006)"

    def _get_viscous_coeff(self, u_approx: expansion.Taylor):
        s_0 = -4 * np.log10(u_approx.degree())
        smoothness = detector.Persson2006.get_smoothness_value(u_approx)
        s_gap = np.log10(smoothness) - s_0
        nu = u_approx.length() / u_approx.degree()
        if s_gap > self._kappa:
            pass
        elif s_gap > -self._kappa:
            nu *= 0.5 * (1 + np.sin(s_gap / self._kappa * np.pi / 2))
        else:
            nu = 0.0
        return nu

    def generate(self, scheme: concept.SpatialScheme, troubled_cell_indices):
        self._coeffs = np.zeros(scheme.n_element())
        for i_cell in troubled_cell_indices:
            u_approx = scheme.get_element_by_index(i_cell).get_expansion()
            self._coeffs[i_cell] = self._get_viscous_coeff(u_approx)

    def get_coeff(self, i_cell: int):
        return self._coeffs[i_cell]


if __name__ == '__main__':
    pass
