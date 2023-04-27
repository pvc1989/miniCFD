import numpy as np

import concept
import expansion
import detector


class Persson2006(detector.Persson2006):
    """Artificial viscosity for DG and FR schemes.

    See Per-Olof Persson and Jaime Peraire, "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods", in 44th AIAA Aerospace Sciences Meeting and Exhibit (Reno, Nevada, USA: American Institute of Aeronautics and Astronautics, 2006).
    """

    def __init__(self, kappa=0.1) -> None:
        self._kappa = kappa

    def get_viscous_coeff(self, u_approx: expansion.Taylor):
        s_0 = -4 * np.log10(u_approx.degree())
        s_gap = np.log10(self.get_smoothness_value(u_approx)) - s_0
        nu = u_approx.length() / u_approx.degree()
        if s_gap > self._kappa:
            pass
        elif s_gap > -self._kappa:
            nu *= 0.5 * (1 + np.sin(s_gap / self._kappa * np.pi / 2))
        else:
            nu = 0.0
        return nu


if __name__ == '__main__':
    pass
