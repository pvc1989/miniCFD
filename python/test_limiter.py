"""Tests for various limiters.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import equation
import riemann
import spatial
import detector
import limiter
import test_detector


class TestLimiters(unittest.TestCase):
    """Test various oscillation limiters.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = -np.pi * 4
        self._x_right = np.pi * 4
        self._n_element = 73
        self._equation = equation.LinearAdvection(1.0)
        self._riemann = riemann.LinearAdvection(1.0)

    def build_scheme(self, method: spatial.PiecewiseContinuous,
            degree: int) -> spatial.PiecewiseContinuous:
        scheme = method(self._equation, self._riemann,
            degree, self._n_element, self._x_left, self._x_right)
        return scheme

    def test_limiters(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreFR, degree)
        u_init = test_detector.smooth
        scheme.initialize(u_init)
        detectors = [
            detector.Krivodonova2004(),
            detector.LiAndRen2011(),
        ]
        limiters = [
            limiter.SimpleWENO(),
            limiter.PWeighted(k_trunc=1e0),
            limiter.PWeighted(k_trunc=1e100),
        ]
        markers = ['1', '2', '3', '4']
        _, ax = plt.subplots(figsize=[6, 5])
        if u_init == test_detector.jumps:
            axins = ax.inset_axes([0.75, 0.10, 0.20, 0.25])
            axins.set_xlim(-1, 1)
            axins.set_ylim(-3.2, -2.2)
        else:
            axins = ax.inset_axes([0.35, 0.10, 0.35, 0.25])
            axins.set_xlim(-22, -19)
            axins.set_ylim(-0.4, +0.6)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x()
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_approx, '--', label=r'$p=4$, No Limiter')
        axins.plot(x_values, u_approx, '--')
        for i in range(len(limiters)):
            scheme.initialize(u_init)
            # indices = detectors[1].get_troubled_cell_indices(scheme)
            indices = np.arange(0, scheme.n_element())
            limiters[i].reconstruct(scheme, indices)
            for k in range(len(points)):
                u_approx[k] = scheme.get_solution_value(points[k])
            plt.plot(x_values, u_approx, marker=markers[i],
                label=r'$p=4$, '+f'{limiters[i].name()}')
            axins.plot(x_values, u_approx, marker=markers[i])
        plt.plot(x_values, u_exact, '-', label='Exact Solution')
        axins.plot(x_values, u_exact, '-')
        ax.indicate_inset_zoom(axins, edgecolor="gray")
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$u^h$')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_limiters.pdf')


if __name__ == '__main__':
    unittest.main()
