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
        self._n_element = 53
        self._equation = equation.LinearAdvection(1.0)
        self._riemann = riemann.LinearAdvection(1.0)
        self._detector = detector.ReportAll()

    def build_scheme(self, method: spatial.FiniteElement,
            degree: int) -> spatial.FiniteElement:
        scheme = method(self._equation, self._riemann,
            degree, self._n_element, self._x_left, self._x_right)
        return scheme

    def test_limiters_on_jumps(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreFR, degree)
        u_init = test_detector.jumps
        limiters = [
            limiter.LiWangRen2020(k_trunc=0),
            limiter.LiWangRen2020(k_trunc=1e0),
            limiter.LiWangRen2020(k_trunc=1e100),
            limiter.Xu2023(alpha=1.0),
            limiter.ZhongShu2013(),
        ]
        markers = ['1', '2', '3', '4', '+']
        _, ax = plt.subplots(figsize=[6, 5])
        axins1 = ax.inset_axes([0.1, 0.08, 0.20, 0.20])
        xmin = -1 / 73 * self._n_element
        xmax = +1 / 73 * self._n_element
        axins1.set_xlim(xmin, xmax)
        axins1.set_ylim(-3.2, -2.2)
        axins2 = ax.inset_axes([0.80, 0.08, 0.18, 0.30])
        xmin = +6 / 53 * self._n_element
        xmax = +8 / 53 * self._n_element
        axins2.set_xlim(xmin, xmax)
        axins2.set_ylim(-7.5, -3.5)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x()
        y_values = np.ndarray(len(points))
        for i in range(len(limiters)):
            scheme.initialize(u_init)
            indices = self._detector.get_troubled_cell_indices(scheme)
            limiters[i].reconstruct(scheme, indices)
            for k in range(len(points)):
                y_values[k] = scheme.get_solution_value(points[k])
            plt.plot(x_values, y_values, marker=markers[i],
                label=f'{limiters[i].name()}')
            axins1.plot(x_values, y_values, marker=markers[i])
            axins2.plot(x_values, y_values, marker=markers[i])
        scheme.initialize(u_init)
        for i in range(len(points)):
            y_values[i] = u_init(points[i])
        plt.plot(x_values, y_values, '-', label='Exact Solution')
        axins1.plot(x_values, y_values, '-')
        axins2.plot(x_values, y_values, '-')
        ax.indicate_inset_zoom(axins1, edgecolor="gray")
        ax.indicate_inset_zoom(axins2, edgecolor="gray")
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$u^h$')
        plt.title(r'$p=$'+f'{degree}, '+r'$h=$'+f'{scheme.delta_x():.3f}')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_limiters_on_jumps.pdf')

    def test_limiters_on_smooth(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreFR, degree)
        u_init = test_detector.smooth
        limiters = [
            limiter.LiWangRen2020(k_trunc=0),
            limiter.LiWangRen2020(k_trunc=1e0),
            limiter.LiWangRen2020(k_trunc=1e100),
            limiter.Xu2023(alpha=1.0),
            limiter.ZhongShu2013(),
        ]
        markers = ['1', '2', '3', '4', '+']
        _, ax = plt.subplots(figsize=[6, 5])
        axins = ax.inset_axes([0.35, 0.10, 0.35, 0.25])
        xmin = -28 / 73 * self._n_element
        xmax = -18 / 73 * self._n_element
        axins.set_xlim(xmin, xmax)
        axins.set_ylim(-1.1, +1.1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x()
        y_values = np.ndarray(len(points))
        for i in range(len(limiters)):
            scheme.initialize(u_init)
            indices = self._detector.get_troubled_cell_indices(scheme)
            limiters[i].reconstruct(scheme, indices)
            for k in range(len(points)):
                y_values[k] = scheme.get_solution_value(points[k])
            plt.plot(x_values, y_values, marker=markers[i],
                label=f'{limiters[i].name()}')
            axins.plot(x_values, y_values, marker=markers[i])
        scheme.initialize(u_init)
        for i in range(len(points)):
            y_values[i] = u_init(points[i])
        plt.plot(x_values, y_values, '-', label='Exact Solution')
        axins.plot(x_values, y_values, '-')
        ax.indicate_inset_zoom(axins, edgecolor="gray")
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$u^h$')
        plt.title(r'$p=$'+f'{degree}, '+r'$h=$'+f'{scheme.delta_x():.3f}')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_limiters_on_smooth.pdf')


if __name__ == '__main__':
    unittest.main()
