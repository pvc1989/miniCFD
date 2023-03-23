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


class TestLimiters(unittest.TestCase):
    """Test various oscillation limiters.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = -np.pi * 4
        self._x_right = np.pi * 4
        self._n_element = 43
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
        def u_init(x):
            y = (x - scheme.x_left()*0.75) * (x - scheme.x_right()*0.75)
            return np.sign(np.sin(x)) * np.abs(y) * (y < 0)
        scheme.initialize(u_init)
        detectors = [
            detector.Krivodonova2004(),
            detector.LiAndRen2011(),
        ]
        limiters = [
            limiter.SimpleWENO(),
        ]
        markers = ['1', '2', '3']
        plt.figure()
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(points, u_approx, '--', label=r'$p=4$, no limiter')
        for i in range(len(limiters)):
            indices = detectors[1].get_smoothness_values(scheme) > 1
            limiters[i].reconstruct(scheme, indices)
            for k in range(len(points)):
                u_approx[k] = scheme.get_solution_value(points[k])
            plt.plot(points, u_approx, marker=markers[i],
                label=r'$p=4$, '+f'{limiters[i].name()}')
        plt.plot(points, u_exact, 'r-', label=r'$p=\infty$, i.e. Exact')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u^h$')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_limiters.pdf')


if __name__ == '__main__':
    unittest.main()
