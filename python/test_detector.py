"""Tests for various jump detectors.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import equation
import riemann
import spatial
import detector


class TestJumpDetectors(unittest.TestCase):
    """Test various jump detectors.
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

    def test_smoothness_values(self):
        degree = 4
        scheme = self.build_scheme(spatial.LagrangeFR, degree)
        def u_init(x):
            y = (x - scheme.x_left()*0.75) * (x - scheme.x_right()*0.75)
            return np.sign(np.sin(x)) * np.abs(y) * (y < 0) + 101
        scheme.initialize(u_init)
        centers = scheme.delta_x()/2 + np.linspace(scheme.x_left(),
            scheme.x_right() - scheme.delta_x(), self._n_element)
        detectors = [
          detector.Krivodonova2004(),
          detector.LiAndRen2011(),
          detector.ZhuAndQiu2021()
        ]
        markers = ['1', '2', '3']
        plt.figure(figsize=(6,6))
        plt.subplot(3,1,1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(points, u_approx, '.', label=r'$p=4$')
        plt.plot(points, u_exact, 'r-', label=r'$p=\infty$')
        plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u^h$')
        plt.subplot(3,1,(2,3))
        plt.semilogy()
        for i in range(len(detectors)):
            values = detectors[i].get_smoothness_values(scheme)
            plt.plot(centers, values, markers[i], label=detectors[i].name())
        plt.plot([self._x_left, self._x_right], [1, 1], label=r'$Smoothness=1$')
        plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$Smoothness$')
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_detectors.pdf')


if __name__ == '__main__':
    unittest.main()
