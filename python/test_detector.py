"""Tests for various jump detectors.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import equation
import riemann
import spatial
import detector


def smooth(x):
    x_shift = 8
    gauss_width = 1.25
    value = (np.exp(-((x - x_shift) / gauss_width)**2 / 2)
        * np.sin(5 * x))
    value += (np.exp(-((x + x_shift) / gauss_width)**2 / 2)
        * np.sin(10 * x))
    return value + 0


def jumps(x):
    a = -np.pi * 3
    b = +np.pi * 3
    sign = np.sign(np.sin(x))
    amplitude = b - x
    if x < 0:
        amplitude = x - a
    return sign * amplitude + 0


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

    def build_scheme(self, method: spatial.FiniteElement,
            degree: int) -> spatial.FiniteElement:
        scheme = method(self._equation, self._riemann,
            degree, self._n_element, self._x_left, self._x_right)
        return scheme

    def test_smoothness_values(self):
        degree = 4
        scheme = self.build_scheme(spatial.LagrangeFR, degree)
        u_init = jumps
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
        x_values = points / scheme.delta_x()
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_approx, '.', label=r'$p=4$')
        plt.plot(x_values, u_exact, 'r-', label=r'$p=\infty$')
        plt.legend()
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$u^h$')
        plt.subplot(3,1,(2,3))
        plt.semilogy()
        x_values = centers / scheme.delta_x()
        for i in range(len(detectors)):
            y_values = detectors[i].get_smoothness_values(scheme)
            plt.plot(x_values, y_values, markers[i],
                label=r'$p=4$, '+detectors[i].name())
        x_values = [
            scheme.x_left() / scheme.delta_x(),
            scheme.x_right() / scheme.delta_x()
        ]
        plt.plot(x_values, [1, 1], label=r'$Smoothness=1$')
        plt.legend()
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$Smoothness$')
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_detectors.pdf')


if __name__ == '__main__':
    unittest.main()
