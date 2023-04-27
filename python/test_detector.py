"""Tests for various jump detectors.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import concept
import equation
import riemann
import spatial
import detector


markers = ['1', '2', '3', '4', '+', 'x']


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

    def test_smoothness_on_jumps(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreDG, degree)
        u_init = jumps
        scheme.initialize(u_init)
        centers = scheme.delta_x()/2 + np.linspace(scheme.x_left(),
            scheme.x_right() - scheme.delta_x(), self._n_element)
        detectors = [
          detector.Krivodonova2004(),
          detector.LiRen2011(),
          detector.ZhuShuQiu2021(),
          detector.LiRen2022(),
          detector.Persson2006(),
        ]
        plt.figure(figsize=(6,6))
        plt.subplot(3,1,1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x()
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_exact, 'r-', label=r'$p=\infty$')
        plt.plot(x_values, u_approx, 'g--', label=r'$p=4$')
        plt.legend()
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$u^h$')
        plt.grid()
        plt.subplot(3,1,(2,3))
        plt.semilogy()
        x_values = centers / scheme.delta_x()
        for i in range(len(detectors)):
            y_values = detectors[i].get_smoothness_values(scheme)
            plt.plot(x_values, y_values, markers[i],
                label=detectors[i].name())
        x_values = [
            scheme.x_left() / scheme.delta_x(),
            scheme.x_right() / scheme.delta_x()
        ]
        plt.plot(x_values, [1, 1], label=r'$Smoothness=1$')
        plt.legend()
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$Smoothness$')
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_smoothness_on_jumps.pdf')

    def test_smoothness_on_smooth(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreDG, degree)
        def kh(x):
            kh_min = np.pi * 0.0
            kh_max = np.pi * 0.5 * (scheme.degree() + 1)
            kh = (kh_max - kh_min) * (x - scheme.x_left())
            kh /= scheme.length()
            kh += kh_min
            return kh
        def u_init(x):
            k = kh(x) / scheme.delta_x()
            return np.sin(k * (x - scheme.x_left()))
        scheme.initialize(u_init)
        centers = scheme.delta_x()/2 + np.linspace(scheme.x_left(),
            scheme.x_right() - scheme.delta_x(), self._n_element)
        detectors = [
          detector.Krivodonova2004(),
          detector.LiRen2011(),
          detector.ZhuShuQiu2021(),
          detector.LiRen2022(),
          detector.Persson2006(),
        ]
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(3,1,1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x()
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_exact, 'r-', label=r'$p=\infty$')
        plt.plot(x_values, u_approx, 'g--', label=r'$p=4$')
        plt.legend()
        plt.xlabel(r'$x/h$')
        plt.ylabel(r'$u^h$')
        plt.grid()
        ax = fig.add_subplot(3,1,(2,3))
        plt.semilogy()
        x_values = kh(centers)
        for i in range(len(detectors)):
            y_values = detectors[i].get_smoothness_values(scheme)
            plt.plot(x_values, y_values, markers[i],
                label=detectors[i].name())
        x_values = [kh(scheme.x_left()), kh(scheme.x_right())]
        plt.plot(x_values, [1, 1], label=r'$Smoothness=1$')
        plt.legend(loc='right')
        plt.xlabel(r'$\kappa h$')
        plt.ylabel(r'$Smoothness$')
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_smoothness_on_smooth.pdf')

    def test_detectors_on_smooth(self):
        degree = 4
        scheme = self.build_scheme(spatial.LagrangeDG, degree)
        detectors = [
          detector.Krivodonova2004(),
          detector.LiRen2011(),
          detector.ZhuShuQiu2021(),
          detector.LiRen2022(),
          detector.Persson2006(),
        ]
        k_max = int((degree+1) * scheme.n_element() / 2)
        kappa_h = np.ndarray(k_max)
        active_counts = np.ndarray((len(detectors), k_max))
        for k in range(k_max):
            kappa = (k+1) * 2 * np.pi / scheme.length()
            kappa_h[k] = kappa * scheme.delta_x()
            def u_init(x):
                return np.sin(kappa * (x - scheme.x_left()))
            scheme.initialize(u_init)
            for i in range(len(detectors)):
                detector_i = detectors[i]
                assert isinstance(detector_i, concept.JumpDetector)
                active_counts[i][k] = len(
                    detector_i.get_troubled_cell_indices(scheme))
        fig = plt.figure()
        for i in range(len(detectors)):
            plt.plot(kappa_h/np.pi, active_counts[i]/scheme.n_element()*100,
                f'-{markers[i]}', label=detectors[i].name())
        plt.legend()
        plt.title(r'$\sin(\kappa x)$ approximated by '+scheme.name())
        plt.xlabel(r'$\kappa h/\pi$')
        plt.ylabel('Troubled Cell Count (%)')
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_detectors_on_smooth.pdf')


if __name__ == '__main__':
    unittest.main()
