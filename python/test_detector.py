"""Tests for various jump detectors.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import concept
import riemann
import spatial
import detector


markers = ['1', '2', '3', '4', '+', 'x']

def smooth(x, k_left=10, k_right=5):
    x_shift = 6
    gauss_width = 1.25
    value = (np.exp(-((x - x_shift) / gauss_width)**2 / 2)
        * np.sin(k_right * x))
    value += (np.exp(-((x + x_shift) / gauss_width)**2 / 2)
        * np.sin(k_left * x))
    return value + 10


def jumps(x):
    a = -np.pi * 3
    b = +np.pi * 3
    sign = np.sign(np.sin(x))
    amplitude = b - x
    if x < 0:
        amplitude = x - a
    return sign * amplitude + 10


class TestDetectors(unittest.TestCase):
    """Test various jump detectors.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = -np.pi * 4
        self._x_right = np.pi * 4
        self._n_element = 73
        self._riemann = riemann.LinearAdvection(1.0)

    def build_scheme(self, method: spatial.FiniteElement,
            degree: int) -> spatial.FiniteElement:
        scheme = method(self._riemann,
            degree, self._n_element, self._x_left, self._x_right)
        return scheme

    def test_smoothness_on_jumps(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreDG, degree)
        u_init = jumps
        scheme.initialize(u_init)
        centers = scheme.delta_x(0)/2 + np.linspace(scheme.x_left(),
            scheme.x_right() - scheme.delta_x(0), self._n_element)
        detectors = [
          detector.Krivodonova2004(),
          detector.LiWanAi2011(),
          detector.ZhuJun2021(),
          detector.LiYanHui2022(),
          detector.Persson2006(),
        ]
        plt.figure(figsize=(6,6))
        plt.subplot(3,1,1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x(0)
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_exact, 'r-', label='Exact')
        plt.plot(x_values, u_approx, 'g--', label=r'$p=4$')
        plt.legend()
        plt.title('Jumps Approximated by '+scheme.name(False))
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$u^h$')
        plt.grid()
        plt.subplot(3,1,(2,3))
        plt.semilogy()
        x_values = centers / scheme.delta_x(0)
        for i in range(len(detectors)):
            y_values = detectors[i].get_smoothness_values(scheme)
            plt.plot(x_values, y_values, markers[i],
                label=detectors[i].name())
        x_values = [
            scheme.x_left() / scheme.delta_x(0),
            scheme.x_right() / scheme.delta_x(0)
        ]
        plt.plot(x_values, [1, 1], label=r'$Smoothness=1$')
        plt.legend()
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$Smoothness$')
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_smoothness_on_jumps.svg')

    def test_smoothness_on_smooth(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreDG, degree)
        k1, k2 = 10, 20
        def u_init(x):
            return smooth(x, k1, k2)
        scheme.initialize(u_init)
        centers = scheme.delta_x(0)/2 + np.linspace(scheme.x_left(),
            scheme.x_right() - scheme.delta_x(0), self._n_element)
        detectors = [
          detector.Krivodonova2004(),
          detector.LiWanAi2011(),
          detector.ZhuJun2021(),
          detector.LiYanHui2022(),
          detector.Persson2006(),
        ]
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(3,1,1)
        points = np.linspace(self._x_left, self._x_right, self._n_element * 10)
        x_values = points / scheme.delta_x(0)
        u_approx = np.ndarray(len(points))
        u_exact = np.ndarray(len(points))
        for i in range(len(points)):
            u_approx[i] = scheme.get_solution_value(points[i])
            u_exact[i] = u_init(points[i])
        plt.plot(x_values, u_exact, 'r-', label='Exact')
        plt.plot(x_values, u_approx, 'g--', label=r'$p=4$')
        plt.legend()
        k1_h = k1 * scheme.delta_x(0)
        k2_h = k2 * scheme.delta_x(0)
        plt.title(r'$kh=$'+f'({k1_h:.2f}, {k2_h:.2f})' +
            ' Approximated by '+scheme.name(False))
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$u^h$')
        plt.grid()
        ax = fig.add_subplot(3,1,(2,3))
        plt.semilogy()
        for i in range(len(detectors)):
            y_values = detectors[i].get_smoothness_values(scheme)
            plt.plot(centers/scheme.delta_x(0), y_values, markers[i],
                label=detectors[i].name())
        x_values = [
            scheme.x_left() / scheme.delta_x(0),
            scheme.x_right() / scheme.delta_x(0)]
        plt.plot(x_values, [1, 1], label=r'$Smoothness=1$')
        plt.xlabel(r'$x\,/\,h$')
        plt.ylabel(r'$Smoothness$')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_smoothness_on_smooth.svg')

    def test_detectors_on_smooth(self):
        degree = 4
        scheme = self.build_scheme(spatial.LegendreDG, degree)
        detectors = [
          detector.Krivodonova2004(),
          detector.LiWanAi2011(),
          detector.ZhuJun2021(),
          detector.LiYanHui2022(),
          detector.Persson2006(),
        ]
        k_max = int((degree+1) * scheme.n_element() / 2)
        kappa_h = np.ndarray(k_max)
        active_counts = np.ndarray((len(detectors), k_max))
        for k in range(k_max):
            kappa = (k+1) * 2 * np.pi / scheme.length()
            kappa_h[k] = kappa * scheme.delta_x(0)
            def u_init(x):
                return 1+np.sin(kappa * (x - scheme.x_left()))
            scheme.initialize(u_init)
            for i in range(len(detectors)):
                detector_i = detectors[i]
                assert isinstance(detector_i, concept.Detector)
                troubled_cell_indices = \
                    detector_i.get_troubled_cell_indices(scheme)
                active_counts[i][k] = len(troubled_cell_indices)
        plt.figure()
        for i in range(len(detectors)):
            plt.plot(kappa_h/np.pi, active_counts[i]/scheme.n_element()*100,
                f'-{markers[i]}', label=detectors[i].name())
        plt.legend()
        plt.title(r'On $1+\sin(\kappa x)$ Approximated by '+scheme.name())
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.ylabel('Troubled Cell Count (%)')
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig('compare_detectors_on_smooth.svg')


if __name__ == '__main__':
    unittest.main()
