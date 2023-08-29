"""Test elements for spatial scheme.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from element import FRonLegendreRoots
from coordinate import LinearCoordinate
from riemann import LinearAdvection
from polynomial import Vincent


class TestFRonLegendreRoots(unittest.TestCase):
    """Test the element for implement flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._riemann = LinearAdvection(a_const=np.random.rand())
        self._equation = self._riemann.equation()
        self._degree = 4
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._test_points = np.linspace(self._x_left, self._x_right)
        self.coordinate = LinearCoordinate(self._x_left, self._x_right)
        self._element = FRonLegendreRoots(self._riemann, self._degree,
            self.coordinate)
        self._element.approximate(np.sin)

    def test_plot(self):
        """Plot the curves of F^{discontinous} and F^{continous}.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        n_point = 201
        points = np.linspace(self._x_left, self._x_right, n_point)
        dg_flux = np.zeros(n_point)
        fr_flux = np.zeros(n_point)
        for i in range(n_point):
            point_i = points[i]
            dg_flux[i] = self._element.get_dg_flux(point_i, 0.0)
            fr_flux[i] = self._element.get_fr_flux(
                point_i, upwind_flux_left, upwind_flux_right)
        plt.figure()
        plt.plot(points, dg_flux, 'r--', label='DG Flux')
        plt.plot(points, fr_flux, 'b-', label='FR Flux')
        plt.plot(self._x_left, upwind_flux_left, 'k<',
            label='Upwind Flux At Left')
        plt.plot(self._x_right, upwind_flux_right, 'k>',
            label='Upwind Flux At Right')
        plt.legend()
        # plt.show()
        plt.savefig("RadauFRonLegendreRoots.svg")

    def test_get_dg_flux(self):
        """Test the values of the discontinuous flux.
        """
        for x_global in self._test_points:
            flux_actual = self._element.get_dg_flux(x_global, 0.0)
            # dg_flux = a * u
            u_approx = self._element.get_solution_value(x_global)
            flux_expect = self._equation.get_convective_flux(u_approx)
            self.assertAlmostEqual(flux_expect, flux_actual)

    def test_get_fr_flux(self):
        """Test the values of the reconstructed continuous flux.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        self.assertAlmostEqual(upwind_flux_left,
            self._element.get_fr_flux(self._x_left,
                upwind_flux_left, upwind_flux_right))
        self.assertAlmostEqual(upwind_flux_right,
            self._element.get_fr_flux(self._x_right,
                upwind_flux_left, upwind_flux_right))
        vincent = Vincent(self._degree, Vincent.huyhn_lump_lobatto)
        for x_global in self._test_points:
            flux_actual = self._element.get_fr_flux(x_global,
                upwind_flux_left, upwind_flux_right)
            # fr_flux = dg_flux + correction
            flux_expect = self._element.get_dg_flux(x_global, 0.0)
            x_local = self.coordinate.global_to_local(x_global)
            left, right = vincent.local_to_value(x_local)
            flux_expect += left * (upwind_flux_left
                - self._element.get_dg_flux(self._x_left, 0.0))
            flux_expect += right * (upwind_flux_right
                - self._element.get_dg_flux(self._x_right, 0.0))
            self.assertAlmostEqual(flux_expect, flux_actual)

    def test_get_fr_flux_gradient(self):
        """Test the gradient of the reconstructed continuous flux.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        points = np.linspace(self._x_left, self._x_right)
        for x_global in points:
            gradient_actual = self._element.get_fr_flux_gradient(x_global,
                upwind_flux_left, upwind_flux_right)
            # 2nd-order finite difference approximation
            delta_x = 0.0001
            flux_right = self._element.get_fr_flux(x_global + delta_x,
                    upwind_flux_left, upwind_flux_right)
            flux_left = self._element.get_fr_flux(x_global - delta_x,
                    upwind_flux_left, upwind_flux_right)
            gradient_approx = (flux_right - flux_left) / (delta_x + delta_x)
            self.assertAlmostEqual(gradient_actual, gradient_approx)


if __name__ == '__main__':
    unittest.main()
