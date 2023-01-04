"""Test ODE systems from spatial schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from spatial import LagrangeFR
import equation
import riemann

class TestLagrangeFR(unittest.TestCase):
    """Test the element for implement flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._n_element = 3
        self._degree = 1
        a_const = np.pi
        self._spatial = LagrangeFR(
            equation.LinearAdvection(a_const),
            riemann.LinearAdvection(a_const),
            self._degree, self._n_element, self._x_left, self._x_right)

    def test_plot(self):
        """Plot the curves of the approximate solution and the two fluxes.
        """
        function = np.sin
        self._spatial.initialize(function)
        n_point = 101
        points = np.linspace(self._x_left, self._x_right, n_point)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        discontinuous_flux = np.ndarray(n_point)
        continuous_flux = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = function(point_i)
            discontinuous_flux[i] = self._spatial.get_discontinuous_flux(point_i)
            continuous_flux[i] = self._spatial.get_continuous_flux(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(points, approx_solution, 'b.', label='Approximate Solution')
        plt.plot(points, expect_solution, 'r--', label='Exact Solution')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, discontinuous_flux, 'b.', label='Discontinuous Flux')
        plt.plot(points, continuous_flux, 'r--', label='Continuous Flux')
        plt.legend()
        # plt.show()
        plt.savefig("fr_approx.pdf")


if __name__ == '__main__':
    unittest.main()
