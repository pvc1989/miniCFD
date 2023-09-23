"""Tests for integrators.
"""
import unittest

import numpy as np
from scipy import integrate, special

import coordinate
import integrator


class TestGaussOnLinearCoordinate(unittest.TestCase):

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = np.random.rand()
        self._x_right = 1 + np.random.rand()
        self._coordinate = coordinate.Linear(self._x_left, self._x_right)

    def test_gauss_legendre(self):
        """Test Gauss--Legendre's (2k - 1)-degree algebraic accuracy.
        """
        for n_point in range(1, 20):
            gauss = integrator.GaussLegendre(self._coordinate, n_point)
            degree = 2 * n_point - 1 - self._coordinate.jacobian_degree()
            def integrand(x_global):
                return x_global**degree
            self.assertAlmostEqual(1.0, gauss.integrate(integrand) /
                integrate.quad(integrand, self._x_left, self._x_right)[0])

    def test_gauss_lobatto(self):
        """Test Gauss-Lobatto's (2k - 3)-degree algebraic accuracy.
        """
        for n_point in range(3, 20):
            gauss = integrator.GaussLobatto(self._coordinate, n_point)
            degree = 2 * n_point - 3 - self._coordinate.jacobian_degree()
            def integrand(x_global):
                return x_global**degree
            self.assertAlmostEqual(1.0, gauss.integrate(integrand) /
                integrate.quad(integrand, self._x_left, self._x_right)[0])


if __name__ == '__main__':
    unittest.main()
