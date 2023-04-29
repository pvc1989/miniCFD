"""Tests for integrators.
"""
import unittest

import numpy as np
from scipy import integrate

from coordinate import LinearCoordinate
from integrator import GaussLegendre


class TestGaussLegendre(unittest.TestCase):

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = np.random.rand()
        self._x_right = 1 + np.random.rand()
        self._coordinate = LinearCoordinate(self._x_left, self._x_right)
        self._integrator = GaussLegendre(self._coordinate)

    def test_accuracy(self):
        """Test algebraic accuracy order.
        """
        for degree in range(0, 8):
            def integrand(x_global):
                return x_global**degree
            n_point = 1 + (degree+self._coordinate.jacobian_degree())//2
            self.assertAlmostEqual(
                self._integrator.fixed_quad_global(integrand, n_point),
                integrate.quad(integrand, self._x_left, self._x_right)[0])


if __name__ == '__main__':
    unittest.main()
