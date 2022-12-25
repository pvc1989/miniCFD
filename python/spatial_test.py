"""Test ODE systems from spatial discretizations.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from spatial import FluxReconstruction
import equation, riemann

class TestFluxReconstruction(unittest.TestCase):
    """Test the element for implement flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._n_element = 4
        self._degree = 1
        a_const = np.pi
        self._spatial = FluxReconstruction(self._x_left, self._x_right,
              self._n_element, self._degree,
              equation.LinearAdvection(a_const),
              riemann.LinearAdvection(a_const))

    def test_sizes(self):
        """Test various sizes.
        """
        n_dof = self._n_element * (self._degree + 1)
        self.assertEqual(n_dof, self._spatial.n_dof())
        column = self._spatial.get_unknown()
        self.assertEqual(n_dof, len(column))
        column = self._spatial.get_residual()
        self.assertEqual(n_dof, len(column))


if __name__ == '__main__':
    unittest.main()
