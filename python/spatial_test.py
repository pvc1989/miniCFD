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
        self._n_element = 5
        self._degree = 2
        a_const = np.pi
        self._spatial = FluxReconstruction(self._x_left, self._x_right,
              self._n_element, self._degree,
              equation.LinearAdvection(a_const),
              riemann.LinearAdvection(a_const))


    def test_plot(self):
        """Plot the curves of U^{discontinous} and F^{continous}.
        """
        function = np.sin
        self._spatial.initialize(function)
        n_point = 101
        points = np.linspace(self._x_left, self._x_right, n_point)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = function(point_i)
        plt.figure()
        plt.plot(points, expect_solution, 'r--', label='Exact Solution')
        plt.plot(points, approx_solution, 'b.', label='Approximate Solution')
        plt.legend()
        plt.show()
        plt.savefig("fr_approx.pdf")


if __name__ == '__main__':
    unittest.main()
