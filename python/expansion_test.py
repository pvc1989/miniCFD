"""Tests for some expansions.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from expansion import Lagrange


class TestLagrange(unittest.TestCase):
    """Test the Lagrange class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        x_left = 0.0
        x_right = 10.0
        points = np.linspace(x_left + 0.5, x_right - 0.5, 5)
        self._lagrange = Lagrange(points, x_left, x_right)

    def test_coordinate_transforms(self):
        """Test coordinate transforms.
        """
        points = np.linspace(0.0, 10.0, 20)
        for x_global in points:
            x_local = self._lagrange.global_to_local(x_global)
            self.assertAlmostEqual(x_global,
                self._lagrange.local_to_global(x_local))

    def test_plot(self):
        """Plot the curves of a function and its approximations."""
        x_left = 0.0
        x_right = np.pi
        n_point = 201
        points = np.linspace(x_left, x_right, n_point)
        def my_function(point):
            return np.sin(point)
        exact_values = my_function(points)
        plt.plot(points, exact_values, '^', label='Exact')
        for degree in range(5):
            sample_points = np.linspace(x_left, x_right, degree+1)
            lagrange = Lagrange(sample_points, x_left, x_right)
            lagrange.approximate(my_function)
            approx_values = np.ndarray(n_point)
            for i in range(n_point):
                approx_values[i] = lagrange.get_function_value(points[i])
            plt.plot(points, approx_values, label= f'{degree}-degree Lagrange')
        plt.legend()
        # plt.show()
        plt.savefig("lagrange_expansion.pdf")

    def test_values_at_sample_points(self):
        """Test values at sample points.
        """
        my_function = np.sin
        self._lagrange.approximate(my_function)
        for point in self._lagrange.get_sample_points():
            self.assertEqual(my_function(point), self._lagrange.get_function_value(point))

    def test_get_gradient_value(self):
        """Test the method for getting gradient values.
        """
        def my_function(point):
            return (point-5.0)**4 + (point-4.0)**3
        def my_function_gradient(point):
            return 4 * (point-5.0)**3 + 3 * (point-4.0)**2
        self._lagrange.approximate(my_function)
        n_point = 20
        points = np.linspace(0.0, 10.0, n_point)
        for point in points:
            self.assertAlmostEqual(my_function_gradient(point),
                self._lagrange.get_gradient_value(point))


if __name__ == '__main__':
    unittest.main()
