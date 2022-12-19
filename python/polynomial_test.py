"""Tests for some temporal schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from polynomial import Radau, Lagrange


class TestRadau(unittest.TestCase):
    """Test the Radau class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._degree = 5
        self._radau = Radau(self._degree)

    def test_plot(self):
        """Plot the curves of the two Radau polynomials and their derivatives."""
        n_point = 201
        points = np.linspace(-1.0, 1.0, n_point)
        left_values = np.zeros(n_point)
        left_derivatives = np.zeros(n_point)
        right_values = np.zeros(n_point)
        right_derivatives = np.zeros(n_point)
        for i in range(n_point):
            point_i = points[i]
            left_values[i], right_values[i] = self._radau.get_function_value(point_i)
            left_derivatives[i], right_derivatives[i] = self._radau.get_gradient_value(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(points, left_values, 'r--', label=r'$R_{-1}(\xi)$')
        plt.plot(points, right_values, 'b-', label=r'$R_{+1}(\xi)$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, left_derivatives, 'r--', label= r'$dR_{-1}/d\xi$')
        plt.plot(points, right_derivatives, 'b-', label= r'$dR_{+1}/d\xi$')
        plt.legend()
        plt.show()
        plt.savefig("radau.pdf")


class TestLagrange(unittest.TestCase):
    """Test the Lagrange class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._n_point = 5
        self._sample_points = np.linspace(0.0, 10.0, self._n_point)
        self._lagrange = Lagrange(self._sample_points)

    def test_coordinate_transforms(self):
        """Test coordinate transforms.
        """
        points = np.linspace(0.0, 10.0, 20)
        for x_global in points:
            x_local = self._lagrange._global_to_local(x_global)
            self.assertAlmostEqual(x_global,
                self._lagrange._local_to_global(x_local))

    def test_plot(self):
        """Plot the curves of a function and its approximation."""
        def my_function(point):
            return (point-5.0)**5
        self._lagrange.approximate(my_function)
        n_point = 201
        points = np.linspace(0.0, 10.0, n_point)
        exact_values = my_function(points)
        approx_values = np.zeros(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_values[i] = self._lagrange.get_function_value(point_i)
        plt.figure()
        plt.plot(points, exact_values, 'r--', label='Exact')
        plt.plot(points, approx_values, 'b-', label= 'Lagrange')
        plt.legend()
        # plt.show()
        plt.savefig("lagrange.pdf")

    def test_values_at_sample_points(self):
        """Test values at sample points.
        """
        my_function = np.sin
        self._lagrange.approximate(my_function)
        for point in self._sample_points:
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
