"""Tests for some temporal schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from polynomial import RightRadau, Lagrange


class TestRightRadau(unittest.TestCase):
    """Test the RightRadau class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._degree = 5
        self._radau = RightRadau(self._degree)

    def test_plot(self):
        """Plot the curves of the polynomial and its derivative."""
        n_plot = 201
        x_plot = np.linspace(-1.0, 1.0, n_plot)
        values = np.zeros(n_plot)
        derivatives = np.zeros(n_plot)
        for i in range(n_plot):
            point_i = x_plot[i]
            values[i] = self._radau.get_function_value(point_i)
            derivatives[i] = self._radau.get_gradient_value(point_i)
        plt.figure()
        plt.plot(x_plot, values, 'r--', label=r'$R(\xi)$')
        plt.plot(x_plot, derivatives, 'b-', label= r'$dR/d\xi$')
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
        n_plot = 201
        x_plot = np.linspace(0.0, 10.0, n_plot)
        u_exact = my_function(x_plot)
        u_approx = np.zeros(n_plot)
        for i in range(n_plot):
            point_i = x_plot[i]
            u_approx[i] = self._lagrange.get_function_value(point_i)
        plt.figure()
        plt.plot(x_plot, u_exact, 'r--', label='Exact')
        plt.plot(x_plot, u_approx, 'b-', label= 'Lagrange')
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
        n_test = 20
        x_test = np.linspace(0.0, 10.0, n_test)
        for point in x_test:
            self.assertAlmostEqual(my_function_gradient(point),
                self._lagrange.get_gradient_value(point))


if __name__ == '__main__':
    unittest.main()
