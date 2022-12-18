"""Tests for some temporal schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from polynomial import Lagrange


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
            return np.cos(point / 10.0)
        self._lagrange.approximate(my_function)
        n_plot = 201
        x_plot = np.linspace(0.0, 10.0, n_plot)
        u_exact = my_function(x_plot)
        u_approx = np.zeros(n_plot)
        for i in range(n_plot):
            point_i = x_plot[i]
            u_approx[i] = self._lagrange.get_function_value(point_i)
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


if __name__ == '__main__':
    unittest.main()
