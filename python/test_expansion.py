"""Tests for some expansions.
"""
import unittest
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

import expansion


class TestLagrange(unittest.TestCase):
    """Test the expansion.Lagrange class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = 10.0
        self._expansion = expansion.Lagrange(5, self._x_left, self._x_right)

    def test_coordinate_transforms(self):
        """Test coordinate transforms.
        """
        points = np.linspace(self._x_left, self._x_right, num=201)
        for x_global in points:
            x_local = self._expansion.global_to_local(x_global)
            self.assertAlmostEqual(x_global,
                self._expansion.local_to_global(x_local))

    def test_plot(self):
        """Plot the curves of a function and its approximations."""
        points = np.linspace(self._x_left, self._x_right, num=201)
        def my_function(point):
            return np.sin(point)
        exact_values = my_function(points)
        plt.figure()
        plt.plot(points, exact_values, '^', label='Exact')
        for degree in range(8):
            my_expansion = expansion.Lagrange(degree,
                self._x_left, self._x_right)
            my_expansion.approximate(my_function)
            approx_values = np.ndarray(len(points))
            for i in range(len(points)):
                approx_values[i] = my_expansion.get_function_value(points[i])
            plt.plot(points, approx_values, label= f'$p={degree}$')
        plt.legend()
        # plt.show()
        plt.savefig("expansion_on_lagrange.pdf")

    def test_values_at_sample_points(self):
        """Test values at sample points.
        """
        my_function = np.sin
        self._expansion.approximate(my_function)
        for point in self._expansion.get_sample_points():
            self.assertAlmostEqual(my_function(point), self._expansion.get_function_value(point))

    def test_get_basis_values_and_gradients(self):
        """Test methods for getting values and gradients of basis.
        """
        points = np.linspace(self._x_left, self._x_right, num=201)
        for point in points:
            # 2nd-order finite difference approximation
            delta = 0.0001
            values_right = self._expansion.get_basis_values(point + delta)
            values_left = self._expansion.get_basis_values(point - delta)
            gradients_approx = (values_right - values_left) / (delta * 2)
            gradients_actual = self._expansion.get_basis_gradients(point)
            norm = np.linalg.norm(gradients_actual - gradients_approx)
            self.assertAlmostEqual(norm, 0.0)

    def test_get_gradient_value(self):
        """Test the method for getting gradient values.
        """
        def my_function(point):
            return (point-5.0)**4 + (point-4.0)**3
        def my_function_gradient(point):
            return 4 * (point-5.0)**3 + 3 * (point-4.0)**2
        self._expansion.approximate(my_function)
        n_point = 20
        points = np.linspace(0.0, 10.0, n_point)
        for point in points:
            self.assertAlmostEqual(my_function_gradient(point),
                self._expansion.get_gradient_value(point))


class TestLegendre(unittest.TestCase):
    """Test the Legendre class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = 10.0
        self._expansion = expansion.Legendre(5, self._x_left, self._x_right)

    def test_plot(self):
        """Plot the curves of a function and its approximations."""
        points = np.linspace(self._x_left, self._x_right, num=201)
        def my_function(point):
            return np.sin(point)
        exact_values = my_function(points)
        plt.figure()
        plt.plot(points, exact_values, '^', label='Exact')
        for degree in range(8):
            my_expansion = expansion.Legendre(degree,
                self._x_left, self._x_right)
            my_expansion.approximate(my_function)
            approx_values = np.ndarray(len(points))
            for i in range(len(points)):
                approx_values[i] = my_expansion.get_function_value(points[i])
            plt.plot(points, approx_values, label= f'$p={degree}$')
        plt.legend()
        # plt.show()
        plt.savefig("expansion_on_legendre.pdf")

    def test_get_basis_values_and_gradients(self):
        """Test methods for getting values and gradients of basis.
        """
        points = np.linspace(self._x_left, self._x_right, num=201)
        for point in points:
            # 2nd-order finite difference approximation
            delta = 0.0001
            values_right = self._expansion.get_basis_values(point + delta)
            values_left = self._expansion.get_basis_values(point - delta)
            gradients_approx = (values_right - values_left) / (delta * 2)
            gradients_actual = self._expansion.get_basis_gradients(point)
            norm = np.linalg.norm(gradients_actual - gradients_approx)
            self.assertAlmostEqual(norm, 0.0)

    def test_get_gradient_value(self):
        """Test the method for getting gradient values.
        """
        def my_function(point):
            return (point-5.0)**4 + (point-4.0)**3
        def my_function_gradient(point):
            return 4 * (point-5.0)**3 + 3 * (point-4.0)**2
        self._expansion.approximate(my_function)
        points = np.linspace(self._x_left, self._x_right, num=201)
        for point in points:
            self.assertAlmostEqual(my_function_gradient(point),
                self._expansion.get_gradient_value(point))

    def test_orthogonality(self):
        weight_matrix = np.eye(self._expansion.n_term())
        for k in range(self._expansion.n_term()):
            weight_matrix[k][k] = self._expansion.get_mode_weight(k)
        for k in range(self._expansion.n_term()):
            for l in range(self._expansion.n_term()):
                  def integrand(x_global):
                      values = self._expansion.get_basis_values(x_global)
                      return values[k] * values[l]
                  integral, _ = integrate.quad(integrand,
                      self._x_left, self._x_right)
                  self.assertAlmostEqual(integral, weight_matrix[k][l])

if __name__ == '__main__':
    unittest.main()
