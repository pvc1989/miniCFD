"""Tests for some expansions.
"""
import unittest
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

import expansion
from coordinate import LinearCoordinate


class TestTaylor(unittest.TestCase):
    """Test the expansion.Taylor class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = -0.4
        self._x_right = 0.6  # avoid x_center = 0.0
        self._coordinate = LinearCoordinate(self._x_left, self._x_right)
        self._expansion = expansion.Taylor(5, self._coordinate)

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

    def test_get_basis_derivatives(self):
        """Test methods for getting derivatives of basis.
        """
        points = np.linspace(self._x_left, self._x_right, num=201)
        for point in points:
            gradients_expect = self._expansion.get_basis_gradients(point)
            gradients_actual = self._expansion.get_basis_derivatives(point)[1]
            norm = np.linalg.norm(gradients_actual - gradients_expect)
            self.assertAlmostEqual(norm, 0.0)

    def test_get_derivative_values(self):
        """Test methods for getting derivatives of u^h.
        """
        taylor = expansion.Taylor(5, self._coordinate, complex)
        # only approximate well near the center
        points = np.linspace(self._x_left/2, self._x_right/2, num=201)
        def function(x):
            return np.exp(1j * x)
        def derivative(x, k):
            return np.exp(1j * x) * (1j)**k
        taylor.approximate(function)
        for x in points:
            values = taylor.get_derivative_values(x)
            for k in range(1, taylor.n_term()):
                self.assertAlmostEqual(values[k], derivative(x, k),
                    places=taylor.degree()-k)

    def test_plot(self):
        """Plot the curves of a function and its approximations."""
        x_left, x_right = 0.0, 10.0
        points = np.linspace(x_left, x_right, num=201)
        def my_function(point):
            return np.sin(point)
        exact_values = my_function(points)
        plt.figure()
        plt.plot(points, exact_values, 'o', label='Exact')
        for degree in range(8):
            my_expansion = expansion.Taylor(degree,
                LinearCoordinate(x_left, x_right))
            my_expansion.approximate(my_function)
            approx_values = np.ndarray(len(points))
            for i in range(len(points)):
                approx_values[i] = my_expansion.get_function_value(points[i])
            plt.plot(points, approx_values, label= f'$p={degree}$')
        plt.legend()
        plt.ylim([-1.5, 2.0])
        # plt.show()
        plt.savefig("expansion_on_taylor.pdf")


class TestLagrange(unittest.TestCase):
    """Test the expansion.Lagrange class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = 10.0
        self._coordinate = LinearCoordinate(self._x_left, self._x_right)
        self._expansion = expansion.Lagrange(5, self._coordinate)

    def test_plot(self):
        """Plot the curves of a function and its approximations."""
        points = np.linspace(self._x_left, self._x_right, num=201)
        def my_function(point):
            return np.sin(point)
        exact_values = my_function(points)
        plt.figure()
        plt.plot(points, exact_values, 'o', label='Exact')
        for degree in range(8):
            my_expansion = expansion.Lagrange(degree, self._coordinate)
            my_expansion.approximate(my_function)
            approx_values = np.ndarray(len(points))
            for i in range(len(points)):
                approx_values[i] = my_expansion.get_function_value(points[i])
            plt.plot(points, approx_values, label= f'$p={degree}$')
        plt.legend()
        plt.ylim([-1.5, 2.0])
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

    def test_consistency_with_taylor(self):
        self._expansion.approximate(np.sin)
        points = np.linspace(self._x_left, self._x_right, num=201)
        for x in points:
            self.assertAlmostEqual(
                self._expansion.get_function_value(x),
                expansion.Taylor.get_function_value(self._expansion, x))
        self._expansion.set_coeff(np.random.rand(self._expansion.n_term()))
        for x in points:
            self.assertAlmostEqual(
                self._expansion.get_function_value(x),
                expansion.Taylor.get_function_value(self._expansion, x))


class TestLegendre(unittest.TestCase):
    """Test the Legendre class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = 10.0
        self._coordinate = LinearCoordinate(self._x_left, self._x_right)
        self._expansion = expansion.Legendre(5, self._coordinate)

    def test_plot(self):
        """Plot the curves of a function and its approximations."""
        points = np.linspace(self._x_left, self._x_right, num=201)
        def my_function(point):
            return np.sin(point)
        exact_values = my_function(points)
        plt.figure()
        plt.plot(points, exact_values, 'o', label='Exact')
        for degree in range(8):
            my_expansion = expansion.Legendre(degree, self._coordinate)
            my_expansion.approximate(my_function)
            approx_values = np.ndarray(len(points))
            for i in range(len(points)):
                approx_values[i] = my_expansion.get_function_value(points[i])
            plt.plot(points, approx_values, label= f'$p={degree}$')
        plt.legend()
        plt.ylim([-1.5, 2.0])
        # plt.show()
        plt.savefig("expansion_on_legendre.pdf")

    def test_get_average(self):
        """Test the method for getting average values.
        """
        functions = [np.sin, np.cos, np.arctan]
        for function in functions:
            self._expansion.approximate(function)
            integral, _ = integrate.quad(function, self._x_left, self._x_right)
            self.assertAlmostEqual(self._expansion.get_average(),
                integral/(self._x_right - self._x_left), places=3)

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
        inner_products = self._expansion.get_basis_innerproducts()
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
                  self.assertAlmostEqual(integral, inner_products[k][l])

    def test_consistency_with_taylor(self):
        self._expansion.approximate(np.sin)
        points = np.linspace(self._x_left, self._x_right, num=201)
        for x in points:
            self.assertAlmostEqual(
                self._expansion.get_function_value(x),
                expansion.Taylor.get_function_value(self._expansion, x))
        self._expansion.set_coeff(np.random.rand(self._expansion.n_term()))
        for x in points:
            self.assertAlmostEqual(
                self._expansion.get_function_value(x),
                expansion.Taylor.get_function_value(self._expansion, x))


class TestTruncatedLegendre(unittest.TestCase):
    """Test the TruncatedLegendre class.
    """

    def test_consistency(self):
        x_left, x_right = 0.0, 10
        coordinate = LinearCoordinate(x_left, x_right)
        points = np.linspace(x_left, x_right, num=201)
        p_high = 5
        legendre_high = expansion.Legendre(p_high, coordinate)
        u_init = np.sin
        legendre_high.approximate(u_init)
        for p_low in range(p_high+1):
            legendre_trunc = expansion.TruncatedLegendre(p_low, legendre_high)
            legendre_low = expansion.Legendre(p_low, coordinate)
            legendre_low.set_coeff(legendre_high.get_coeff_ref()[0:p_low+1])
            for x in points:
                self.assertAlmostEqual(legendre_low.get_function_value(x),
                    legendre_trunc.get_function_value(x))
                diff = (legendre_low.get_derivative_values(x)
                    - legendre_trunc.get_derivative_values(x))
                self.assertEqual(np.linalg.norm(diff), 0)
                diff = (legendre_low.get_basis_values(x)
                    - legendre_trunc.get_basis_values(x))
                self.assertEqual(np.linalg.norm(diff), 0)


if __name__ == '__main__':
    unittest.main()
