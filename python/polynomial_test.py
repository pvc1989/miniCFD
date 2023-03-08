"""Tests for special polynomials.
"""
import unittest
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

from polynomial import Radau, Vincent, LagrangeBasis


class TestRadau(unittest.TestCase):
    """Test the Radau class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._degree = 5
        self._radau = Radau(self._degree)

    def test_values_at_ends(self):
        """Test values and -1 and +1.
        """
        self.assertEqual((0.0, 1.0), self._radau.get_function_value(-1.0))
        self.assertEqual((1.0, 0.0), self._radau.get_function_value(+1.0))

    def test_orthogonality(self):
        """Test Radau_{k} ⟂ Polynpmial_{k-2}.
        """
        for k in range(self._degree - 1):
            integral, _ = integrate.quad(lambda x: x**k * self._radau.get_function_value(x)[0],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)
            integral, _ = integrate.quad(lambda x: x**k * self._radau.get_function_value(x)[1],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)

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
        plt.plot(points, left_values, 'r--', label=r'$g_{+1}(\xi)$')
        plt.plot(points, right_values, 'b-', label=r'$g_{-1}(\xi)$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, left_derivatives, 'r--', label= r'$dg_{+1}/d\xi$')
        plt.plot(points, right_derivatives, 'b-', label= r'$dg_{-1}/d\xi$')
        plt.legend()
        # plt.show()
        plt.savefig("radau.pdf")


class TestVincent(unittest.TestCase):
    """Test the Vincent class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._degree = 4
        self._huyhn = Vincent(self._degree,
            lambda k: 2 * (k+1) / (2*k + 1) / k)

    def test_values_at_ends(self):
        """Test values and derivatives and -1 and +1.
        """
        left, right = self._huyhn.get_function_value(+1.0)
        self.assertAlmostEqual(0.0, left)
        self.assertAlmostEqual(1.0, right)
        left, right = self._huyhn.get_function_value(-1.0)
        self.assertAlmostEqual(0.0, right)
        self.assertAlmostEqual(1.0, left)
        left, right = self._huyhn.get_function_value(+1.0)
        self.assertAlmostEqual(0.0, left)
        left, right = self._huyhn.get_function_value(-1.0)
        self.assertAlmostEqual(0.0, right)

    def test_orthogonality(self):
        """Test Huyhn_{k} ≡ g_{k+1} ⟂ Polynpmial_{k-2}.
        """
        huyhn = Vincent(self._degree, lambda k: 2 * (k+1) / (2*k + 1) / k)
        for k in range(self._degree - 1):
            integral, _ = integrate.quad(lambda x: x**k * huyhn.get_function_value(x)[0],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)
            integral, _ = integrate.quad(lambda x: x**k * huyhn.get_function_value(x)[1],
                -1.0, 1.0)
            self.assertAlmostEqual(0.0, integral)

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
            left_values[i], right_values[i] = self._huyhn.get_function_value(point_i)
            left_derivatives[i], right_derivatives[i] = self._huyhn.get_gradient_value(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(points, left_values, 'r--', label=r'$g_{-1}(\xi)$')
        plt.plot(points, right_values, 'b-', label=r'$g_{+1}(\xi)$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, left_derivatives, 'r--', label= r'$dg_{-1}/d\xi$')
        plt.plot(points, right_derivatives, 'b-', label= r'$dg_{+1}/d\xi$')
        plt.legend()
        # plt.show()
        plt.savefig("huyhn.pdf")


class TestLagrangeBasis(unittest.TestCase):
    """Test the LagrangeBasis class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._n_point = 5
        points = np.linspace(-1.0, 1.0, self._n_point)
        self._lagrange = LagrangeBasis(points)

    def test_plot(self):
        """Plot the curves of each Lagrange polynomial.
        """
        n_point = 101
        points = np.linspace(-1.0, 1.0, n_point)
        values = np.zeros((n_point, self._n_point))
        for i in range(n_point):
            point_i = points[i]
            values[i] = self._lagrange.get_function_value(point_i)
        plt.figure()
        for i in range(self._n_point):
            plt.plot(points, values[:, i], label=f'$L_{i}$')
        plt.legend()
        # plt.show()
        plt.savefig("lagrange.pdf")

    def test_delta_property(self):
        """Test the delta_property, i.e. L_i(x_j) = delta_{ij}.
        """
        points = np.linspace(-1.0, 1.0, self._n_point)
        for i in range(self._n_point):
            values = self._lagrange.get_function_value(points[i])
            for j in range(self._n_point):
                if i == j:
                    self.assertEqual(1.0, values[j])
                else:
                    self.assertEqual(0.0, values[j])


if __name__ == '__main__':
    unittest.main()
