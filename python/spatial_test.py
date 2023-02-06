"""Test ODE systems from spatial schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from spatial import LagrangeFR, LagrangeDG, DGwithLagrangeFR
import equation
import riemann

class TestLagrangeFR(unittest.TestCase):
    """Test the element for implement flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._n_element = 3
        self._degree = 1
        a_const = np.pi
        self._spatial = LagrangeFR(
            equation.LinearAdvection(a_const),
            riemann.LinearAdvection(a_const),
            self._degree, self._n_element, self._x_left, self._x_right)

    def test_plot(self):
        """Plot the curves of the approximate solution and the two fluxes.
        """
        function = np.sin
        self._spatial.initialize(function)
        n_point = 101
        points = np.linspace(self._x_left, self._x_right, n_point)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        discontinuous_flux = np.ndarray(n_point)
        continuous_flux = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = function(point_i)
            discontinuous_flux[i] = self._spatial.get_discontinuous_flux(point_i)
            continuous_flux[i] = self._spatial.get_continuous_flux(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(points, approx_solution, 'b.', label='Approximate Solution')
        plt.plot(points, expect_solution, 'r--', label='Exact Solution')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, discontinuous_flux, 'b.', label='Discontinuous Flux')
        plt.plot(points, continuous_flux, 'r--', label='Continuous Flux')
        plt.legend()
        # plt.show()
        plt.savefig("fr_approx.pdf")


class PlotModifiedWavenumbers(unittest.TestCase):
    """Plot modified-wavenumbers for various spatial schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._spatial = LagrangeFR(
            equation.LinearAdvection(1.0),
            riemann.LinearAdvection(1.0),
            degree=5, n_element=50,
            x_left=0.0, x_right=10.0, value_type=complex)
        self._n_point = self._spatial.n_element() + 1

    def test_plot(self):
        """.
        """
        l = self._spatial.length() / 2
        i = 1j
        n = self._n_point - 1
        w = np.exp(i * 2 * np.pi / n)
        h = self._spatial.length() / n
        kappa_h = np.ndarray(self._n_point // 2)
        kappa_tilde_h = np.ndarray(self._n_point // 2, complex)
        for k in range(1, 1 + self._n_point // 2):
            kappa = k * np.pi / l
            kappa_h[k-1] = kappa * h
            def u_init(x):
                return np.exp(i * kappa * x)
            self._spatial.initialize(u_init)
            gradients = np.ndarray(self._spatial.n_element(), complex)
            v_k = 0.0
            for r in range(self._spatial.n_element()):
                x_r = self._spatial.x_left() + r * h + h * 0.5
                # finite-difference approximation
                dx = 0.0001
                gradients[r] = (self._spatial.get_continuous_flux(x_r+dx)
                    - self._spatial.get_continuous_flux(x_r-dx)) / (2*dx)
                v_k += gradients[r] * w**(-r * k)
            kappa_tilde_h[k-1] = v_k * h / (i * n)
        plt.subplot(2,1,1)
        plt.plot(kappa_h, kappa_tilde_h.real, 'b+')
        plt.plot([0, np.pi], [0, np.pi], 'r-')
        plt.subplot(2,1,2)
        plt.plot(kappa_h, kappa_tilde_h.imag, 'b+')
        plt.plot([0, np.pi], [0, 0], 'r-')
        # plt.show()
        plt.savefig('modified_wave_numbers.pdf')


if __name__ == '__main__':
    unittest.main()
