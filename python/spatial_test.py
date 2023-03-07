"""Test ODE systems from spatial schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

from spatial import LagrangeFR, LagrangeDG, DGwithLagrangeFR
import equation
import riemann
import temporal

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
        self._a_const = 1.0
        self._spatial = LagrangeFR(
            equation.LinearAdvection(self._a_const),
            riemann.LinearAdvection(self._a_const),
            degree=2, n_element=10,
            x_left=0.0, x_right=10.0, value_type=float)

    def test_plot(self):
        """.
        """
        a = self._a_const
        tau = 0.0001
        l = self._spatial.length() / 2
        i = 1.0j
        npe = self._spatial.degree() + 1
        n = self._spatial.n_element() * npe
        w = np.exp(i * 2 * np.pi / n)
        h = self._spatial.length() / self._spatial.n_element()
        eps = h / 2 / npe
        points = np.linspace(eps, self._spatial.length()-eps, n)
        assert len(points) == n
        kappa_h = np.ndarray(n)
        kappa_tilde_h = np.ndarray(n, complex)
        ode_solver = temporal.RungeKutta(3)
        for k in range(1, 1 + n):
            kappa = k * np.pi / l
            kappa_h[k-1] = kappa * h
            def u_init(x):
                return np.exp(i * kappa * x)
            # DFT for u_init
            u_k_0 = 0.0 + 0.0j
            for j in range(n):
                x_j = points[j]
                u_k_0 += u_init(x_j) * w**(-j*k)
            # solve the real part
            u_k_tau_re = 0.0
            self._spatial.initialize(lambda x: u_init(x).real)
            ode_solver.update(self._spatial, delta_t=tau)
            for j in range(n):
                x_j = points[j]
                u_k_tau_re += self._spatial.get_solution_value(x_j) * w**(-j*k)
            # solve the imag part
            u_k_tau_im = 0.0
            self._spatial.initialize(lambda x: u_init(x).imag)
            ode_solver.update(self._spatial, delta_t=tau)
            for j in range(n):
                x_j = points[j]
                u_k_tau_im += self._spatial.get_solution_value(x_j) * w**(-j*k)
            # put together
            u_k_tau = u_k_tau_re + i * u_k_tau_im
            kappa_tilde_h[k-1] = i * h / (a * tau) * np.log(u_k_tau / u_k_0)
        plt.subplot(2,1,1)
        plt.plot(kappa_h, kappa_tilde_h.real, 'b+')
        plt.plot([0, np.pi], [0, np.pi], 'r-')
        plt.subplot(2,1,2)
        plt.plot(kappa_h, kappa_tilde_h.imag, 'b+')
        plt.plot([0, np.pi], [0, 0], 'r-')
        plt.show()
        # plt.savefig('modified_wave_numbers.pdf')


if __name__ == '__main__':
    unittest.main()
