"""Test ODE systems from spatial schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import concept
import spatial
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
        self._spatial = spatial.LagrangeFR(
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
        self._a = 1.0
        self._equation = equation.LinearAdvection(self._a)
        self._riemann = riemann.LinearAdvection(self._a)
        self._tau = 0.0001
        self._n_element = 20
        delta_x = 100.0
        self._x_left = 0.0
        self._x_right = delta_x * self._n_element
        n_sample_per_element = 10
        n_sample = self._n_element * n_sample_per_element
        self._w = np.exp(1.0j * 2 * np.pi / n_sample)
        half_sample_gap = (delta_x / n_sample_per_element) / 2
        self._sample_points = np.linspace(self._x_left+half_sample_gap,
            self._x_right-half_sample_gap, n_sample)

    def get_kth_fourier_coeff(self, k, u):
        """Get the kth fourier coefficient of a scalar-valued function u(x).
        """
        n_sample = len(self._sample_points)
        kth_fourier_coeff = 0.0
        for j in range(n_sample):
            x_j = self._sample_points[j]
            u_j = u(x_j)
            kth_fourier_coeff += u_j * self._w**(-j*k)
        return kth_fourier_coeff / n_sample

    def get_wavenumbers(self, spatial_scheme: spatial.PiecewiseContinuous):
        """Get the reduced and modified wavenumbers of a PiecewiseContinuous scheme.
        """
        k_max = spatial_scheme.n_element() // 2
        k_max *= spatial_scheme.get_element(0.0).n_term()
        reduced_wavenumbers = np.ndarray(k_max)
        modified_wavenumbers = np.ndarray(k_max, complex)
        ode_solver = temporal.RungeKutta(3)
        for k in range(1, 1 + k_max):
            kappa = k * np.pi / (spatial_scheme.length() / 2)
            reduced_wavenumbers[k-1] = kappa * spatial_scheme.delta_x()
            def u_init(x):
                return np.exp(1.0j * kappa * x)
            kth_fourier_of_u_init = 0.0j
            kth_fourier_of_u_tau = 0.0j
            # solve the real part
            spatial_scheme.initialize(lambda x: u_init(x).real)
            kth_fourier_of_u_init += self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            ode_solver.update(spatial_scheme, delta_t=self._tau)
            kth_fourier_of_u_tau += self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            # solve the imag part
            spatial_scheme.initialize(lambda x: u_init(x).imag)
            kth_fourier_of_u_init += 1j * self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            ode_solver.update(spatial_scheme, delta_t=self._tau)
            kth_fourier_of_u_tau += 1j * self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            # put together
            modified_wavenumbers[k-1] = (1.0j * spatial_scheme.delta_x()
                / (self._a * self._tau)
                * np.log(kth_fourier_of_u_tau / kth_fourier_of_u_init))
        return reduced_wavenumbers, modified_wavenumbers

    def test_plot(self):
        spatials = [
            spatial.LagrangeFR(self._equation, self._riemann,
                0, self._n_element, self._x_left, self._x_right),
            spatial.LagrangeFR(self._equation, self._riemann,
                1, self._n_element, self._x_left, self._x_right),
            spatial.LagrangeFR(self._equation, self._riemann,
                2, self._n_element, self._x_left, self._x_right),
            spatial.LagrangeFR(self._equation, self._riemann,
                3, self._n_element, self._x_left, self._x_right),
        ]
        markers = ['1', '2', '3', '4']
        labels = ['FR1', 'FR2', 'FR3', 'FR4']
        kh_max = 4 * np.pi
        plt.subplot(2,1,1)
        plt.plot([0, kh_max, kh_max], [0, kh_max, 0], '--')
        plt.ylabel(r'$\Re(\Omega)$')
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\Omega)$')
        plt.xlabel(r'$\kappa h$')
        plt.plot([0, kh_max], [0, 0], '--')
        for i in range(len(labels)):
            reduced, modified = self.get_wavenumbers(spatials[i])
            # reduced /= (i + 2)
            # modified /= (i + 2)
            plt.subplot(2,1,1)
            plt.plot(reduced, modified.real, marker=markers[i], label=labels[i])
            plt.subplot(2,1,2)
            plt.plot(reduced, modified.imag, marker=markers[i], label=labels[i])
        plt.subplot(2,1,1)
        plt.legend()
        plt.subplot(2,1,2)
        plt.legend()
        plt.show()
        # plt.savefig('modified_wave_numbers.pdf')


if __name__ == '__main__':
    unittest.main()
