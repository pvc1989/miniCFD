"""Test ODE systems from spatial schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import spatial
import equation
import riemann


class TestLagrangeFR(unittest.TestCase):
    """Test the element for implement flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._n_element = 5
        self._degree = 1
        self._a_const = np.pi
        self._spatial = spatial.LagrangeFR(
            equation.LinearAdvection(self._a_const),
            riemann.LinearAdvection(self._a_const),
            self._degree, self._n_element, self._x_left, self._x_right)

    def test_reconstruction(self):
        """Plot the curves of the approximate solution and the two fluxes.
        """
        kappa = 4 * np.pi / self._spatial.length()
        def u(x):
            return np.sin(kappa * x)
        self._spatial.initialize(u)
        n_point = 201
        points = np.linspace(self._x_left, self._x_right, n_point)
        exact_solution = np.ndarray(n_point)
        discontinuous_solution = np.ndarray(n_point)
        exact_flux = np.ndarray(n_point)
        discontinuous_flux = np.ndarray(n_point)
        continuous_flux = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            discontinuous_solution[i] = self._spatial.get_solution_value(
                point_i)
            exact_solution[i] = u(point_i)
            exact_flux[i] = self._spatial.equation.get_convective_flux(
                exact_solution[i])
            discontinuous_flux[i] = self._spatial.get_discontinuous_flux(
                point_i)
            continuous_flux[i] = self._spatial.get_continuous_flux(point_i)
        plt.figure()
        plt.subplot(2, 1, 1)
        points /= self._spatial.delta_x()
        plt.plot(points, exact_solution, 'r-', label='Exact')
        plt.plot(points, discontinuous_solution, 'b+', label='Discontinuous')
        plt.xlabel(r'$x/h$')
        plt.ylabel('Solution')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(points, exact_flux, 'r-', label='Exact')
        plt.plot(points, discontinuous_flux, 'b+', label='Discontinuous')
        plt.plot(points, continuous_flux, 'gx', label='Reconstructed')
        plt.xlabel(r'$x/h$')
        plt.ylabel('Flux')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig("flux_reconstruction.pdf")

    def test_resolution(self):
        degree = 4
        scheme = spatial.LagrangeFR(
            equation.LinearAdvection(self._a_const),
            riemann.LinearAdvection(self._a_const),
            degree, self._n_element, self._x_left, self._x_right)
        points = np.linspace(scheme.x_left(), scheme.x_right(), 201)
        x_data = points / scheme.delta_x()
        plt.figure(figsize=(6, degree*2))
        for k in range(1, 1+degree):
            plt.subplot(degree, 1, k)
            kappa = k * np.pi / scheme.delta_x()
            u_exact = lambda x: np.sin(kappa * (x - scheme.x_left()))
            y_data = u_exact(points)
            plt.plot(x_data, y_data, label='Exact')
            scheme.initialize(u_exact)
            for i in range(len(points)):
                y_data[i] = scheme.get_solution_value(points[i])
            plt.plot(x_data, y_data, '--', label=scheme.name())
            plt.xlabel(r'$x/h$')
            plt.ylabel(f'sin({k}'+r'$\pi x/h)$')
            plt.legend(loc='upper right')
            plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig("compare_resolutions.pdf")


if __name__ == '__main__':
    unittest.main()
