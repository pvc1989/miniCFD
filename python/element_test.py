import unittest
import numpy as np
from matplotlib import pyplot as plt

from element import FluxReconstruction
from equation import LinearAdvection
from polynomial import Radau
from interpolation import Lagrange


class TestFluxReconstruction(unittest.TestCase):
    """Test the element for implement flux reconstruction schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._equation = LinearAdvection(a_const=np.random.rand())
        self._degree = 4
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._test_points = np.linspace(self._x_left, self._x_right)
        self._element = FluxReconstruction(self._equation, self._degree, self._x_left, self._x_right)
        self._element.approximate(np.sin)
        self._radau = Radau(self._degree + 1)
        self._lagrange = Lagrange(np.linspace(self._x_left, self._x_right, self._degree + 1))

    def test_plot(self):
        """Plot the curves of F^{discontinous} and F^{continous}.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        n_point = 201
        points = np.linspace(self._x_left, self._x_right, n_point)
        discontinuous_flux = np.zeros(n_point)
        continuous_flux = np.zeros(n_point)
        for i in range(n_point):
            point_i = points[i]
            discontinuous_flux[i] = self._element.get_discontinuous_flux(point_i)
            continuous_flux[i] = self._element.get_continuous_flux(point_i, upwind_flux_left, upwind_flux_right)
        plt.figure()
        plt.plot(points, discontinuous_flux, 'r--', label='Discontinuous FLux')
        plt.plot(points, continuous_flux, 'b-', label='Continuous FLux')
        plt.plot(self._x_left, upwind_flux_left, 'k<', label='Upwind Flux At Left')
        plt.plot(self._x_right, upwind_flux_right, 'k>', label='Upwind Flux At Right')
        plt.legend()
        plt.show()
        plt.savefig("fr_by_radau.pdf")

    def test_get_discontinuous_flux(self):
        """Test the values of the discontinuous flux.
        """
        for x_global in self._test_points:
            flux_actual = self._element.get_discontinuous_flux(x_global)
            # discontinuous_flux = a * u
            flux_expect = self._equation.F(self._element.get_solution_value(x_global))
            self.assertAlmostEqual(flux_expect, flux_actual)

    def test_get_continuous_flux(self):
        """Test the values of the reconstructed continuous flux.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        self.assertAlmostEqual(upwind_flux_left,
            self._element.get_continuous_flux(self._x_left, upwind_flux_left, upwind_flux_right))
        self.assertAlmostEqual(upwind_flux_right,
            self._element.get_continuous_flux(self._x_right, upwind_flux_left, upwind_flux_right))
        for x_global in self._test_points:
            flux_actual = self._element.get_continuous_flux(x_global, upwind_flux_left, upwind_flux_right)
            # continuous_flux = discontinuous_flux + correction
            flux_expect = self._element.get_discontinuous_flux(x_global)
            x_local = self._lagrange.global_to_local(x_global)
            radau_left, radau_right = self._radau.get_function_value(x_local)
            flux_expect += radau_right * (upwind_flux_left
                - self._element.get_discontinuous_flux(self._x_left))
            flux_expect += radau_left * (upwind_flux_right
                - self._element.get_discontinuous_flux(self._x_right))
            self.assertAlmostEqual(flux_expect, flux_actual)

    def test_get_flux_gradient(self):
        """Test the gradient of the reconstructed continuous flux.
        """
        upwind_flux_left = np.random.rand()
        upwind_flux_right = np.random.rand()
        points = np.linspace(self._x_left, self._x_right)
        for x_global in points:
            gradient_actual = self._element.get_flux_gradient(x_global, upwind_flux_left, upwind_flux_right)
            # 2nd-order finite difference approximation
            delta_x = 0.0001
            gradient_approx = (
                self._element.get_continuous_flux(x_global + delta_x, upwind_flux_left, upwind_flux_right) -
                self._element.get_continuous_flux(x_global - delta_x, upwind_flux_left, upwind_flux_right)) / (delta_x + delta_x)
            self.assertAlmostEqual(gradient_actual, gradient_approx)


if __name__ == '__main__':
    unittest.main()
