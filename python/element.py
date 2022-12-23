"""Implement some element for spatial discretization.
"""
import numpy as np

from interpolation import Lagrange
from polynomial import Radau


class FluxReconstruction(object):
    """Element for implement flux reconstruction schemes.
    """

    def __init__(self, equation, degree: int, x_min, x_max) -> None:
        self._equation = equation
        self._degree = degree
        self._solution_points = np.linspace(x_min, x_max, degree + 1)
        self._lagrange = Lagrange(self._solution_points)
        self._radau = Radau(degree + 1)

    def approximate(self, function):
        """
        """

    def get_unknown_values(self, point):
        """
        """
        return 0.0

    def get_discontinuous_flux(self, point):
        """
        """
        return 0.0

    def get_continuous_flux(self, point, upwind_flux_left, upwind_flux_right):
        """
        """
        return 0.0

    def get_flux_gradient(self, point, upwind_flux_left, upwind_flux_right):
        """
        """
        return 0.0


if __name__ == '__main__':
    pass
