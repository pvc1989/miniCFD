"""Tests for various smoothness indicators.
"""
import unittest
import numpy as np

import equation
import riemann
import spatial
from smoothness import LiAndRen2011


class TestSmoothness(unittest.TestCase):
    """Test various smoothness indicators.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._n_element = 20
        self._equation = equation.LinearAdvection(1.0)
        self._riemann = riemann.LinearAdvection(1.0)

    def build_scheme(self, method: spatial.PiecewiseContinuous,
            degree: int) -> spatial.PiecewiseContinuous:
        scheme = method(self._equation, self._riemann,
            degree, self._n_element, self._x_left, self._x_right)
        return scheme

    def test_smoothness_values(self):
        degree = 4
        scheme = self.build_scheme(spatial.LagrangeFR, degree)
        scheme.initialize(lambda x: np.sin(x) * 10)
        indicaror = LiAndRen2011()
        smoothness = indicaror.get_smoothness_values(scheme)
        print(smoothness)


if __name__ == '__main__':
    unittest.main()
