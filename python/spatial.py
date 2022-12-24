"""Concrete implementations of spatial schemes.
"""
import numpy as np

from concept import OdeSystem
import element


class FluxReconstruction(OdeSystem):
    """The ODE system given by the FR spatial discretization.
    """

    def __init__(self, x_min: float, x_max: float, n_element: int,
            degree: int, equation) -> None:
        super().__init__()
        assert x_min < x_max
        assert n_element > 1
        assert degree > 1
        self._n_element = n_element
        self._n_point_per_element = degree + 1
        self._elements = []
        delta_x = (x_max - x_min) / n_element
        x_left = x_min
        for i_element in range(n_element):
            assert x_left == x_min + i_element * delta_x
            x_right = x_left + delta_x
            self._elements.append(element.FluxReconstruction(
                  equation, degree, x_left, x_right))
            x_left = x_right
        assert x_left == x_max

    def n_dof(self):
        return self._n_element * self._n_point_per_element

    def set_unknown(self, column):
        assert len(column) == self.n_dof()
        first = 0
        for element in self._elements:
            last = first + self._n_point_per_element
            element.set_solution_coeff(column[first:last])
            first = last
        assert first == self.n_dof()

    def get_unknown(self):
        column = np.zeros(self.n_dof())
        first = 0
        for element in self._elements:
            last = first + self._n_point_per_element
            column[first:last] = element.get_solution_coeff()
            first = last
        assert first == self.n_dof()
        return column

    def get_residual(self):
        column = np.zeros(self.n_dof())
        return column


if __name__ == '__main__':
    pass
