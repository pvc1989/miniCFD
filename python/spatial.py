"""Concrete implementations of spatial schemes.
"""
import numpy as np

from concept import OdeSystem
import element


class FluxReconstruction(OdeSystem):
    """The ODE system given by the FR spatial discretization.
    """

    def __init__(self, equation, riemann, degree: int,
            n_element: int, x_min: float, x_max: float) -> None:
        super().__init__()
        assert x_min < x_max
        assert n_element > 1
        assert degree >= 0
        self._equation = equation
        self._riemann = riemann
        self._n_element = n_element
        self._n_point_per_element = degree + 1
        self._elements = []
        delta_x = (x_max - x_min) / n_element
        self._delta_x = delta_x
        x_left = x_min
        for i_element in range(n_element):
            assert x_left == x_min + i_element * delta_x
            x_right = x_left + delta_x
            self._elements.append(element.FluxReconstruction(
                  equation, degree, x_left, x_right))
            x_left = x_right
        assert x_left == x_max

    def x_left(self):
        return self._elements[0].x_left()

    def x_right(self):
        return self._elements[-1].x_right()

    def length(self):
        return self.x_right() - self.x_left()

    def n_element(self):
        return self._n_element

    def n_dof(self):
        return self._n_element * self._n_point_per_element

    def set_solution_column(self, column):
        assert len(column) == self.n_dof()
        first = 0
        for element in self._elements:
            last = first + self._n_point_per_element
            element.set_solution_coeff(column[first:last])
            first = last
        assert first == self.n_dof()

    def get_solution_column(self):
        column = np.zeros(self.n_dof())
        first = 0
        for element in self._elements:
            last = first + self._n_point_per_element
            column[first:last] = element.get_solution_column()
            first = last
        assert first == self.n_dof()
        return column

    def get_residual_column(self):
        column = np.zeros(self.n_dof())
        interface_flux = np.ndarray(self._n_element + 1)
        # interface_flux[i] := flux on interface(element[i-1], element[i])
        for i in range(1, self._n_element):
            x_right = self._elements[i].x_left()
            u_right= self._elements[i].get_solution_value(x_right)
            x_left = self._elements[i-1].x_right()
            u_left = self._elements[i-1].get_solution_value(x_left)
            interface_flux[i] = self._riemann.F_upwind(u_left, u_right)
        # periodic boundary condtion
        x_right = self._elements[0].x_left()
        u_right = self._elements[0].get_solution_value(x_right)
        x_left = self._elements[-1].x_right()
        u_left = self._elements[-1].get_solution_value(x_left)
        interface_flux[0] = self._riemann.F_upwind(u_left, u_right)
        interface_flux[-1] = interface_flux[0]
        # evaluate flux gradients
        i_dof = 0
        for i in range(self._n_element):
            element = self._elements[i]
            upwind_flux_left = interface_flux[i]
            upwind_flux_right = interface_flux[i+1]
            values = -element.get_flux_gradients(
                upwind_flux_left, upwind_flux_right)
            column[i_dof:i_dof+len(values)] = values
            i_dof += len(values)
        assert i_dof == self.n_dof()
        return column

    def get_element_index(self, point):
        i_element = int((point - self.x_left()) / self._delta_x)
        if i_element == self._n_element:
            i_element -= 1
        return i_element

    def get_element(self, point):
        return self._elements[self.get_element_index(point)]

    def get_solution_value(self, point):
        element = self.get_element(point)
        return element.get_solution_value(point)

    def get_discontinuous_flux(self, point):
        element = self.get_element(point)
        return element.get_discontinuous_flux(point)

    def get_continuous_flux(self, point):
        curr = self.get_element_index(point)
        # solve riemann problem at the left end of curr element
        x_right = self._elements[curr].x_left()
        u_right = self._elements[curr].get_solution_value(x_right)
        prev = curr - 1
        x_left = self._elements[prev].x_right()
        u_left = self._elements[prev].get_solution_value(x_left)
        upwind_flux_left = self._riemann.F_upwind(u_left, u_right)
        # solve riemann problem at the right end of curr element
        x_left = self._elements[curr].x_right()
        u_left = self._elements[curr].get_solution_value(x_left)
        next = (curr + 1) % self._n_element
        x_right = self._elements[next].x_left()
        u_right = self._elements[next].get_solution_value(x_right)
        upwind_flux_right = self._riemann.F_upwind(u_left, u_right)
        return self._elements[curr].get_continuous_flux(point,
            upwind_flux_left, upwind_flux_right)

    def initialize(self, function: callable):
        for element in self._elements:
            element.approximate(function)


if __name__ == '__main__':
    pass
