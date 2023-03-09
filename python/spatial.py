"""Concrete implementations of spatial schemes.
"""
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import integrate

from concept import SpatialScheme, Equation, RiemannSolver
import element


class PiecewiseContinuous(SpatialScheme):
    """The base of all piecewise continuous schemes.
    """

    def __init__(self, equation: Equation, riemann: RiemannSolver,
            n_element: int, x_left: float, x_right: float) -> None:
        super().__init__(n_element, x_left, x_right)
        self._equation = equation
        self._riemann = riemann

    def n_dof(self):
        return self.n_element() * self._elements[0].n_dof()

    def get_element_index(self, point):
        i_element = int((point - self.x_left()) / self.delta_x())
        if i_element == self._n_element:
            i_element -= 1
        return i_element

    def get_element(self, point):
        return self._elements[self.get_element_index(point)]

    def get_interface_fluxes(self):
        """Get the interface flux at each element interface.
        """
        interface_fluxes = np.ndarray(self._n_element + 1)
        # interface_flux[i] := flux on interface(element[i-1], element[i])
        for i in range(1, self._n_element):
            x_right = self._elements[i].x_left()
            u_right= self._elements[i].get_solution_value(x_right)
            x_left = self._elements[i-1].x_right()
            u_left = self._elements[i-1].get_solution_value(x_left)
            interface_fluxes[i] = self._riemann.get_upwind_flux(u_left, u_right)
        # periodic boundary condtion
        x_right = self._elements[0].x_left()
        u_right = self._elements[0].get_solution_value(x_right)
        x_left = self._elements[-1].x_right()
        u_left = self._elements[-1].get_solution_value(x_left)
        interface_fluxes[0] = self._riemann.get_upwind_flux(u_left, u_right)
        interface_fluxes[-1] = interface_fluxes[0]
        return interface_fluxes

    def get_solution_value(self, point):
        element = self.get_element(point)
        return element.get_solution_value(point)

    def get_discontinuous_flux(self, point):
        element = self.get_element(point)
        return element.get_discontinuous_flux(point)

    def set_solution_column(self, column):
        assert len(column) == self.n_dof()
        first = 0
        for element in self._elements:
            last = first + element.n_dof()
            element.set_solution_coeff(column[first:last])
            first = last
        assert first == self.n_dof()

    def get_solution_column(self):
        column = np.zeros(self.n_dof())
        first = 0
        for element in self._elements:
            last = first + element.n_dof()
            column[first:last] = element.get_solution_column()
            first = last
        assert first == self.n_dof()
        return column

    def initialize(self, function: callable):
        for element in self._elements:
            element.approximate(function)


class LegendreDG(PiecewiseContinuous):
    """The ODE system given by the DG method using a Legendre expansion.
    """

    def __init__(self, equation: Equation, riemann: RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__(equation, riemann, n_element, x_left, x_right)
        x_left_i = x_left
        for i_element in range(n_element):
            assert_almost_equal(x_left_i, x_left + i_element * self.delta_x())
            x_right_i = x_left_i + self.delta_x()
            element_i = element.LegendreDG(
                  equation, degree, x_left_i, x_right_i, value_type)
            self._elements[i_element] = element_i
            x_left_i = x_right_i
        assert_almost_equal(x_left_i, x_right)

    @staticmethod
    def name():
        return 'LegendreDG'

    def get_residual_column(self):
        column = np.zeros(self.n_dof())
        interface_fluxes = self.get_interface_fluxes()
        i_dof = 0
        for i in range(self._n_element):
            element = self._elements[i]
            n_dof = element.n_dof()
            # build element_i's residual column
            # 1st: evaluate the internal integral
            def integrand(points):
                n_row = n_dof
                n_col = len(points)
                values = np.ndarray((n_row, n_col))
                for c in range(n_col):
                    column = element.get_basis_gradients(points[c])
                    gradient = element.get_discontinuous_flux(points[c])
                    values[:,c] = +column * gradient
                return values
            values, _ = integrate.fixed_quad(integrand,
                element.x_left(), element.x_right(), n=max(1, element.degree()))
            # 2nd: evaluate the boundary integral
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            values += upwind_flux_left * element.get_basis_values(element.x_left())
            values -= upwind_flux_right * element.get_basis_values(element.x_right())
            # 3rd: multiply the inverse of the mass matrix
            # TODO: factor out the only difference between various DG methods.
            values = element.divide_mass_matrix(values)
            # write to the global column
            column[i_dof:i_dof+n_dof] = values
            i_dof += n_dof
        assert i_dof == self.n_dof()
        return column


class LagrangeDG(PiecewiseContinuous):
    """The ODE system given by the DG method using a Lagrange expansion.
    """

    def __init__(self, equation: Equation, riemann: RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__(equation, riemann, n_element, x_left, x_right)
        self._local_mass_matrices = np.ndarray(n_element, np.ndarray)
        x_left_i = x_left
        for i_element in range(n_element):
            assert_almost_equal(x_left_i, x_left + i_element * self.delta_x())
            x_right_i = x_left_i + self.delta_x()
            element_i = element.LagrangeDG(
                  equation, degree, x_left_i, x_right_i, value_type)
            self._elements[i_element] = element_i
            self._local_mass_matrices[i_element] = element_i.build_mass_matrix()
            x_left_i = x_right_i
        assert_almost_equal(x_left_i, x_right)

    @staticmethod
    def name():
        return 'LagrangeDG'

    def get_residual_column(self):
        column = np.zeros(self.n_dof())
        interface_fluxes = self.get_interface_fluxes()
        i_dof = 0
        for i in range(self._n_element):
            element = self._elements[i]
            n_dof = element.n_dof()
            # build element_i's residual column
            # 1st: evaluate the internal integral
            def integrand(points):
                n_row = n_dof
                n_col = len(points)
                values = np.ndarray((n_row, n_col))
                for c in range(n_col):
                    column = element.get_basis_gradients(points[c])
                    gradient = element.get_discontinuous_flux(points[c])
                    values[:,c] = +column * gradient
                return values
            values, _ = integrate.fixed_quad(integrand,
                element.x_left(), element.x_right(), n=max(1, element.degree()))
            # 2nd: evaluate the boundary integral
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            values += upwind_flux_left * element.get_basis_values(element.x_left())
            values -= upwind_flux_right * element.get_basis_values(element.x_right())
            # 3rd: multiply the inverse of the mass matrix
            values = np.linalg.solve(self._local_mass_matrices[i], values)
            # write to the global column
            column[i_dof:i_dof+n_dof] = values
            i_dof += n_dof
        assert i_dof == self.n_dof()
        return column


class LagrangeFR(PiecewiseContinuous):
    """The ODE system given by Huyhn's FR method.
    """

    def __init__(self, equation: Equation, riemann: RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__(equation, riemann, n_element, x_left, x_right)
        assert degree >= 0
        x_left_i = x_left
        for i_element in range(n_element):
            assert_almost_equal(x_left_i, x_left + i_element * self.delta_x())
            x_right_i = x_left_i + self.delta_x()
            self._elements[i_element] = element.LagrangeFR(
                  equation, degree, x_left_i, x_right_i, value_type)
            x_left_i = x_right_i
        assert_almost_equal(x_left_i, x_right)

    @staticmethod
    def name():
        return 'LagrangeFR'

    def get_residual_column(self):
        column = np.zeros(self.n_dof())
        interface_fluxes = self.get_interface_fluxes()
        # evaluate flux gradients
        i_dof = 0
        for i in range(self._n_element):
            element = self._elements[i]
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            values = -element.get_flux_gradients(
                upwind_flux_left, upwind_flux_right)
            column[i_dof:i_dof+len(values)] = values
            i_dof += len(values)
        assert i_dof == self.n_dof()
        return column

    def get_continuous_flux(self, point):
        curr = self.get_element_index(point)
        # solve riemann problem at the left end of curr element
        x_right = self._elements[curr].x_left()
        u_right = self._elements[curr].get_solution_value(x_right)
        prev = curr - 1
        x_left = self._elements[prev].x_right()
        u_left = self._elements[prev].get_solution_value(x_left)
        upwind_flux_left = self._riemann.get_upwind_flux(u_left, u_right)
        # solve riemann problem at the right end of curr element
        x_left = self._elements[curr].x_right()
        u_left = self._elements[curr].get_solution_value(x_left)
        next = (curr + 1) % self._n_element
        x_right = self._elements[next].x_left()
        u_right = self._elements[next].get_solution_value(x_right)
        upwind_flux_right = self._riemann.get_upwind_flux(u_left, u_right)
        return self._elements[curr].get_continuous_flux(point,
            upwind_flux_left, upwind_flux_right)


class DGwithLagrangeFR(LagrangeFR):
    """A DG scheme built upon LagrangeFR.
    """

    def __init__(self, equation: Equation, riemann: RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__(equation, riemann, degree, n_element, x_left, x_right,
            value_type)
        # Most work has been delegated to LagrangeFR.__init__(),
        # but mass matrices should be built here:
        self._local_mass_matrices = np.ndarray(n_element, np.ndarray)
        for i in range(n_element):
            self._local_mass_matrices[i] = self._elements[i].build_mass_matrix()

    @staticmethod
    def name():
        return 'DGFR'

    def get_residual_column(self):
        column = np.zeros(self.n_dof())
        interface_fluxes = self.get_interface_fluxes()
        i_dof = 0
        for i in range(self._n_element):
            element = self._elements[i]
            n_dof = element.n_dof()
            # build element_i's residual column
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            def integrand(points):
                n_row = n_dof
                n_col = len(points)
                values = np.ndarray((n_row, n_col))
                for c in range(n_col):
                    column = element.get_basis_values(points[c])
                    gradient = element.get_flux_gradient(points[c],
                        upwind_flux_left, upwind_flux_right)
                    values[:,c] = -column * gradient
                return values
            values, _ = integrate.fixed_quad(integrand,
                element.x_left(), element.x_right(), n=element.degree()+1)
            values = np.linalg.solve(self._local_mass_matrices[i], values)
            column[i_dof:i_dof+n_dof] = values
            i_dof += n_dof
        assert i_dof == self.n_dof()
        return column


if __name__ == '__main__':
    pass
