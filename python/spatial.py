"""Concrete implementations of spatial schemes.
"""
import numpy as np
from numpy.testing import assert_almost_equal

import concept
import expansion
import element
import integrate


class FiniteElement(concept.SpatialScheme):
    """The base of all finite element schemes for conservation laws.
    """

    def __init__(self, equation: concept.Equation,
            riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            element_type: concept.Element, value_type=float) -> None:
        concept.SpatialScheme.__init__(self, equation,
            n_element, x_left, x_right)
        assert degree >= 0
        self._riemann = riemann
        x_left_i = x_left
        for i_element in range(n_element):
            assert_almost_equal(x_left_i, x_left + i_element * self.delta_x())
            x_right_i = x_left_i + self.delta_x()
            element_i = element_type(
                equation, degree, x_left_i, x_right_i,
                concept.LinearCoordinateMap(x_left, x_right), value_type)
            self._elements[i_element] = element_i
            x_left_i = x_right_i
        assert_almost_equal(x_left_i, x_right)
        self._value_type = value_type

    def n_dof(self):
        return self.n_element() * self.get_element_by_index(0).n_dof()

    def get_element_index(self, point):
        i_element = int((point - self.x_left()) / self.delta_x())
        if i_element == self._n_element:
            i_element -= 1
        return i_element

    def _get_interface_flux(self, cell_left: concept.Element,
            cell_right: concept.Element, extra_viscous: float):
        # Solve the local Riemann problem:
        x_left = cell_left.x_right()
        u_left = cell_left.get_solution_value(x_left)
        x_right = cell_right.x_left()
        u_right = cell_right.get_solution_value(x_right)
        flux = self._riemann.get_upwind_flux(u_left, u_right)
        # Add the diffusive interface flux:
        expansion_left = cell_left.get_expansion()
        assert isinstance(expansion_left, expansion.Taylor)
        derivatives = expansion_left.get_derivative_values(x_left)
        if self.degree() > 1:
            du_left, ddu_left = derivatives[1], derivatives[2]
        elif self.degree() == 1:
            du_left, ddu_left = derivatives[1], 0
        else:
            du_left, ddu_left = 0, 0
        expansion_right = cell_right.get_expansion()
        assert isinstance(expansion_right, expansion.Taylor)
        derivatives = expansion_right.get_derivative_values(x_right)
        if self.degree() > 1:
            du_right, ddu_right = derivatives[1], derivatives[2]
        elif self.degree() == 1:
            du_right, ddu_right = derivatives[1], 0
        else:
            du_right, ddu_right = 0, 0
        # Use the DDG method to get the value of ∂u/∂x at interface:
        du = (du_left + du_right) / 2
        distance = (cell_left.length() + cell_right.length()) / 2
        beta_0 = 3
        du += beta_0 / distance * (u_right - u_left)
        beta_1 = 1/12
        du += beta_1 * distance * (ddu_right - ddu_left)
        viscous = extra_viscous + self.equation().get_diffusive_coeff()
        return flux - viscous * du

    def get_interface_fluxes(self):
        """Get the interface flux at each element interface.
        """
        interface_fluxes = np.ndarray(self._n_element + 1, self._value_type)
        # interface_flux[i] := flux on interface(element[i-1], element[i])
        for i in range(1, self._n_element):
            curr, prev = self._elements[i], self._elements[i-1]
            viscous = max(self._viscous.get_coeff(i),
                self._viscous.get_coeff(i-1))
            interface_fluxes[i] = self._get_interface_flux(prev, curr, viscous)
        if self.is_periodic():
            curr, prev = self._elements[0], self._elements[-1]
            viscous = max(self._viscous.get_coeff(0),
                self._viscous.get_coeff(-1))
            interface_fluxes[0] = self._get_interface_flux(prev, curr, viscous)
            interface_fluxes[-1] = interface_fluxes[0]
        else:  # TODO: support other boundary condtions
            assert False
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
            assert isinstance(element, concept.Element)
            last = first + element.n_dof()
            element.set_solution_coeff(column[first:last])
            first = last
        assert first == self.n_dof()
        self._detect_and_limit()

    def get_solution_column(self):
        column = np.zeros(self.n_dof(), self._value_type)
        first = 0
        for element in self._elements:
            assert isinstance(element, concept.Element)
            last = first + element.n_dof()
            column[first:last] = element.get_solution_column()
            first = last
        assert first == self.n_dof()
        return column

    def initialize(self, function: callable):
        for element in self._elements:
            assert isinstance(element, concept.Element)
            element.approximate(function)
        self._detect_and_limit()


class DiscontinuousGalerkin(FiniteElement):
    """An mid-level class that defines common methods for all DG schemes.
    """

    def get_residual_column(self):
        column = np.zeros(self.n_dof(), self._value_type)
        interface_fluxes = self.get_interface_fluxes()
        i_dof = 0
        for i in range(self._n_element):
            element = self.get_element_by_index(i)
            n_dof = element.n_dof()
            # build element_i's residual column
            # 1st: evaluate the internal integral
            extra_viscous = self._viscous.get_coeff(i)
            def integrand(x_global):
                column = element.get_basis_gradients(x_global)
                gradient = element.get_discontinuous_flux(x_global, extra_viscous)
                return column * gradient
            values = integrate.fixed_quad_global(integrand,
                element.x_left(), element.x_right(), max(1, element.degree()))
            # 2nd: evaluate the boundary integral
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            values += upwind_flux_left * element.get_basis_values(element.x_left())
            values -= upwind_flux_right * element.get_basis_values(element.x_right())
            # 3rd: multiply the inverse of the mass matrix
            values = element.divide_mass_matrix(values)
            # write to the global column
            column[i_dof:i_dof+n_dof] = values
            i_dof += n_dof
        assert i_dof == self.n_dof()
        return column


class LegendreDG(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Legendre expansion.
    """

    def __init__(self, equation: concept.Equation,
            riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        FiniteElement.__init__(self, equation, riemann, degree,
            n_element, x_left, x_right, element.LegendreDG, value_type)

    def name(self, verbose=True):
        my_name = 'LegendreDG'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LagrangeDG(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Lagrange expansion.
    """

    def __init__(self, equation: concept.Equation,
            riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        DiscontinuousGalerkin.__init__(self, equation, riemann, degree,
            n_element, x_left, x_right, element.LagrangeDG, value_type)

    def name(self, verbose=True):
        my_name = 'LagrangeDG'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class FluxReconstruction(FiniteElement):
    """An mid-level class that defines common methods for all FR schemes.
    """

    def get_residual_column(self):
        column = np.zeros(self.n_dof(), self._value_type)
        interface_fluxes = self.get_interface_fluxes()
        # evaluate flux gradients
        i_dof = 0
        for i in range(self._n_element):
            element = self.get_element_by_index(i)
            extra_viscous = self._viscous.get_coeff(i)
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            values = -element.get_flux_gradients(extra_viscous,
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


class LagrangeFR(FluxReconstruction):
    """The ODE system given by Huyhn's FR method.
    """

    def __init__(self, equation: concept.Equation,
            riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        FluxReconstruction.__init__(self, equation, riemann, degree,
            n_element, x_left, x_right, element.LagrangeFR, value_type)

    def name(self, verbose=True):
        my_name = 'LagrangeFR'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LegendreFR(FluxReconstruction):
    """The ODE system given by Huyhn's FR method.
    """

    def __init__(self, equation: concept.Equation,
            riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        FluxReconstruction.__init__(self, equation, riemann, degree,
            n_element, x_left, x_right, element.LegendreFR, value_type)

    def name(self, verbose=True):
        my_name = 'LegendreFR'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class DGwithFR(LagrangeFR):
    """A DG scheme built upon LagrangeFR.
    """

    def __init__(self, equation: concept.Equation,
            riemann: concept.RiemannSolver,
            degree: int, n_element: int, x_left: float, x_right: float,
            value_type=float) -> None:
        LagrangeFR.__init__(self, equation, riemann, degree,
            n_element, x_left, x_right, value_type)

    def name(self, verbose=True):
        return 'DGwithFR (Lagrange, '+r'$p$='+f'{self.degree()})'

    def get_residual_column(self):
        column = np.zeros(self.n_dof(), self._value_type)
        interface_fluxes = self.get_interface_fluxes()
        i_dof = 0
        for i in range(self._n_element):
            element = self.get_element_by_index(i)
            n_dof = element.n_dof()
            # build element_i's residual column
            # 1st: evaluate the internal integral
            upwind_flux_left = interface_fluxes[i]
            upwind_flux_right = interface_fluxes[i+1]
            def integrand(x_global):
                column = element.get_basis_values(x_global)
                gradient = element.get_flux_gradient(x_global,
                    upwind_flux_left, upwind_flux_right)
                return -column * gradient
            values = integrate.fixed_quad_global(integrand,
                element.x_left(), element.x_right(), element.n_term())
            # 2nd: evaluate the boundary integral
            # Nothing to do here.
            # 3rd: multiply the inverse of the mass matrix
            values = element.divide_mass_matrix(values)
            column[i_dof:i_dof+n_dof] = values
            i_dof += n_dof
        assert i_dof == self.n_dof()
        return column


if __name__ == '__main__':
    pass
