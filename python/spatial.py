"""Concrete implementations of spatial schemes.
"""
import numpy as np
from numpy.testing import assert_almost_equal
import bisect

import concept
import coordinate
import expansion
import element


class FiniteVolume(concept.SpatialScheme):
    """An ODE system given by some FiniteVolume scheme.
    """


class FiniteElement(concept.SpatialScheme):
    """An ODE system given by some FiniteElement scheme.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float,
            ElementType: concept.Element) -> None:
        concept.SpatialScheme.__init__(self, riemann,
            periodic, n_element, x_left, x_right)
        assert degree >= 0
        delta_x = (x_right - x_left) / n_element
        self._x_left_sorted = np.ndarray(n_element)
        x_left_i = x_left
        for i_element in range(n_element):
            assert_almost_equal(x_left_i, x_left + i_element * delta_x)
            self._x_left_sorted[i_element] = x_left_i
            x_right_i = x_left_i + delta_x
            element_i = ElementType(riemann, degree,
                coordinate.Linear(x_left_i, x_right_i))
            self._elements[i_element] = element_i
            x_left_i = x_right_i
        assert_almost_equal(x_left_i, x_right)
        self.link_neighbors()

    def n_dof(self):
        return self.n_element() * self.get_element_by_index(0).n_dof()

    def get_element_index(self, x_global):
        i_element = bisect.bisect_right(self._x_left_sorted, x_global)
        # bisect_right(a, x) gives such an i that a[:i] <= x < a[i:]
        return i_element - 1

    def get_interface_fluxes_and_bjumps(self):
        interface_fluxes = np.ndarray(self.n_element() + 1, self.value_type())
        interface_bjumps = np.ndarray(self.n_element() + 1, self.value_type())
        # interface_flux[i] := flux on interface(element[i-1], element[i])
        for i in range(1, self.n_element()):
            curr = self.get_element_by_index(i)
            prev = self.get_element_by_index(i-1)
            try:
                interface_fluxes[i], interface_bjumps[i] = \
                    self._riemann.get_interface_flux_and_bjump(prev, curr)
            except Exception as e:
                print(f'Riemann solver failed between element[{i-1}] and element[{i}].')
                raise e
        if self.is_periodic():
            i_prev = self.n_element() - 1
            curr = self.get_element_by_index(0)
            prev = self.get_element_by_index(i_prev)
            interface_fluxes[0], interface_bjumps[0] = \
                self._riemann.get_interface_flux_and_bjump(prev, curr)
            interface_fluxes[-1] = interface_fluxes[0]
            interface_bjumps[-1] = interface_bjumps[0]
        else:  # TODO: support other boundary condtions
            curr = self.get_element_by_index(0)
            interface_fluxes[0] = curr.get_dg_flux(curr.x_left())
            interface_bjumps[0] = interface_bjumps[1] * 0
            curr = self.get_element_by_index(-1)
            interface_fluxes[-1] = curr.get_dg_flux(curr.x_right())
            interface_bjumps[-1] = interface_bjumps[0]
        return interface_fluxes, interface_bjumps

    def get_solution_value(self, point):
        return self.get_element(point).get_solution_value(point)

    def _get_shifted_expansion(self, i_cell: int, x_shift: float):
        cell_i = self.get_element_by_index(i_cell)
        return expansion.Shifted(cell_i.expansion(), x_shift)

    def link_neighbors(self):
        """Link each element to its neighbors' expansions.
        """
        self.get_element_by_index(0)._right_expansion = \
            self.get_element_by_index(1).expansion()
        for i_cell in range(1, self.n_element() - 1):
            cell_i = self.get_element_by_index(i_cell)
            cell_i._left_expansion = \
                self.get_element_by_index(i_cell - 1).expansion()
            cell_i._right_expansion = \
                self.get_element_by_index(i_cell + 1).expansion()
        self.get_element_by_index(-1)._left_expansion = \
            self.get_element_by_index(-2).expansion()
        if self.is_periodic():
            self.get_element_by_index(0)._left_expansion = \
                self._get_shifted_expansion(-1, -self.length())
            self.get_element_by_index(-1)._right_expansion = \
                self._get_shifted_expansion(0, +self.length())

    def set_solution_column(self, column):
        assert len(column) == self.n_dof()
        first = 0
        for element_i in self._elements:
            assert isinstance(element_i, concept.Element)
            last = first + element_i.n_dof()
            element_i.set_solution_coeff(column[first:last])
            first = last
        assert first == self.n_dof()
        self.suppress_oscillations()

    def get_solution_column(self):
        column = np.zeros(self.n_dof(), self.scalar_type())
        first = 0
        for element_i in self._elements:
            assert isinstance(element_i, concept.Element)
            last = first + element_i.n_dof()
            column[first:last] = element_i.get_solution_column()
            first = last
        assert first == self.n_dof()
        return column

    def _write_to_column(self, column: np.ndarray, values: np.ndarray, i_dof):
        n_row = len(values)
        if np.isscalar(values[0]):
            column[i_dof:i_dof+n_row] = values
            i_dof += n_row
        else:
            n_col = len(values[0])
            for i_col in range(n_col):
                for i_row in range(n_row):
                    column[i_dof] = values[i_row][i_col]
                    i_dof += 1
        return i_dof

    def initialize(self, function: callable):
        for element_i in self._elements:
            assert isinstance(element_i, concept.Element)
            element_i.approximate(function)
        self.suppress_oscillations()

    def get_residual_column(self):
        column = np.zeros(self.n_dof(), self.scalar_type())
        interface_fluxes, interface_jumps = \
            self.get_interface_fluxes_and_bjumps()
        i_dof = 0
        for i in range(self.n_element()):
            element_i = self.get_element_by_index(i)
            # build element_i's residual column
            # 1st: evaluate the internal integral
            residual = element_i.get_interior_residual()
            # 2nd: evaluate the boundary integral
            element_i.add_interface_residual(
                interface_fluxes[i], interface_fluxes[i+1], residual)
            element_i.add_interface_correction(
                interface_jumps[i], interface_jumps[i+1], residual)
            # 3rd: multiply the inverse of the mass matrix
            residual = element_i.divide_mass_matrix(residual)
            # write to the global column
            i_dof = self._write_to_column(column, residual, i_dof)
        assert i_dof == self.n_dof()
        return column


class DiscontinuousGalerkin(FiniteElement):
    """An mid-level class that defines common methods for all DG schemes.
    """

    def get_flux_value(self, point):
        i_cell = self.get_element_index(point)
        cell_i = self.get_element_by_index(i_cell)
        return cell_i.get_dg_flux(point)


class LegendreDG(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Legendre expansion.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float):
        DiscontinuousGalerkin.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.LegendreDG)

    def name(self, verbose=True):
        my_name = 'LegendreDG'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class DGonUniformRoots(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Lagrange expansion on uniform roots.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float):
        DiscontinuousGalerkin.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.DGonUniformRoots)

    def name(self, verbose=True):
        my_name = 'LagrangeDG'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class DGonLegendreRoots(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Lagrange expansion on Legendre roots.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float):
        DiscontinuousGalerkin.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.DGonLegendreRoots)

    def name(self, verbose=True):
        my_name = 'DGonLegendreRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class DGonLobattoRoots(DiscontinuousGalerkin):
    """The ODE system given by the DG method using a Lagrange expansion on Lobatto roots.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float):
        DiscontinuousGalerkin.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.DGonLobattoRoots)

    def name(self, verbose=True):
        my_name = 'DGonLobattoRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class FluxReconstruction(FiniteElement):
    """An mid-level class that defines common methods for all FR schemes.
    """

    def get_flux_value(self, point):
        curr = self.get_element_index(point)
        # solve riemann problem at the left end of curr element
        right = self.get_element_by_index(curr)
        left = self.get_element_by_index(curr-1)
        upwind_flux_left = self._riemann.get_interface_flux(left, right)
        # solve riemann problem at the right end of curr element
        left = self.get_element_by_index(curr)
        right = self.get_element_by_index((curr + 1) % self.n_element())
        upwind_flux_right = self._riemann.get_interface_flux(left, right)
        assert (isinstance(left, element.LagrangeFR)
            or isinstance(left, element.LegendreFR))
        return left.get_fr_flux(point,
            upwind_flux_left, upwind_flux_right)


class FRonUniformRoots(FluxReconstruction):
    """The ODE system given by the DG method using a Lagrange expansion on uniform roots.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float):
        FluxReconstruction.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.FRonUniformRoots)

    def name(self, verbose=True):
        my_name = 'FRonUniformRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class FRonLegendreRoots(FluxReconstruction):
    """The ODE system given by the FR method using a Lagrange expansion on Legendre roots.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float):
        FluxReconstruction.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.FRonLegendreRoots)

    def name(self, verbose=True):
        my_name = 'FRonLegendreRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class FRonLobattoRoots(FluxReconstruction):
    """The ODE system given by the FR method using a Lagrange expansion on Lobatto roots.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int, 
            periodic: bool, n_element: int, x_left: float, x_right: float):
        FluxReconstruction.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.FRonLobattoRoots)

    def name(self, verbose=True):
        my_name = 'FRonLobattoRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LegendreFR(FluxReconstruction):
    """The ODE system given by Huyhn's FR method.
    """

    def __init__(self, riemann: concept.RiemannSolver, degree: int,
            periodic: bool, n_element: int, x_left: float, x_right: float):
        FluxReconstruction.__init__(self, riemann, degree,
            periodic, n_element, x_left, x_right, element.LegendreFR)

    def name(self, verbose=True):
        my_name = 'LegendreFR'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


if __name__ == '__main__':
    pass
