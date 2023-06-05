"""Define the abstract interfaces that must be defined in concrete implementations.
"""
import abc
import numpy as np
# from collections.abc import Tuple


class Polynomial(abc.ABC):
    """Polynomial functions defined on [-1, 1].
    """
    # TODO: rename to LocalPolynomial

    @abc.abstractmethod
    def local_to_value(self, x_local):
        """Evaluate the polynomial and return-by-value the result.
        """

    @abc.abstractmethod
    def local_to_gradient(self, x_local):
        """Evaluate the gradient of the polynomial and return-by-value the result.
        """


class Coordinate(abc.ABC):
    """A cell-like object that performs coordinate maps on it.
    """

    @abc.abstractmethod
    def jacobian_degree(self) -> int:
        """Degree of the Jacobian determinant.
        """

    @abc.abstractmethod
    def local_to_global(self, x_local: float) -> float:
        """Coordinate transform from local to global.
        """

    @abc.abstractmethod
    def global_to_local(self, x_global: float) -> float:
        """Coordinate transform from global to local.
        """

    @abc.abstractmethod
    def local_to_jacobian(self, x_local: float) -> float:
        """Get the Jacobian value at a point given by its local coordinate.
        """

    @abc.abstractmethod
    def global_to_jacobian(self, x_global: float) -> float:
        """Get the Jacobian value at a point given by its global coordinate.
        """

    def x_left(self):
        """Get the coordinate of this element's left boundary.
        """
        return self.local_to_global(-1)

    def x_right(self):
        """Get the coordinate of this element's right boundary.
        """
        return self.local_to_global(+1)

    def x_center(self):
        """Get the coordinate of this element's centroid.
        """
        return (self.x_right() + self.x_left()) / 2

    def length(self):
        """Get the length of this element.
        """
        return self.x_right() - self.x_left()


class ShiftedCoordinate(Coordinate):
    """An wrapper that acts as if a given coordinate is shifted along x-axis by a given amount.
    """

    def __init__(self, coordinate: Coordinate, x_shift: float):
        self._unshifed_coordinate = coordinate
        self._x_shift = x_shift

    def jacobian_degree(self) -> int:
        return self._unshifed_coordinate.jacobian_degree()

    def local_to_global(self, x_local: float) -> float:
        x_unshifted = self._unshifed_coordinate.local_to_global(x_local)
        return x_unshifted + self._x_shift

    def global_to_local(self, x_global: float) -> float:
        x_unshifed = x_global - self._x_shift
        return self._unshifed_coordinate.global_to_local(x_unshifed)

    def local_to_jacobian(self, x_local: float) -> float:
        return self._unshifed_coordinate.local_to_jacobian(x_local)

    def global_to_jacobian(self, x_global: float) -> float:
        x_unshifed = x_global - self._x_shift
        return self._unshifed_coordinate.global_to_jacobian(x_unshifed)

    def x_left(self):
        x_unshifted = self._unshifed_coordinate.x_left()
        return x_unshifted + self._x_shift

    def x_right(self):
        x_unshifted = self._unshifed_coordinate.x_right()
        return x_unshifted + self._x_shift

    def x_center(self):
        x_unshifted = self._unshifed_coordinate.x_center()
        return x_unshifted + self._x_shift

    def length(self):
        return self._unshifed_coordinate.length()


class Integrator(abc.ABC):
    """A cell-like object that performs integrations on it.
    """
    # TODO: rename to IntegrableElement

    def __init__(self, coordinate: Coordinate) -> None:
        assert isinstance(coordinate, Coordinate)
        self._coordinate = coordinate

    def coordinate(self) -> Coordinate:
        """Get the underlying coordinate mapping object.
        """
        return self._coordinate

    @abc.abstractmethod
    def get_quadrature_points(self, n_point: int) -> np.ndarray:
        """Get the global coordinates of a given number of quadrature points.
        """

    @abc.abstractmethod
    def fixed_quad_local(self, function: callable, n_point: int):
        """Integrate a function defined in local coordinates on [-1, 1] with a given number of quadrature points."""

    def fixed_quad_global(self, function: callable, n_point: int):
        """Integrate a function defined in global coordinates with a given number of quadrature points."""
        n_point = max(1, n_point)
        def integrand(x_local):
            x_global = self.coordinate().local_to_global(x_local)
            jacobian = self.coordinate().local_to_jacobian(x_local)
            return function(x_global) * jacobian
        return self.fixed_quad_local(integrand, n_point)

    def average(self, function: callable, n_point: int):
        """Get the average value of a function on this element.
        """
        integral = self.fixed_quad_global(function, n_point)
        return integral / self.coordinate().length()

    def norm_1(self, function: callable, n_point: int):
        """Get the L_1 norm of a function on this element.
        """
        value = self.fixed_quad_global(
            lambda x_global: np.abs(function(x_global)),
            n_point)
        return value

    def norm_2(self, function: callable, n_point: int):
        """Get the L_2 norm of a function on this element.
        """
        value = self.fixed_quad_global(
            lambda x_global: np.abs(function(x_global))**2,
            n_point)
        return np.sqrt(value)

    def norm_infty(self, function: callable, n_point: int):
        """Get the (approximate) L_infty norm of a function on this element.
        """
        points = np.linspace(
            self.coordinate().x_left(), self.coordinate().x_right(), n_point)
        # Alternatively, quadrature points can be used:
        # points = self.get_quadrature_points(n_point)
        value = 0.0
        for x_global in points:
            value = max(value, np.abs(function(x_global)))
        return value

    def inner_product(self, phi: callable, psi: callable, n_point):
        value = self.fixed_quad_global(
            lambda x_global: phi(x_global) * psi(x_global),
            n_point)
        return value


class Expansion(abc.ABC):
    """The polynomial approximation of a general function.
    """

    def __init__(self, coordinate: Coordinate, integrator: Integrator,
            value_type=float) -> None:
        self._coordinate = coordinate
        self._integrator = integrator
        self._value_type = value_type

    def value_type(self):
        """Get the type of value.
        """
        return self._value_type

    def coordinate(self) -> Coordinate:
        """Get a refenece to the underlying Coordinate object.
        """
        return self._coordinate

    def integrator(self) -> Integrator:
        """Get a refenece to the underlying Integrator object.
        """
        return self._integrator

    def x_left(self):
        return self.coordinate().x_left()

    def x_right(self):
        return self.coordinate().x_right()

    def x_center(self):
        return self.coordinate().x_center()

    def length(self):
        return self.coordinate().length()

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the expansion.
        """

    @abc.abstractmethod
    def n_term(self):
        """Number of terms in the approximated function.
        """

    @abc.abstractmethod
    def degree(self):
        """The highest degree of the approximated function.
        """

    @abc.abstractmethod
    def approximate(self, function: callable):
        """Approximate a general function.
        """

    @abc.abstractmethod
    def average(self):
        """Get the average value of the approximated function.
        """

    @abc.abstractmethod
    def get_basis(self, i_basis: int) -> callable:
        """Get i-th basis function, which maps x_global to its value.
        """

    @abc.abstractmethod
    def get_basis_values(self, x_global: float) -> np.ndarray:
        """Get the values of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_gradients(self, x_global: float) -> np.ndarray:
        """Get the gradients of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_hessians(self, x_global: float) -> np.ndarray:
        """Get the hessians of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_innerproducts(self):
        """Get the inner-products of each pair of basis functions.
        """

    @abc.abstractmethod
    def global_to_value(self, x_global: float):
        """Evaluate the approximation and return-by-value the result.
        """

    @abc.abstractmethod
    def global_to_gradient(self, x_global: float):
        """Evaluate the gradient of the approximation and return-by-value the result.
        """

    @abc.abstractmethod
    def set_coeff(self, coeff: np.ndarray):
        """Set the coefficient for each basis.
        """

    @abc.abstractmethod
    def get_coeff_ref(self) -> np.ndarray:
        """Get the reference to the coefficient for each basis.
        """


class ShiftedExpansion(Expansion):
    """An wrapper that acts as if a given expansion is shifted along x-axis by a given amount.
    """

    def __init__(self, expansion: Expansion, x_shift: float):
        Expansion.__init__(self,
            ShiftedCoordinate(expansion.coordinate(), x_shift),
            None, expansion.value_type())
        self._unshifted_expansion = expansion
        self._x_shift = x_shift

    def name(self, verbose: bool) -> str:
        return self._unshifted_expansion.name(verbose)

    def n_term(self):
        return self._unshifted_expansion.n_term()

    def degree(self):
        return self._unshifted_expansion.degree()

    def global_to_value(self, x_global: float):
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.global_to_value(x_unshifted)

    def global_to_gradient(self, x_global: float):
        x_unshifted = x_global - self._x_shift
        return self._unshifted_expansion.global_to_gradient(x_unshifted)

    def get_coeff_ref(self) -> np.ndarray:
        return self._unshifted_expansion.get_coeff_ref()

    # The following methods are banned for this wrapper.

    def approximate(self, function: callable):
        assert False

    def average(self):
        assert False

    def get_basis(self, i_basis: int) -> callable:
        assert False 

    def get_basis_values(self, x_global: float) -> np.ndarray:
        assert False

    def get_basis_gradients(self, x_global: float) -> np.ndarray:
        assert False

    def get_basis_hessians(self, x_global: float) -> np.ndarray:
        assert False

    def get_basis_innerproducts(self):
        assert False

    def set_coeff(self, coeff: np.ndarray):
        assert False


class Equation(abc.ABC):
    """A PDE in the form of ∂U/∂t + ∂F/∂x = ∂G/∂x + H.
    """

    def __init__(self, value_type) -> None:
        self._value_type = value_type

    def value_type(self):
        """Get the type of value.
        """
        return self._value_type

    @abc.abstractmethod
    def n_component(self):
        """Number of scalar components in U.
        """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the equation.
        """

    @abc.abstractmethod
    def get_convective_flux(self, u_given):
        """Get the value of F(U) for a given U.
        """

    @abc.abstractmethod
    def get_convective_speed(self, u_given):
        """Get the value of ∂F(U)/∂U for a given U.
        """

    @abc.abstractmethod
    def get_diffusive_coeff(self, u_given):
        """Get the value of b for a given U if G(U, ∂U/∂x) = b(U) * ∂U/∂x.
        """

    @abc.abstractmethod
    def get_diffusive_flux(self, u_given, du_dx_given):
        """Get the value of G(U, ∂U/∂x) for a given pair of U and ∂U/∂x.
        """

    @abc.abstractmethod
    def get_source(self, u_given):
        """Get the value of H(U) for a given U.
        """


class RiemannSolver(abc.ABC):
    """An exact or approximate solver for the Riemann problem of an Equation.
    """

    def __init__(self, equation: Equation) -> None:
        self._equation = equation

    def equation(self) -> Equation:
        return self._equation

    def value_type(self):
        return self.equation().value_type()

    @abc.abstractmethod
    def get_value(self, x_global, t_curr):
        """Get the solution of a Riemann problem.
        """

    @abc.abstractmethod
    def get_upwind_flux(self, u_left, u_right):
        """Get the value of the convective flux on the interface.
        """

    @abc.abstractmethod
    def get_interface_gradient(self, h_left: float, h_right: float,
            u_jump, du_mean, ddu_jump):
        """Get the value of ∂U/∂x on the interface.
        """

    @abc.abstractmethod
    def get_interface_flux(self, u_left: Expansion, u_right: Expansion,
            viscous: float):
        """Get the value of F - G on the interface.
        
        It is assumed G is in the form of nu * ∂U/∂x.
        """


class Element(abc.ABC):
    """A cell-like object that carries some expansion of the solution.
    """
    # TODO: rename to PhysicalElement

    def __init__(self, riemann: RiemannSolver, expansion: Expansion) -> None:
        assert isinstance(riemann, RiemannSolver)
        self._riemann = riemann
        self._expansion = expansion
        self._coordinate = expansion.coordinate()
        self._integrator = expansion.integrator()
        self._left_expansion = None
        self._right_expansion = None

    def equation(self) -> Equation:
        """Get a refenece to the underlying Equation object.
        """
        return self._riemann.equation()

    def value_type(self):
        return self.equation().value_type()

    def coordinate(self) -> Coordinate:
        """Get a refenece to the underlying Coordinate object.
        """
        return self._coordinate

    def integrator(self) -> Integrator:
        """Get a refenece to the underlying Integrator object.
        """
        return self._integrator

    def expansion(self) -> Expansion:
        """Get a refenece to the underlying Expansion object.
        """
        return self._expansion

    def neighbor_expansions(self) -> tuple[Expansion, Expansion]:
        return self._left_expansion, self._right_expansion

    def x_left(self):
        return self.coordinate().x_left()

    def x_right(self):
        return self.coordinate().x_right()

    def x_center(self):
        return self.coordinate().x_center()

    def length(self):
        return self.coordinate().length()

    def degree(self):
        """The highest degree of the approximated solution.
        """
        return self.expansion().degree()

    def n_term(self):
        """Number of terms in the approximated function.
        """
        return self.expansion().n_term()

    def n_dof(self):
        """Count degrees of freedom in current object.
        """
        return self.n_term() * self.equation().n_component()

    def approximate(self, function: callable):
        """Approximate a general function as u^h.
        """
        self.expansion().approximate(function)

    def fixed_quad_global(self, function: callable, n_point: int):
        return self.integrator().fixed_quad_global(function, n_point)

    def get_basis_values(self, x_global: float) -> np.ndarray:
        """Get the values of basis at a given point.
        """
        return self.expansion().get_basis_values(x_global)

    def get_basis_gradients(self, x_global: float) -> np.ndarray:
        """Get the gradients of basis at a given point.
        """
        return self.expansion().get_basis_gradients(x_global)

    def get_basis_hessians(self, x_global: float) -> np.ndarray:
        """Get the hessians of basis at a given point.
        """
        return self.expansion().get_basis_hessians(x_global)

    def _build_mass_matrix(self):
        mass_matrix = self.expansion().get_basis_innerproducts()
        assert self.n_term() == self.n_dof()
        # Otherwise, it should be spanned to a block diagonal matrix.
        return mass_matrix

    def get_convective_speed(self, x_global: float):
        """Get the value of a(u^h) at a given point.
        """
        return self.equation().get_convective_speed(
              self.get_solution_value(x_global))

    def get_discontinuous_flux(self, x_global, extra_viscous=0.0):
        """Get the value of f(u^h) at a given point.
        """
        u_approx = self.get_solution_value(x_global)
        flux = self.equation().get_convective_flux(u_approx)
        du_approx = self.expansion().global_to_gradient(x_global)
        flux -= self.equation().get_diffusive_flux(u_approx, du_approx)
        flux -= extra_viscous * du_approx
        return flux

    def get_solution_value(self, x_global: float):
        """Get the value of u^h at a given point.
        """
        return self.expansion().global_to_value(x_global)

    def get_solution_gradient(self, x_global: float):
        """Get the gradient value of u^h at a given point.
        """
        return self.expansion().global_to_gradient(x_global)

    def set_solution_coeff(self, column):
        """Set coefficients of the solution's expansion.

        The element is responsible for the column-to-coeff conversion.
        """
        assert self.equation().n_component() == 1
        self.expansion().set_coeff(column)

    def get_solution_column(self) -> np.ndarray:
        """Get coefficients of the solution's expansion.

        The element is responsible for the coeff-to-column conversion.
        """
        assert self.equation().n_component() == 1
        return self.expansion().get_coeff_ref()

    @abc.abstractmethod
    def divide_mass_matrix(self, column: np.ndarray):
        """Divide the mass matrix of this element by the given column.

        Solve the system of linear equations Ax = b, in which A is the mass matrix of this element.
        """

    @abc.abstractmethod
    def suggest_cfl(self, rk_order: int) -> float:
        """Suggest a CFL number for explicit RK time stepping.
        """

    def suggest_delta_t(self, extra_viscous: float):
        points = np.linspace(self.x_left(), self.x_right(), self.n_term()+1)
        h = self.length()
        p = self.degree()
        if p == 0:
            spatial_factor = 1
        elif p == 1:
            spatial_factor = 4
        elif p == 2:
            spatial_factor = 6
        elif p == 3:
            spatial_factor = 12
        else:
            spatial_factor = 26
        delta_t = np.infty
        for x in points:
            u = self.get_solution_value(x)
            a = self.equation().get_convective_speed(u)
            lambda_c = np.abs(a)
            b = self.equation().get_diffusive_coeff(u) + extra_viscous
            lambda_d = b / h
            delta_t = min(delta_t, h / (lambda_c + spatial_factor * lambda_d))
        return delta_t * self.suggest_cfl(3)


class Grid(abc.ABC):

    def __init__(self, n_element: int):
        self._elements = np.ndarray(n_element, Element)

    @abc.abstractmethod
    def get_element_index(self, point) -> int:
        """Get the index of the element in which the given point locates.
        """

    def get_element_by_index(self, index: int) -> Element:
        """Get the index of the element in which the point locates.
        """
        return self._elements[index]

    def get_element(self, point) -> Element:
        """Get the element in which the given point locates.
        """
        return self._elements[self.get_element_index(point)]

    def x_left(self):
        """Get the coordinate of this object's left boundary.
        """
        return self.get_element_by_index(0).x_left()

    def x_right(self):
        """Get the coordinate of this object's right boundary.
        """
        return self.get_element_by_index(-1).x_right()

    def length(self):
        """Get the length of this object.
        """
        return self.x_right() - self.x_left()

    def n_element(self):
        """Count elements in this object.
        """
        return len(self._elements)

    def delta_x(self, i_cell: int):
        """Get the length of the i-th element.
        """
        return self.get_element_by_index(i_cell).length()


class Detector(abc.ABC):
    """An object that detects jumps on an element.
    """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the detector.
        """

    @abc.abstractmethod
    def get_troubled_cell_indices(self, grid: Grid):
        """Whether the current element is troubled.
        """


class Limiter(abc.ABC):
    """An object that limits oscillations on an element.
    """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the limiter.
        """

    @abc.abstractmethod
    def reconstruct(self, troubled_cell_indices, grid: Grid, periodic: bool):
        """Reconstruct the expansion on each troubled cell.
        """

    @abc.abstractmethod
    def get_new_coeff(self, curr: Element, neighbors) -> np.ndarray:
        """Reconstruct the expansion on a troubled cell.
        """


class Viscous(abc.ABC):
    """An object that adds artificial viscosity to a spatial scheme.
    """

    def __init__(self) -> None:
        self._index_to_coeff = dict()

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the limiter.
        """

    @abc.abstractmethod
    def generate(self, troubled_cell_indices, grid: Grid, periodic: bool):
        """Generate artificial viscosity for each troubled cell.
        """

    def get_coeff(self, i_cell: int):
        """Get the viscous coefficient of the ith cell.
        """
        assert 0 <= i_cell
        if i_cell in self._index_to_coeff:
            return self._index_to_coeff[i_cell]
        else:
            return 0.0


class OdeSystem(abc.ABC):
    """A system of ODEs in the form of dU/∂t = R.
    """

    @abc.abstractmethod
    def set_solution_column(self, column):
        """Overwrite the values in the solution column.
        """

    @abc.abstractmethod
    def get_solution_column(self):
        """Get a copy of the solution column.
        """

    @abc.abstractmethod
    def get_residual_column(self):
        """Get a copy of the residual column.
        """

    def set_time(self, t_curr: float):
        """Set the current time value.

        For steady system, it does nothing.
        """


class OdeSolver(abc.ABC):
    """A solver for the standard ODE system dU/dt = R.

    The solution column U and the residual column R is provided by a OdeSystem object.
    """

    @abc.abstractmethod
    def update(self, ode_system: OdeSystem, delta_t: float, t_curr: float):
        """Update the given OdeSystem object to the next time step.
        """


class SpatialScheme(Grid, OdeSystem):
    """An ODE system given by some spatial scheme.
    """

    def __init__(self, riemann: RiemannSolver,
            n_element: int, x_left: float, x_right: float,
            detector=None, limiter=None, viscous=None) -> None:
        assert x_left < x_right
        assert n_element > 1
        Grid.__init__(self, n_element)
        self._riemann = riemann
        self._detector = detector
        self._limiter = limiter
        self._viscous = viscous

    def value_type(self):
        return self._riemann.value_type()

    def degree(self):
        """Get the degree of the polynomial approximation.
        """
        return self.get_element_by_index(0).degree()

    def equation(self) -> Equation:
        return self._riemann.equation()

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the compact string representation of the method.
        """

    def is_periodic(self):
        """Whether a periodic boundary condition is applied.
        """
        return True

    def _get_shifted_expansion(self, i_cell: int, x_shift: float):
        cell_i = self.get_element_by_index(i_cell)
        return ShiftedExpansion(cell_i.expansion(), x_shift)

    def link_neighbors(self):
        """Link each element to its neighbors' expansions.
        """
        for i_cell in range(1, self.n_element() - 1):
            cell_i = self.get_element_by_index(i_cell)
            cell_i._left_expansion = \
                self.get_element_by_index(i_cell - 1).expansion()
            cell_i._right_expansion = \
                self.get_element_by_index(i_cell + 1).expansion()
        if self.is_periodic():
            self.get_element_by_index(0)._left_expansion = \
                self._get_shifted_expansion(-1, -self.length())
            self.get_element_by_index(-1)._right_expansion = \
                self._get_shifted_expansion(0, +self.length())

    @abc.abstractmethod
    def n_dof(self):
        """Count degrees of freedom in current object.
        """

    @abc.abstractmethod
    def initialize(self, function: callable):
        """Set the initial condition.
        """

    @abc.abstractmethod
    def get_solution_value(self, point):
        """Get the solution value at a given point.
        """

    @abc.abstractmethod
    def get_flux_value(self, point):
        """Get the flux value at a given point.
        """

    def set_detector_and_limiter(self, detector, limiter, viscous):
        if isinstance(detector, Detector):
            self._detector = detector
        if isinstance(limiter, Limiter):
            self._limiter = limiter
        if isinstance(viscous, Viscous):
            self._viscous = viscous

    def suppress_oscillations(self):
        if self._detector:
            indices = self._detector.get_troubled_cell_indices(
                self, self.is_periodic())
            if self._limiter:
                self._limiter.reconstruct(indices, self, self.is_periodic())
            if self._viscous:
                self._viscous.generate(indices, self, self.is_periodic())

    def suggest_delta_t(self, delta_t):
        for i_cell in range(self.n_element()):
            cell_i = self.get_element_by_index(i_cell)
            extra_viscous = 0.0
            if self._viscous:
                extra_viscous = self._viscous.get_coeff(i_cell)
            delta_t = min(delta_t, cell_i.suggest_delta_t(extra_viscous))
        return delta_t


if __name__ == '__main__':
    pass
