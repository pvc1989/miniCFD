"""Define the abstract interfaces that must be defined in concrete implementations.
"""
import abc
import numpy as np


class Polynomial(abc.ABC):
    """Polynomial functions defined on [-1, 1].
    """
    # TODO: rename to LocalPolynomial

    @abc.abstractmethod
    def get_function_value(self, x_local):
        """Evaluate the polynomial and return-by-value the result.
        """

    @abc.abstractmethod
    def get_gradient_value(self, x_local):
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
        return self.local_to_global(0)

    def length(self):
        """Get the length of this element.
        """
        return self.x_right() - self.x_left()


class Integrator(abc.ABC):
    """A cell-like object that performs integrations on it.
    """
    # TODO: rename to IntegrableElement

    def __init__(self, coordinate: Coordinate) -> None:
        assert isinstance(coordinate, Coordinate)
        self.coordinate = coordinate

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
            x_global = self.coordinate.local_to_global(x_local)
            jacobian = self.coordinate.local_to_jacobian(x_local)
            return function(x_global) * jacobian
        return self.fixed_quad_local(integrand, n_point)

    def average(self, function: callable, n_point: int):
        """Get the average value of a function on this element.
        """
        integral = self.fixed_quad_global(function, n_point)
        return integral / self.coordinate.length()

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
            self.coordinate.x_left(), self.coordinate.x_right(), n_point)
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

    def __init__(self, coordinate: Coordinate, integrator: Integrator) -> None:
        self.coordinate = coordinate
        self.integrator = integrator

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
    def get_average(self):
        """Get the average value of the approximated function.
        """

    @abc.abstractmethod
    def get_basis(self, i_basis: int) -> callable:
        """Get i-th basis function, which maps x_global to its value.
        """

    @abc.abstractmethod
    def get_basis_values(self, x_global: float):
        """Get the values of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_gradients(self, x_global: float):
        """Get the gradients of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_innerproducts(self):
        """Get the inner-products of each pair of basis functions.
        """

    @abc.abstractmethod
    def get_function_value(self, x_global: float):
        """Evaluate the approximation and return-by-value the result.
        """

    @abc.abstractmethod
    def get_gradient_value(self, x_global: float):
        """Evaluate the gradient of the approximation and return-by-value the result.
        """

    @abc.abstractmethod
    def set_coeff(self, coeff):
        """Set the coefficient for each basis.
        """

    @abc.abstractmethod
    def get_coeff_ref(self):
        """Get the reference to the coefficient for each basis.
        """


class Equation(abc.ABC):
    """A PDE in the form of ∂U/∂t + ∂F/∂x = ∂G/∂x + H.
    """

    def n_component(self):
        """Number of scalar components in U.
        """
        return 1

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the equation.
        """

    @abc.abstractmethod
    def get_convective_flux(self, u_given):
        """Get the value of F(U) for a given U.
        """

    @abc.abstractmethod
    def get_convective_jacobian(self, u_given):
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


class Element(abc.ABC):
    """A cell-like object that carries some expansion of the solution.
    """
    # TODO: rename to PhysicalElement

    def __init__(self, equation: Equation, coordinate: Coordinate,
            expansion: Expansion, value_type=float) -> None:
        assert isinstance(equation, Equation)
        assert isinstance(coordinate, Coordinate)
        self._equation = equation
        self.coordinate = coordinate
        self.integrator = expansion.integrator
        self.expansion = expansion
        self._value_type = value_type

    def x_left(self):
        return self.coordinate.x_left()

    def x_right(self):
        return self.coordinate.x_right()

    def x_center(self):
        return self.coordinate.x_center()

    def length(self):
        return self.coordinate.length()

    def degree(self):
        """The highest degree of the approximated solution.
        """
        return self.expansion.degree()

    def n_term(self):
        """Number of terms in the approximated function.
        """
        return self.expansion.n_term()

    def n_dof(self):
        """Count degrees of freedom in current object.
        """
        return self.n_term() * self._equation.n_component()

    def approximate(self, function: callable):
        """Approximate a general function as u^h.
        """
        self.expansion.approximate(function)

    def fixed_quad_global(self, function: callable, n_point: int):
        return self.integrator.fixed_quad_global(function, n_point)

    def get_basis_values(self, x_global: float):
        """Get the values of basis at a given point.
        """
        return self.expansion.get_basis_values(x_global)

    def get_basis_gradients(self, x_global):
        return self.expansion.get_basis_gradients(x_global)

    def _build_mass_matrix(self):
        mass_matrix = self.expansion.get_basis_innerproducts()
        assert self.n_term() == self.n_dof()
        # Otherwise, it should be spanned to a block diagonal matrix.
        return mass_matrix

    def get_convective_jacobian(self, x_global: float):
        """Get the value of a(u^h) at a given point.
        """
        return self._equation.get_convective_jacobian(
              self.get_solution_value(x_global))

    def get_discontinuous_flux(self, x_global, extra_viscous=0.0):
        """Get the value of f(u^h) at a given point.
        """
        u_approx = self.get_solution_value(x_global)
        flux = self._equation.get_convective_flux(u_approx)
        du_approx = self.expansion.get_gradient_value(x_global)
        flux -= self._equation.get_diffusive_flux(u_approx, du_approx)
        flux -= extra_viscous * du_approx
        return flux

    def get_solution_value(self, x_global: float):
        """Get the value of u^h at a given point.
        """
        return self.expansion.get_function_value(x_global)

    def get_solution_gradient(self, x_global: float):
        """Get the gradient value of u^h at a given point.
        """
        return self.expansion.get_gradient_value(x_global)

    def set_solution_coeff(self, column):
        """Set coefficients of the solution's expansion.

        The element is responsible for the column-to-coeff conversion.
        """
        assert self._equation.n_component() == 1
        self.expansion.set_coeff(column)

    def get_solution_column(self):
        """Get coefficients of the solution's expansion.

        The element is responsible for the coeff-to-column conversion.
        """
        assert self._equation.n_component() == 1
        return self.expansion.get_coeff_ref()

    @abc.abstractmethod
    def divide_mass_matrix(self, column: np.ndarray):
        """Divide the mass matrix of this element by the given column.

        Solve the system of linear equations Ax = b, in which A is the mass matrix of this element.
        """


class RiemannSolver(abc.ABC):
    """An exact or approximate solver for the Riemann problem of a conservation law.
    """

    @abc.abstractmethod
    def get_value(self, x_global, t_curr):
        """Get the solution of a Riemann problem.
        """

    @abc.abstractmethod
    def get_upwind_flux(self, u_left, u_right):
        """Get the flux at (x=0, t) from the solution of a Riemann problem.
        """


class Detector(abc.ABC):
    """An object that detects jumps on an element.
    """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the detector.
        """

    @abc.abstractmethod
    def get_troubled_cell_indices(self, elements, periodic: bool):
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
    def reconstruct(self, troubled_cell_indices, elements, periodic: bool):
        """Reconstruct the expansion on each troubled cell.
        """

    @abc.abstractmethod
    def get_new_coeff(self, curr: Element, neighbors) -> np.ndarray:
        """Reconstruct the expansion on a troubled cell.
        """


class Viscous(abc.ABC):
    """An object that adds artificial viscosity to a spatial scheme.
    """

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the name of the limiter.
        """

    @abc.abstractmethod
    def generate(self, troubled_cell_indices, elements, periodic: bool):
        """Generate artificial viscosity for each troubled cell.
        """

    def get_coeff(self, i_cell: int):
        """Get the viscous coefficient of the ith cell.
        """
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


class SpatialScheme(OdeSystem):
    """An ODE system given by some spatial scheme.
    """

    def __init__(self, equation: Equation,
            n_element: int, x_left: float, x_right: float,
            detector=None, limiter=None, viscous=None) -> None:
        assert x_left < x_right
        assert n_element > 1
        self.equation = equation
        self._elements = np.ndarray(n_element, Element)
        self._detector = detector
        self._limiter = limiter
        self._viscous = viscous

    def degree(self):
        """Get the degree of the polynomial approximation.
        """
        return self.get_element_by_index(0).degree()

    @abc.abstractmethod
    def name(self, verbose: bool) -> str:
        """Get the compact string representation of the method.
        """

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

    def delta_x(self):
        """Get the length of each element.

        Currently, only uniform meshing is supported.
        """
        return self.get_element_by_index(0).length()

    def is_periodic(self):
        """Whether a periodic boundary condition is applied.
        """
        return True

    @abc.abstractmethod
    def n_dof(self):
        """Count degrees of freedom in current object.
        """

    @abc.abstractmethod
    def initialize(self, function: callable):
        """Set the initial condition.
        """

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

    def set_detector_and_limiter(self, detector, limiter, viscous):
        assert isinstance(detector, Detector)
        assert isinstance(limiter, Limiter)
        assert isinstance(viscous, Viscous)
        self._detector = detector
        self._limiter = limiter
        self._viscous = viscous

    def suppress_oscillations(self):
        if isinstance(self._detector, Detector):
            indices = self._detector.get_troubled_cell_indices(
                self._elements, self.is_periodic())
            if isinstance(self._limiter, Limiter):
                self._limiter.reconstruct(indices,
                    self._elements, self.is_periodic())
            if isinstance(self._viscous, Viscous):
                self._viscous.generate(indices,
                    self._elements, self.is_periodic())


if __name__ == '__main__':
    pass
