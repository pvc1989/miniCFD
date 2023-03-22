"""Define the abstract interfaces that must be defined in concrete implementations.
"""
import abc
import numpy as np


class Polynomial(abc.ABC):
    """Polynomial functions defined on [-1, 1].
    """

    @abc.abstractmethod
    def get_function_value(self, x_local):
        """Evaluate the polynomial and return-by-value the result.
        """

    @abc.abstractmethod
    def get_gradient_value(self, x_local):
        """Evaluate the gradient of the polynomial and return-by-value the result.
        """


class Expansion(abc.ABC):
    """The polynomial approximation of a general function.
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
    def get_basis_values(self, x_global: float):
        """Get the values of all basis functions at a given point.
        """

    @abc.abstractmethod
    def get_basis_gradients(self, x_global: float):
        """Get the gradients of all basis functions at a given point.
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
    def get_coeff(self):
        """Get the coefficient for each basis.
        """


class Equation(abc.ABC):
    """A PDE in the form of ∂U/∂t + ∂F/∂x = ∂G/∂x + H.
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
    def get_diffusive_flux(self, u_given):
        """Get the value of G(U) for a given U.
        """

    @abc.abstractmethod
    def get_source(self, u_given):
        """Get the value of H(U) for a given U.
        """


class Element(abc.ABC):
    """A subdomain that carries some expansion of the solution.
    """

    def __init__(self, equation: Equation, x_left: float, x_right: float, value_type=float) -> None:
        self._equation = equation
        self._boundaries = (x_left, x_right)
        self._value_type = value_type

    def x_left(self):
        """Get the coordinate of this eleement's left boundary.
        """
        return self._boundaries[0]

    def x_right(self):
        """Get the coordinate of this eleement's right boundary.
        """
        return self._boundaries[1]

    def x_center(self):
        """Get the coordinate of this eleement's centroid.
        """
        return (self.x_left() + self.x_right()) / 2

    @abc.abstractmethod
    def degree(self):
        """The highest degree of the approximated solution.
        """

    @abc.abstractmethod
    def n_dof(self):
        """Count degrees of freedom in current object.
        """

    @abc.abstractmethod
    def approximate(self, function: callable):
        """Approximate a general function as u^h.
        """

    @abc.abstractmethod
    def get_basis_values(self, x_global: float):
        """Get the values of basis at a given point.
        """

    @abc.abstractmethod
    def get_solution_value(self, x_global: float):
        """Get the value of u^h at a given point.
        """

    def get_convective_jacobian(self, x_global: float):
        """Get the value of a(u^h) at a given point.
        """
        return self._equation.get_convective_jacobian(
              self.get_solution_value(x_global))

    @abc.abstractmethod
    def set_solution_coeff(self, column):
        """Set coefficients of the solution's expansion.

        The element is responsible for the column-to-coeff conversion.
        """

    @abc.abstractmethod
    def get_solution_column(self):
        """Get coefficients of the solution's expansion.

        The element is responsible for the coeff-to-column conversion.
        """

    @abc.abstractmethod
    def divide_mass_matrix(self, column: np.ndarray):
        """Divide the mass matrix of this element by the given column.

        Solve the system of linear equations Ax = b, in which A is the mass matrix of this element.
        """

    @abc.abstractmethod
    def get_quadrature_points(self, n_point: int) -> np.ndarray:
        """Get the global coordinates of quadrature points.
        """


class RiemannSolver(abc.ABC):
    """An exact or approximate solver for the Riemann problem of a conservation law.
    """

    @abc.abstractmethod
    def get_upwind_flux(self, u_left, u_right):
        """Get the solution of a Riemann problem.
        """


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


class OdeSolver(abc.ABC):
    """A solver for the standard ODE system dU/dt = R.

    The solution column U and the residual column R is provided by a OdeSystem object.
    """

    @abc.abstractmethod
    def update(self, ode_system: OdeSystem, delta_t: float):
        """Update the given OdeSystem object to the next time step.
        """

    def solve(self, ode_system: OdeSystem, plot: callable,
            t_start: float, t_stop: float, delta_t: float):
        """Solve the OdeSystem in a given time range.
        """
        n_step = int(np.ceil((t_stop - t_start) / delta_t))
        delta_t = (t_stop - t_start) / n_step
        plot_steps = n_step // 4
        for i_step in range(n_step + 1):
            t_curr = t_start + i_step * delta_t
            if i_step % plot_steps == 0:
                print(f'step {i_step}, t = {t_curr}')
                plot(t_curr)
            if i_step < n_step:
                self.update(ode_system, delta_t)


class SpatialScheme(OdeSystem):
    """An ODE system given by some spatial scheme.
    """

    def __init__(self, n_element: int, x_left: float, x_right: float) -> None:
        assert x_left < x_right
        assert n_element > 1
        self._n_element = n_element
        self._delta_x = (x_right - x_left) / n_element
        self._elements = np.ndarray(n_element, Element)

    def degree(self):
        """Get the degree of the polynomial approximation.
        """
        return self._elements[0].degree()

    @abc.abstractstaticmethod
    def name():
        """Get the compact string representation of the method.
        """

    def x_left(self):
        """Get the coordinate of this object's left boundary.
        """
        return self._elements[0].x_left()

    def x_right(self):
        """Get the coordinate of this object's right boundary.
        """
        return self._elements[-1].x_right()

    def length(self):
        """Get the length of this object.
        """
        return self.x_right() - self.x_left()

    def n_element(self):
        """Count elements in this object.
        """
        return self._n_element

    def delta_x(self):
        """Get the length of each element.

        Currently, only uniform meshing is supported.
        """
        return self._delta_x

    @abc.abstractmethod
    def n_dof(self):
        """Count degrees of freedom in current object.
        """

    @abc.abstractmethod
    def initialize(self, function: callable):
        """Set the initial condition.
        """


class Smoothness(abc.ABC):
    """An object that evaluates the solution smoothness for each element.
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the detector.
        """

    @abc.abstractmethod
    def get_smoothness_values(self, scheme: SpatialScheme) -> np.ndarray:
        """Get the values of smoothness for each element.
        """


if __name__ == '__main__':
    pass
