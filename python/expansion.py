"""Implement some polynomial approximations for general functions.
"""
import numpy as np
from scipy import special
import numdifftools as nd

from concept import Expansion
import polynomial
import integrate


class Taylor(Expansion):
    """The Taylor expansion of a general function.

    u^h(x) = \sum_{k=0}^{p} u^{(k)} / (k!) * (x-c)^{k}
    """

    def __init__(self, degree: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__()
        assert degree >= 0
        assert x_left < x_right
        self._n_term = degree + 1
        # linear coordinate transform:
        self._x_center = (x_left + x_right) / 2.0
        self._jacobian = (x_right - x_left) / 2.0
        # coefficients of the Taylor expansion at x_center
        self._taylor_coeff = np.ndarray(self._n_term, value_type)
        self._value_type = value_type

    def n_term(self):
        return self._n_term

    def degree(self):
        return self.n_term() - 1

    def jacobian(self, x_global: float):
        """Get the Jacobian value at a given point.
        """
        return self._jacobian

    def x_left(self):
        return self._x_center - self._jacobian

    def x_right(self):
        return self._x_center + self._jacobian

    def length(self):
        return self._jacobian * 2

    def local_to_global(self, x_local):
        """Coordinate transform from local to global.
        """
        return self._x_center + self._jacobian * x_local

    def global_to_local(self, x_global):
        """Coordinate transform from global to local.
        """
        return (x_global - self._x_center) / self._jacobian

    def approximate(self, function: callable):
        self._taylor_coeff[0] = function(self._x_center)
        for k in range(1, self.n_term()):
            df_dx = nd.Derivative(function, n=k, step=self._jacobian/50, order=4)
            derivative = df_dx(self._x_center)
            # print(derivative)
            self._taylor_coeff[k] = derivative / special.factorial(k)

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            return (x_global - self._x_center)**i_basis
        return function

    def get_basis_values(self, x_global):
        x_global -= self._x_center
        values = x_global**np.arange(0, self.n_term(), dtype=int)
        return values

    def get_basis_gradients(self, x_global: float):
        x_global -= self._x_center
        values = np.ndarray(self.n_term())
        values[0] = 0
        for k in range(self.degree()):
            values[k+1] = (k+1) * x_global**k
        return values

    def get_basis_derivatives(self, x_global: float):
        """Get all non-zero-in-general derivatives of the basis.

        values[k][l] = the k-th derivative of (x-c)^{l}.
        """
        x_global -= self._x_center
        values = np.zeros((self.n_term(), self.n_term()))
        for k in range(1, self.n_term()):
            for l in range(k, self.n_term()):
                values[k][l] = x_global**(l-k) * (special.factorial(l)
                    / special.factorial(l-k))
        return values

    def get_function_value(self, x_global: float):
        taylor_basis_values = Taylor.get_basis_values(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_values)

    def get_gradient_value(self, x_global: float):
        taylor_basis_gradients = Taylor.get_basis_values(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_gradients)

    def get_derivative_values(self, x_global: float):
        """Get all non-zero-in-general derivatives of u^h.

        values[k] = the k-th derivative of u^h.
        """
        basis_derivatives = self.get_basis_derivatives(x_global)
        values = np.zeros(self.n_term(), dtype=self._value_type)
        # values[0] = get_function_value(x_global)
        for k in range(1, self.n_term()):
            for l in range(k, self.n_term()):
                values[k] += basis_derivatives[k][l] * self._taylor_coeff[l]
        return values

    def set_coeff(self, coeff):
        self._taylor_coeff[:] = coeff

    def get_coeff(self):
        return self._taylor_coeff

    def set_taylor_coeff(self, points: np.ndarray):
        """Transform another polynomial expansion onto its Taylor basis.

        Store basis values in a row, and coefficients in a column, one has
            base_row = this_row * mat_a => base_col = mat_a^{-1} * this_col.
        """
        assert issubclass(type(self), Taylor)
        assert len(points) == self.n_term()
        this_rows = np.ndarray((self.n_term(), self.n_term()))
        base_rows = np.ndarray((self.n_term(), self.n_term()))
        for k in range(self.n_term()):
            this_rows[k] = self.get_basis_values(points[k])
            base_rows[k] = Taylor.get_basis_values(self, points[k])
        mat_a = np.linalg.solve(this_rows, base_rows)
        this_col = self.get_coeff()
        base_col = np.linalg.solve(mat_a, this_col)
        Taylor.set_coeff(self, base_col)


class Lagrange(Taylor):
    """The Lagrange expansion of a general function.
    """

    def __init__(self, degree: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__(degree, x_left, x_right, value_type)
        n_point = degree + 1
        assert n_point >= 1
        assert x_left < x_right
        # Sample points evenly distributed in the element.
        delta = 0.1
        roots = np.linspace(delta - 1, 1 - delta, n_point)
        # Or, use zeros of special polynomials.
        roots, _ = special.roots_legendre(degree + 1)
        # Build the basis and sample points.
        self._basis = polynomial.LagrangeBasis(roots)
        self._sample_values = np.ndarray(n_point, value_type)
        self._sample_points = self._x_center + roots * self._jacobian

    def get_sample_points(self):
        """Get the global coordinates of all sample points."""
        return self._sample_points

    def set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.

        For Lagrange basis, this_rows will be np.eye(self.n_term()), if
        the i-th row is evaluated at the i-th sample point. So
            base_rows = mat_a => base_col = base_rows^{-1} * this_col.
        """
        base_rows = np.ndarray((self.n_term(), self.n_term()))
        points = self.get_sample_points()
        for k in range(self.n_term()):
            base_rows[k] = Taylor.get_basis_values(self, points[k])
        this_col = self.get_coeff()
        base_col = np.linalg.solve(base_rows, this_col)
        Taylor.set_coeff(self, base_col)

    def set_coeff(self, values):
        """Set values at sample points.

        Users are responsible to ensure (for each i) values[i] is sampled at self._sample_points[i].
        """
        assert len(values) == self._basis.n_term()
        for i in range(len(values)):
            self._sample_values[i] = values[i]
        self.set_taylor_coeff()

    def get_coeff(self):
        return self._sample_values

    def approximate(self, function):
        for i in range(self._basis.n_term()):
            self._sample_values[i] = function(self._sample_points[i])
        self.set_taylor_coeff()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.global_to_local(x_global)
            return self._basis[i_basis].get_function_value(x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.global_to_local(x_global)
        values = self._basis.get_function_value(x_local)
        return values

    def get_basis_gradients(self, x_global):
        x_local = self.global_to_local(x_global)
        values = self._basis.get_gradient_value(x_local)
        values /= self._jacobian
        return values

    def get_function_value(self, x_global):
        values = self.get_basis_values(x_global)
        value = values.dot(self._sample_values)
        return value

    def get_gradient_value(self, x_global):
        basis_grad = self.get_basis_gradients(x_global)
        value = basis_grad.dot(self._sample_values)
        return value


class Legendre(Taylor):
    """Approximate a general function based on Legendre polynomials.
    """

    def __init__(self, degree: int, x_left: float, x_right: float,
            value_type=float) -> None:
        super().__init__(degree, x_left, x_right, value_type)
        self._mode_coeffs = np.ndarray(self._n_term, value_type)
        self._mode_weights = np.ndarray(self._n_term, float)
        for k in range(self._n_term):
            self._mode_weights[k] = self._jacobian * integrate.fixed_quad_local(
                lambda x: (special.eval_legendre(k, x))**2, n_point=k+1)
        # taylor_basis_row * matrix_on_taylor = legendre_basis_row
        self._matrix_on_taylor = np.eye(self._n_term)
        for k in range(2, self._n_term):
            self._matrix_on_taylor[1:k+1, k] = ((2*k-1) / k
                * self._matrix_on_taylor[0:k, k-1])
            self._matrix_on_taylor[0:k-1, k] -= ((k-1) / k
                * self._matrix_on_taylor[0:k-1, k-2])
        # Taylor basis is dimensional, but Legendre basis is dimensionless...
        half_length = (x_right - x_left) / 2
        for k in range(1, self._n_term):
            self._matrix_on_taylor[k] /= half_length**k

    def get_mode_weight(self, k):
        """Get the inner-product of the kth basis with itself.
        """
        return self._mode_weights[k]

    def get_average(self):
        return self._mode_coeffs[0]

    def set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.
        
        For Legendre basis, there is
            taylor_basis_row * matrix_on_taylor = legendre_basis_row
        So
            taylor_coeff_col = matrix_on_taylor * legendre_coeff_col.
        """
        legendre_coeff_col = self.get_coeff()
        taylor_coeff_col = self._matrix_on_taylor.dot(legendre_coeff_col)
        Taylor.set_coeff(self, taylor_coeff_col)

    def set_coeff(self, coeffs):
        """Set coefficient for each mode.
        """
        assert len(coeffs) == self.n_term()
        for i in range(len(coeffs)):
            self._mode_coeffs[i] = coeffs[i]
        self.set_taylor_coeff()

    def get_coeff(self):
        return self._mode_coeffs

    def approximate(self, function):
        for k in range(self.n_term()):
            def integrand(x_local):
                value = special.eval_legendre(k, x_local)
                value *= function(self.local_to_global(x_local))
                return value
            self._mode_coeffs[k] = integrate.fixed_quad_local(integrand,
                self.n_term()) * self._jacobian
            self._mode_coeffs[k] /= self._mode_weights[k]
        self.set_taylor_coeff()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.global_to_local(x_global)
            return special.eval_legendre(i_basis, x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.global_to_local(x_global)
        values = np.ndarray(self.n_term())
        for k in range(self.n_term()):
            values[k] = special.eval_legendre(k, x_local)
        return values

    def get_basis_gradients(self, x_global):
        x_local = self.global_to_local(x_global)
        values = np.ndarray(self.n_term())
        values[0] = 0.0
        for k in range(1, self.n_term()):
            values[k] = (k * special.eval_legendre(k-1, x_local)
                + x_local * values[k-1])
        values /= self.jacobian(x_global)
        return values

    def get_function_value(self, x_global):
        values = self.get_basis_values(x_global)
        value = values.dot(self._mode_coeffs)
        return value

    def get_gradient_value(self, x_global):
        basis_grad = self.get_basis_gradients(x_global)
        value = basis_grad.dot(self._mode_coeffs)
        return value


class TruncatedLegendre(Taylor):
    """A lower-order view of a higher-order Legendre expansion.
    """

    def __init__(self, degree: int, that: Legendre) -> None:
        assert 0 <= degree <= that.degree()
        Taylor.__init__(self, degree, that.x_left(), that.x_right(), that._value_type)
        n_term = degree + 1
        self._taylor_coeff[:] = that._taylor_coeff[0:n_term]
        self._mode_coeffs = that._mode_coeffs[0:n_term]
        self._mode_weights = that._mode_weights[0:n_term]
        self._matrix_on_taylor = that._matrix_on_taylor[0:n_term, 0:n_term]
        Legendre.set_taylor_coeff(self)

    def get_coeff(self):
        return self._mode_coeffs


if __name__ == '__main__':
    pass
