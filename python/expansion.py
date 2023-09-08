"""Implement some polynomial approximations for general functions.
"""
import numpy as np
from scipy import special
import numdifftools as nd

import concept
import integrator
import polynomial


class Taylor(concept.Expansion):
    """The Taylor expansion of a general function.

    u^h(x) = \sum_{k=0}^{p} u^{(k)} / (k!) * (x-c)^{k}
    """

    _factorials = np.ones(20)
    for k in range(1, len(_factorials)):
        _factorials[k] = _factorials[k-1] * k

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            Integrator=integrator.GaussLegendre, value_type=float) -> None:
        assert 0 <= degree < 20
        self._n_term = degree + 1
        concept.Expansion.__init__(self, coordinate,
            Integrator(coordinate, self._n_term),
            value_type)
        # coefficients of the Taylor expansion at x_center
        self._taylor_coeff = np.ndarray(self._n_term, value_type)

    def scalar_type(self):
        value = self._taylor_coeff[0]
        if np.isscalar(value):
            return type(value)
        else:
            return type(value[0])

    def name(self, verbose) -> str:
        my_name = 'Taylor'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def n_term(self):
        return self._n_term

    def degree(self):
        return self.n_term() - 1

    def approximate(self, function: callable):
        x_center = self.x_center()
        self._taylor_coeff[0] = function(x_center)
        def kth_df_dx(f, k) -> callable:
            return nd.Derivative(f, n=k, step=self.length()/100, order=4)
        for k in range(1, self.n_term()):
            if issubclass(self.scalar_type(), complex):
                def df_dx(x):
                    real = kth_df_dx(lambda x: function(x).real, k)
                    imag = kth_df_dx(lambda x: function(x).imag, k)
                    return real(x) + 1j * imag(x)
            elif issubclass(self.scalar_type(), float):
                df_dx = kth_df_dx(function, k)
            else:
                assert False
            derivative = df_dx(x_center)
            # print(derivative)
            self._taylor_coeff[k] = derivative / Taylor._factorials[k]

    def average(self):
        def integrand(x_global):
            return self.global_to_value(x_global)
        return self.integrator().average(integrand)

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            return (x_global - self.x_center())**i_basis
        return function

    def get_basis_values(self, x_global):
        x_global -= self.x_center()
        values = np.ndarray(self.n_term())
        x_power = 1.0
        values[0] = 1.0
        for k in range(0, self.n_term()):
            values[k] = x_power
            x_power *= x_global
        return values

    def get_basis_gradients(self, x_global: float):
        x_global -= self.x_center()
        values = np.ndarray(self.n_term())
        x_power = 1.0
        values[0] = 0
        for k in range(1, self.n_term()):
            values[k] = k * x_power
            x_power *= x_global
        return values

    def get_basis_hessians(self, x_global: float):
        x_global -= self.x_center()
        values = np.ndarray(self.n_term())
        x_power = 1.0
        values[0] = 0
        if self.degree() > 0:
            values[1] = 0
        for k in range(2, self.n_term()):
            values[k] = k * (k-1) * x_power
            x_power *= x_global
        return values

    def get_basis_derivatives(self, x_global: float):
        """Get all non-zero-in-general derivatives of the basis.

        values[k][l] = the k-th derivative of (x-c)^{l}.
        """
        x_global -= self.x_center()
        values = np.zeros((self.n_term(), self.n_term()))
        values[0] = self.get_basis_values(x_global)
        for k in range(1, self.n_term()):
            for l in range(k, self.n_term()):
                values[k][l] = x_global**(l-k) * (Taylor._factorials[l]
                    / Taylor._factorials[l-k])
        return values

    def get_basis_innerproducts(self):
        def integrand(x_global):
            column = self.get_basis_values(x_global)
            matrix = np.tensordot(column, column, axes=0)
            return matrix
        mass_matrix = self.integrator().fixed_quad_global(integrand)
        # mass_matrix[i][j] := inner-product of basis[i] and basis[j]
        return mass_matrix

    def global_to_value(self, x_global: float):
        taylor_basis_values = Taylor.get_basis_values(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_values)

    def global_to_gradient(self, x_global: float):
        taylor_basis_gradients = Taylor.get_basis_gradients(self, x_global)
        return self._taylor_coeff.dot(taylor_basis_gradients)

    def global_to_derivatives(self, x_global: float) -> np.ndarray:
        """Get all non-zero-in-general derivatives of u^h.

        values[k] = the k-th derivative of u^h.
        """
        # TODO: evaluate the k-th derivative only
        basis_derivatives = Taylor.get_basis_derivatives(self, x_global)
        values = np.zeros(self.n_term(), self.value_type())
        values[0] = self.global_to_value(x_global)
        for k in range(1, self.n_term()):
            for l in range(k, self.n_term()):
                values[k] += basis_derivatives[k][l] * self._taylor_coeff[l]
        return values

    def set_coeff(self, coeff):
        for i_row in range(self.n_term()):
            self._taylor_coeff[i_row] = coeff[i_row]

    def get_coeff_ref(self):
        return self._taylor_coeff

    def _set_taylor_coeff(self, points: np.ndarray):
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
        this_col = self.get_coeff_ref()
        base_col = np.linalg.solve(mat_a, this_col)
        Taylor.set_coeff(self, base_col)

    def set_taylor_coeff(self, taylor_coeff: np.ndarray):
        """Update the Expansion by setting its Taylor coeff.
        """
        Taylor.set_coeff(self, taylor_coeff)

    def convert_to(self, Expansion):
        assert issubclass(Expansion, Taylor)
        that = Expansion(self.degree(), self.coordinate(), self.value_type())
        that.approximate(lambda x: self.global_to_value(x))
        return that


class Lagrange(Taylor):
    """The Lagrange expansion of a general function.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            get_local_points: callable, Integrator, value_type=float) -> None:
        Taylor.__init__(self, degree, coordinate, Integrator, value_type)
        n_point = degree + 1
        assert n_point >= 1
        roots = get_local_points(n_point)
        # Build the basis and sample points.
        self._basis = polynomial.LagrangeBasis(roots)
        self._sample_values = np.ndarray(n_point, value_type)
        assert issubclass(Integrator, concept.Integrator)
        if get_local_points is Integrator.get_local_points:
            self._sample_points = self.integrator().get_global_points()
        else:
            self._sample_points = self.coordinate().local_to_global(roots)
        # base_rows[k] := values of the base (Taylor) basis at sample_points[k]
        base_rows = np.ndarray((self.n_term(), self.n_term()))
        for k in range(self.n_term()):
            x_global = self._sample_points[k]
            base_rows[k] = Taylor.get_basis_values(self, x_global)
        """
        Since
            base_rows * lagrange_to_taylor = lagrange_basis_rows = eye,
        we have
            lagrange_to_taylor = base_rows^{-1},
        which can be used in
            taylor_coeff_col = lagrange_to_taylor * lagrange_coeff_col.
        """
        self._taylor_to_lagrange = base_rows
        self._lagrange_to_taylor = np.linalg.inv(base_rows)
        self._cached_average = None

    def _update_cached_average(self):
        self._cached_average = Taylor.average(self)

    def average(self):
        return self._cached_average

    def name(self, verbose) -> str:
        my_name = 'Lagrange'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_sample_points(self) -> np.ndarray:
        """Get the global coordinates of all sample points."""
        return self._sample_points

    def get_sample_values(self) -> np.ndarray:
        """Get the sampled values on each sample point."""
        return self.get_coeff_ref()

    def _set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.
        """
        lagrange_col = self.get_coeff_ref()
        taylor_col = self._lagrange_to_taylor.dot(lagrange_col)
        Taylor.set_coeff(self, taylor_col)

    def set_taylor_coeff(self, taylor_coeff: np.ndarray):
        Taylor.set_coeff(self, taylor_coeff)
        lagrange_coeff = self._taylor_to_lagrange @ taylor_coeff
        for i in range(len(lagrange_coeff)):
            self._sample_values[i] = lagrange_coeff[i]

    def set_coeff(self, values):
        """Set values at sample points.

        Users are responsible to ensure (for each i) values[i] is sampled at self._sample_points[i].
        """
        assert len(values) == self._basis.n_term()
        for i in range(len(values)):
            self._sample_values[i] = values[i]
        self._set_taylor_coeff()
        self._update_cached_average()

    def get_coeff_ref(self):
        return self._sample_values

    def approximate(self, function):
        for i in range(self._basis.n_term()):
            self._sample_values[i] = function(self._sample_points[i])
        self._set_taylor_coeff()
        self._update_cached_average()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.coordinate().global_to_local(x_global)
            return self._basis[i_basis].local_to_value(x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.coordinate().global_to_local(x_global)
        values = self._basis.local_to_value(x_local)
        return values

    def get_basis_gradients(self, x_global):
        """
        Since
            taylor_basis_row * lagrange_to_taylor = lagrange_basis_row,
        we have
            taylor_gradient_row * lagrange_to_taylor = lagrange_gradient_row.
        """
        taylor_gradients = Taylor.get_basis_gradients(self, x_global)
        return taylor_gradients.dot(self._lagrange_to_taylor)

    def get_basis_hessians(self, x_global: float):
        taylor_hessians = Taylor.get_basis_hessians(self, x_global)
        return taylor_hessians.dot(self._lagrange_to_taylor)

    def get_basis_derivatives(self, x_global: float):
        taylor_derivatives = Taylor.get_basis_derivatives(self, x_global)
        return taylor_derivatives.dot(self._lagrange_to_taylor)


class LagrangeOnUniformRoots(Lagrange):
    """Specialized Lagrange expansion using uniform roots as nodes.
    """

    @staticmethod
    def _get_local_roots(n_point):
        # Sample points evenly distributed in the element.
        delta = 2.0 / n_point / 2
        roots = np.linspace(delta - 1, 1 - delta, n_point)
        return roots

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        get_local_points = LagrangeOnUniformRoots._get_local_roots
        Lagrange.__init__(self, degree, coordinate, get_local_points,
            integrator.GaussLegendre, value_type)

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnUniformRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LagrangeOnGaussPoints(Lagrange):
    """Specialized Lagrange expansion using Gauss points as nodes.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            Integrator, value_type=float) -> None:
        n_point = degree + 1
        assert issubclass(Integrator, concept.Integrator)
        Lagrange.__init__(self, degree, coordinate, Integrator.get_local_points,
            Integrator, value_type)
        assert self._sample_points is self.integrator().get_global_points()
        self._sample_weights = self.integrator().get_global_weights()

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnGaussPoints'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_sample_weight(self, k: int) -> float:
        """Get the gloabl quadrature weight of the kth node.
        """
        return self._sample_weights[k]

    def _update_cached_average(self):
        values = self.get_sample_values()
        value = values[0] * self.get_sample_weight(0)
        for i in range(1, self.n_term()):
            value += values[i] * self.get_sample_weight(i)
        self._cached_average = value / self.coordinate().length()


class LagrangeOnLegendreRoots(LagrangeOnGaussPoints):
    """Specialized Lagrange expansion using Legendre roots as nodes.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        LagrangeOnGaussPoints.__init__(self, degree, coordinate,
            integrator.GaussLegendre, value_type)

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnLegendreRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class LagrangeOnLobattoRoots(LagrangeOnGaussPoints):
    """Specialized Lagrange expansion using Legendre roots as nodes.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        LagrangeOnGaussPoints.__init__(self, degree, coordinate,
            integrator.GaussLobatto, value_type)

    def name(self, verbose) -> str:
        my_name = 'LagrangeOnLobattoRoots'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name


class Legendre(Taylor):
    """Approximate a general function based on Legendre polynomials.
    """

    def __init__(self, degree: int, coordinate: concept.Coordinate,
            value_type=float) -> None:
        Taylor.__init__(self, degree, coordinate,
            integrator.GaussLegendre, value_type)
        self._mode_coeffs = np.ndarray(self._n_term, value_type)
        # Legendre polynoamials are only orthogonal for 1-degree coordinate map,
        # whose Jacobian determinant is constant over [-1, 1].
        assert self.coordinate().jacobian_degree() == 0
        jacobian = self.coordinate().local_to_jacobian(0)
        # Mode weights are defined as the inner-products of each basis,
        # which can be explicitly integrated here:
        self._mode_weights = jacobian * 2 / (2 * np.arange(degree+1) + 1)
        # taylor_basis_row * _legendre_to_taylor = legendre_basis_row
        self._legendre_to_taylor = np.eye(self._n_term)
        for k in range(2, self._n_term):
            self._legendre_to_taylor[1:k+1, k] = ((2*k-1) / k
                * self._legendre_to_taylor[0:k, k-1])
            self._legendre_to_taylor[0:k-1, k] -= ((k-1) / k
                * self._legendre_to_taylor[0:k-1, k-2])
        # Taylor basis is dimensional, but Legendre basis is dimensionless...
        for k in range(1, self._n_term):
            self._legendre_to_taylor[k] /= jacobian**k
        self._taylor_to_legendre = np.linalg.inv(self._legendre_to_taylor)

    def name(self, verbose) -> str:
        my_name = 'Legendre'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def get_mode_weight(self, k):
        """Get the inner-product of the kth basis with itself.
        """
        return self._mode_weights[k]

    def get_mode_energy(self, k):
        """Get the inner-product of the kth component with itself.
        """
        return self._mode_weights[k] * self._mode_coeffs[k]**2

    def average(self):
        return self._mode_coeffs[0]

    def _set_taylor_coeff(self):
        """Transform a Lagrage expansion onto its Taylor basis.
        
        For Legendre basis, there is
            taylor_basis_row * legendre_to_taylor = legendre_basis_row
        So
            taylor_coeff_col = legendre_to_taylor * legendre_coeff_col.
        """
        legendre_coeff_col = self.get_coeff_ref()
        taylor_coeff_col = self._legendre_to_taylor.dot(legendre_coeff_col)
        Taylor.set_coeff(self, taylor_coeff_col)

    def set_taylor_coeff(self, taylor_coeff: np.ndarray):
        Taylor.set_coeff(self, taylor_coeff)
        legendre_coeff = self._taylor_to_legendre @ taylor_coeff
        for i in range(len(legendre_coeff)):
            self._mode_coeffs[i] = legendre_coeff[i]

    def set_coeff(self, coeffs):
        """Set coefficient for each mode.
        """
        assert len(coeffs) == self.n_term()
        for i in range(len(coeffs)):
            self._mode_coeffs[i] = coeffs[i]
        self._set_taylor_coeff()

    def get_coeff_ref(self):
        return self._mode_coeffs

    def approximate(self, function):
        for k in range(self.n_term()):
            def integrand(x_global):
                x_local = self.coordinate().global_to_local(x_global)
                value = special.eval_legendre(k, x_local) * function(x_global)
                return value
            self._mode_coeffs[k] = self.integrator().fixed_quad_global(integrand)
            self._mode_coeffs[k] /= self._mode_weights[k]
        self._set_taylor_coeff()

    def get_basis(self, i_basis: int) -> callable:
        assert 0 <= i_basis
        def function(x_global):
            x_local = self.coordinate().global_to_local(x_global)
            return special.eval_legendre(i_basis, x_local)
        return function

    def get_basis_values(self, x_global):
        x_local = self.coordinate().global_to_local(x_global)
        values = np.ndarray(self.n_term())
        for k in range(self.n_term()):
            values[k] = special.eval_legendre(k, x_local)
        return values

    def get_basis_gradients(self, x_global):
        taylor_gradients = Taylor.get_basis_gradients(self, x_global)
        return taylor_gradients.dot(self._legendre_to_taylor)

    def get_basis_hessians(self, x_global: float):
        taylor_hessians = Taylor.get_basis_hessians(self, x_global)
        return taylor_hessians.dot(self._legendre_to_taylor)

    def get_basis_derivatives(self, x_global: float):
        taylor_derivatives = Taylor.get_basis_derivatives(self, x_global)
        return taylor_derivatives.dot(self._legendre_to_taylor)


class TruncatedLegendre(Taylor):
    """A lower-order view of a higher-order Legendre expansion.
    """

    def __init__(self, degree: int, that: Legendre) -> None:
        assert 0 <= degree <= that.degree()
        assert isinstance(that, Legendre)
        Taylor.__init__(self, degree, that.coordinate(),
            integrator.GaussLegendre, that.value_type())
        n_term = degree + 1
        self._taylor_coeff[:] = that._taylor_coeff[0:n_term]
        self._mode_coeffs = that._mode_coeffs[0:n_term]
        self._legendre_to_taylor = \
            that._legendre_to_taylor[0:n_term, 0:n_term]
        Legendre._set_taylor_coeff(self)

    def name(self, verbose) -> str:
        my_name = 'TruncatedLegendre'
        if verbose:
            my_name += r' ($p=$' + f'{self.degree()})'
        return my_name

    def set_coeff(self, coeff):
        assert False

    def set_taylor_coeff(self, taylor_coeff):
        assert False

    def get_coeff_ref(self):
        return Legendre.get_coeff_ref(self)

    def average(self):
        return Legendre.average(self)

    def get_basis(self, i_basis: int) -> callable:
        return Legendre.get_basis(self, i_basis)

    def get_basis_values(self, x_global):
        return Legendre.get_basis_values(self, x_global)

    def get_basis_gradients(self, x_global: float):
        assert False

    def get_basis_hessians(self, x_global: float):
        assert False

    def get_basis_derivatives(self, x_global: float):
        assert False


if __name__ == '__main__':
    pass
