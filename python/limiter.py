"""Concrete implementations of limiters.
"""
import numpy as np
from scipy import special

import concept
from spatial import FiniteElement
import expansion
import integrate


class Dummy(concept.Limiter):
    """A dummy limiter for comparison, which keeps the given object unchanged.
    """

    def name(self):
        return 'Dummy'

    def reconstruct(self, scheme: FiniteElement, troubled_cell_indices):
        pass

    def get_new_coeff(self, curr: concept.Element, neighbors) -> np.ndarray:
        return curr.get_expansion().get_coeff()


class CompactWENO(concept.Limiter):
    """A high-order WENO limiter, which is compact (using only immediate neighbors).
    """

    def reconstruct(self, scheme: FiniteElement, troubled_cell_indices):
        new_coeffs = []
        for i_curr in troubled_cell_indices:
            curr = scheme.get_element_by_index(i_curr)
            neighbors = []
            if scheme.is_periodic() or i_curr > 0:
                i_prev = i_curr - 1
                prev = scheme.get_element_by_index(i_prev)
                neighbors.append(prev)
            if scheme.is_periodic() or i_curr + 1 < scheme.n_element():
                i_next = (i_curr + 1) % scheme.n_element()
                next = scheme.get_element_by_index(i_next)
                neighbors.append(next)
            coeff = self.get_new_coeff(curr, neighbors)
            new_coeffs.append(coeff)
        assert len(new_coeffs) == len(troubled_cell_indices)
        i_new = 0
        for i_curr in troubled_cell_indices:
            curr = scheme.get_element_by_index(i_curr)
            curr.set_solution_coeff(new_coeffs[i_new])
            i_new += 1
        assert i_new == len(new_coeffs)

    def _x_shift(self, this: expansion.Taylor, that: expansion.Taylor):
        x_shift = 0.0
        if this.x_right() < that.x_left() - that.length():  # this << that
            x_shift = that.x_right() - this.x_left()
        elif this.x_left() > that.x_right() + that.length():  # that << this
            x_shift = that.x_left() - this.x_right()
        else:
            assert np.abs((this.x_right() - that.x_left())
                * (this.x_left() - that.x_right())) < 1e-10
        return x_shift


class ZhongShu2013(CompactWENO):
    """A high-order compact WENO limiter, which is simple (just borrowing immediate neighbors' expansions).

    See Zhong and Shu, "A simple weighted essentially nonoscillatory limiter for Runge–Kutta discontinuous Galerkin methods", Journal of Computational Physics 232, 1 (2013), pp. 397--415.
    """

    def __init__(self, epsilon=1e-6, w_small=0.001) -> None:
        self._epsilon = epsilon
        self._w_small = w_small

    def name(self, verbose=False):
        if verbose:
            return 'Zhong–Shu (2013)'
        else:
            return 'Zhong (2013)'

    def _borrow_expansion(self, curr: concept.Element,
            neighbor: concept.Element) -> expansion.Legendre:
        this = curr.get_expansion()
        that = neighbor.get_expansion()
        assert isinstance(this, expansion.Taylor)
        assert isinstance(that, expansion.Taylor)
        if isinstance(this, expansion.Lagrange):
            Expansion = expansion.Lagrange
        elif isinstance(this, expansion.Legendre):
            Expansion = expansion.Legendre
        else:
            assert False
        borrowed = Expansion(this.degree(), this.x_left(), this.x_right(),
            this._value_type)
        x_shift = self._x_shift(this, that)
        that_average = integrate.average(
            lambda x_this: that.get_function_value(x_this + x_shift), this)
        borrowed.approximate(lambda x_this: this.get_average() - that_average
            + that.get_function_value(x_this + x_shift))
        return borrowed

    def _get_smoothness_value(self, taylor: expansion.Taylor):
        beta = 0.0
        def integrand(x_global):
            return taylor.get_derivative_values(x_global)**2
        norms = integrate.fixed_quad_global(integrand, taylor.x_left(),
            taylor.x_right(), n_point=taylor.degree())
        for k in range(1, taylor.degree()+1):
            scale = taylor.length()**(2*k-1) / (special.factorial(k))**2
            beta += norms[k] * scale
        return beta

    def get_new_coeff(self, curr: concept.Element, neighbors) -> np.ndarray:
        candidates = []
        candidates.append(curr.get_expansion())
        for neighbor in neighbors:
            candidates.append(self._borrow_expansion(curr, neighbor))
        # evaluate weights for each candidate
        weights = np.ndarray(len(candidates))
        for i in range(len(candidates)):
            if i == 0:
                linear_weight = 1 - len(neighbors) * self._w_small
            else:
                linear_weight = self._w_small
            smoothness = self._get_smoothness_value(candidates[i])
            weights[i] = linear_weight / (self._epsilon + smoothness)**2
        weights /= np.sum(weights)
        # weight the coeffs
        coeff = candidates[0].get_coeff() * weights[0]
        for i in range(1, len(candidates)):
            coeff += candidates[i].get_coeff() * weights[i]
        return coeff


class LiWangRen2020(CompactWENO):
    """A high-order compact WENO limiter, whose linear weights are related to candidates' degrees.

    See Li and Wang and Ren, "A p-weighted limiter for the discontinuous Galerkin method on one-dimensional and two-dimensional triangular grids", Journal of Computational Physics 407 (2020), pp. 109246.
    """

    def __init__(self, epsilon=1e-16, k_epsilon=0.1, k_trunc=1.0) -> None:
        self._epsilon = epsilon
        self._k_epsilon = k_epsilon
        self._k_trunc = k_trunc

    def name(self, verbose=False):
        if verbose:
            return r'Li–Wang–Ren (2020), $K_\mathrm{trunc}=$'+f'{self._k_trunc:g}'
        else:
            return r'Li (2020, $K_\mathrm{trunc}=$'+f'{self._k_trunc:g})'

    def _borrow_expansion(self, this: expansion.Legendre,
            neighbor: concept.Element) -> expansion.Legendre:
        that = neighbor.get_expansion()
        assert isinstance(that, expansion.Taylor)
        borrowed = expansion.Legendre(1, this.x_left(),
            this.x_right(), this._value_type)
        coeff = np.ndarray(2, dtype=this._value_type)
        this_average = this.get_average()
        coeff[0] = this_average
        x_shift = self._x_shift(this, that)
        def psi(x_that):
            return that.get_function_value(x_that) - this_average
        def phi(x_that):
            return this.get_basis(1)(x_that - x_shift)
        coeff[1] = (integrate.inner_product(phi, psi, neighbor)
            / integrate.norm_2(phi, neighbor))
        borrowed.set_coeff(coeff)
        return borrowed

    def _get_derivative_norms(self, taylor: expansion.Taylor):
        def integrand(x_global):
            return taylor.get_derivative_values(x_global)**2
        norms = integrate.fixed_quad_global(integrand, taylor.x_left(),
            taylor.x_right(), n_point=taylor.degree())
        for k in range(1, taylor.degree()+1):
            scale = taylor.jacobian(0)**(2*k-1) / (special.factorial(k-1))**2
            norms[k] *= scale
        return norms

    def _get_smoothness_value(self, taylor: expansion.Taylor):
        norms = self._get_derivative_norms(taylor)
        return np.sum(norms[1:])

    def get_new_coeff(self, curr: concept.Element, neighbors) -> np.ndarray:
        candidates = []
        # Get candidates by projecting to lower-order spaces:
        curr_legendre = curr.get_expansion()
        if not isinstance(curr_legendre, expansion.Legendre):
            curr_legendre = expansion.Legendre(curr.degree(),
                curr.x_left(), curr.x_right(), curr._value_type)
            curr_legendre.approximate(lambda x: curr.get_solution_value(x))
        for degree in range(curr.degree(), 0, -1):
            candidates.append(
                expansion.TruncatedLegendre(degree, curr_legendre))
        # Get candidates by borrowing from neighbors:
        for neighbor in neighbors:
            candidates.append(self._borrow_expansion(curr_legendre, neighbor))
        assert len(candidates) == curr.degree() + len(neighbors)
        # Get smoothness values: (TODO: use quadrature-free implementation)
        # process 1-degree candidates
        i_candidate = curr.degree() - 1
        beta_curr = self._get_smoothness_value(candidates[i_candidate])
        assert i_candidate+2 <= len(candidates)  # at least 1 candidate
        beta_left = self._get_smoothness_value(candidates[i_candidate+1])
        beta_right = self._get_smoothness_value(candidates[-1])
        beta_mean = (beta_curr + beta_left + beta_right) / 3
        # Get non-linear weights:
        weights = []
        i_candidate = 0
        for degree in range(curr.degree(), 1, -1):  # from p to 2
            norms = self._get_derivative_norms(candidates[i_candidate])
            beta = np.sum(norms[1:])
            linear_weight = 10**(degree+1)
            weights.append(linear_weight / (beta**2 + self._epsilon))
            i_candidate += 1
            if beta_mean > self._k_trunc / (degree-1) * (beta - norms[1]):
                break
        if i_candidate == curr.degree() - 1:  # p = 1, no truncation
            epsilon = self._epsilon + self._k_epsilon * (beta_left**2
                + beta_right**2) / 2
            weights.append(10 / (beta_curr**2 + epsilon))
            i_candidate += 1
            # neighbors
            weights.append(1 / (beta_left**2 + epsilon))
            i_candidate += 1
            if i_candidate < len(candidates):
                weights.append(1 / (beta_right**2 + epsilon))
                i_candidate += 1
            assert i_candidate == len(candidates)
        assert i_candidate == len(weights)
        weight_sum = np.sum(weights)
        # Average the coeffs:
        new_coeff = np.zeros(curr.degree()+1)
        for i_candidate in range(len(weights)):
            coeff = candidates[i_candidate].get_coeff()
            weight = weights[i_candidate] / weight_sum
            new_coeff[0:len(coeff)] += coeff * weight
        # Legendre to Lagrange, if necessary:
        if isinstance(curr.get_expansion(), expansion.Lagrange):
            new_legendre = expansion.Legendre(curr.degree(),
                curr.x_left(), curr.x_right(), curr._value_type)
            new_legendre.set_coeff(new_coeff)
            new_lagrange = expansion.Lagrange(curr.degree(),
                curr.x_left(), curr.x_right(), curr._value_type)
            new_lagrange.approximate(lambda x:
                new_legendre.get_function_value(x))
            new_coeff = new_lagrange.get_coeff()
        else:
            assert isinstance(curr.get_expansion(), expansion.Legendre)
        return new_coeff


class Xu2023(CompactWENO):
    """A limiter using min-max compression.
    """

    def __init__(self, alpha=1.0) -> None:
        self._alpha = alpha

    def name(self):
        return 'Xu (2023, ' + r'$\alpha$' + f'={self._alpha:g})'

    def get_new_coeff(self, curr: concept.Element, neighbors) -> np.ndarray:
        curr_expansion = curr.get_expansion()
        curr_average = curr_expansion.get_average()
        u_max = curr_average
        u_min = u_max
        for cell in neighbors:
            average = cell.get_expansion().get_average()
            u_max = max(u_max, average)
            u_min = min(u_min, average)
        big_a = self._alpha * min(u_max - curr_average, curr_average - u_min)
        if big_a == 0:
            old_coeff = curr_expansion.get_coeff()
            curr_expansion.approximate(lambda x: curr_average)
            new_coeff = curr_expansion.get_coeff()
            curr_expansion.set_coeff(old_coeff)
        else:
            def monotone(x):
                q = curr_expansion.get_function_value(x) - curr_average
                q /= big_a
                return np.tanh(q)
            monotone_average = integrate.average(monotone, curr)
            def new_expansion(x):
                dividend = (monotone(x) - monotone_average) * big_a
                divisor = 1 + np.abs(monotone_average)
                return curr_average + dividend / divisor
            old_coeff = curr_expansion.get_coeff()
            curr_expansion.approximate(new_expansion)
            new_coeff = curr_expansion.get_coeff()
            curr_expansion.set_coeff(old_coeff)
        return new_coeff


if __name__ == '__main__':
    pass
