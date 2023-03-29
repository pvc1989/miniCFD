"""Concrete implementations of limiters.
"""
import numpy as np
from scipy import special

import concept
from spatial import PiecewiseContinuous
import expansion
import integrate


class CompactWENO(concept.Limiter):
    """A high-order WENO limiter, which is compact (using only immediate neighbors).
    """

    def reconstruct(self, scheme: PiecewiseContinuous, troubled_cell_indices):
        new_coeffs = []
        for i_curr in troubled_cell_indices:
            curr = scheme.get_element_by_index(i_curr)
            neighbors = []
            if i_curr > 0:
                i_prev = i_curr - 1
                prev = scheme.get_element_by_index(i_prev)
                neighbors.append(prev)
            if i_curr + 1 < scheme.n_element():
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


class SimpleWENO(CompactWENO):
    """A high-order compact WENO limiter, which is simple (just borrowing immediate neighbors' expansions).
    """

    def name(self):
        return 'Zhong–Shu (2013)'

    def _borrow_expansion(self, curr: concept.Element,
            neighbor: concept.Element) -> expansion.Legendre:
        this = curr.get_expansion()
        that = neighbor.get_expansion()
        assert isinstance(this, expansion.Taylor)
        assert isinstance(that, expansion.Taylor)
        borrowed = expansion.Legendre(this.degree(), this.x_left(),
            this.x_right(), this._value_type)
        that_average = integrate.average(lambda x: that.get_function_value(x),
            this)
        borrowed.approximate(lambda x: that.get_function_value(x)
            + this.get_average() - that_average)
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
        w_small = 0.001
        weights = np.ndarray(len(candidates))
        for i in range(len(candidates)):
            if i == 0:
                linear_weight = 1 - len(neighbors) * w_small
            else:
                linear_weight = w_small
            smoothness = self._get_smoothness_value(candidates[i])
            weights[i] = linear_weight / (1e-6 + smoothness)**2
        weights /= np.sum(weights)
        # weight the coeffs
        coeff = candidates[0].get_coeff() * weights[0]
        for i in range(1, len(candidates)):
            coeff += candidates[i].get_coeff() * weights[i]
        return coeff


class PWeighted(CompactWENO):
    """A high-order compact WENO limiter, whose linear weights are related to candidates' degrees.
    """

    def name(self):
        return 'Li–Wang–Ren (2020)'

    def _borrow_expansion(self, curr: concept.Element,
            neighbor: concept.Element) -> expansion.Legendre:
        this = curr.get_expansion()
        that = neighbor.get_expansion()
        assert isinstance(this, expansion.Legendre)
        assert isinstance(that, expansion.Taylor)
        borrowed = expansion.Legendre(1, this.x_left(),
            this.x_right(), this._value_type)
        coeff = np.ndarray(2, dtype=this._value_type)
        this_average = this.get_average()
        coeff[0] = this_average
        def psi(x_global):
            return that.get_function_value(x_global) - this_average
        coeff[1] = (integrate.inner_product(this.get_basis(1), psi, neighbor)
            / integrate.norm_2(this.get_basis(1), neighbor))
        borrowed.set_coeff(coeff)
        return borrowed

    def _get_smoothness_value(self, taylor: expansion.Taylor):
        beta = 0.0
        def integrand(x_global):
            return taylor.get_derivative_values(x_global)**2
        norms = integrate.fixed_quad_global(integrand, taylor.x_left(),
            taylor.x_right(), n_point=taylor.degree())
        for k in range(1, taylor.degree()+1):
            scale = taylor.jacobian(0)**(2*k-1) / (special.factorial(k-1))**2
            beta += norms[k] * scale
        return beta

    def get_new_coeff(self, curr: concept.Element, neighbors) -> np.ndarray:
        candidates = []
        # Get candidates by projecting to lower-order spaces:
        for degree in range(curr.degree(), 0, -1):
            candidates.append(
                expansion.TruncatedLegendre(degree, curr.get_expansion()))
        # Get candidates by borrowing from neighbors:
        for neighbor in neighbors:
            candidates.append(self._borrow_expansion(curr, neighbor))
        assert len(candidates) == curr.degree() + len(neighbors)
        # Get smoothness values: (TODO: use quadrature-free implementation)
        smoothness_values = np.ndarray(len(candidates))
        for i in range(len(candidates)):
            smoothness_values[i] = self._get_smoothness_value(candidates[i])
        # Get non-linear weights:
        weights = np.ndarray(len(candidates))
        i_candidate = 0
        for degree in range(curr.degree(), 1, -1):
            linear_weight = 10**(degree+1)
            epsilon = 1e-16
            weights[i_candidate] = linear_weight / (
                smoothness_values[i_candidate]**2 + epsilon)
            i_candidate += 1
        # degree == 1
        beta_left = smoothness_values[i_candidate+1]
        beta_right = smoothness_values[-1]
        epsilon = 1e-16 + 0.1 * (beta_left**2 + beta_right**2) / 2
        weights[i_candidate] = 10 / (
            smoothness_values[i_candidate]**2 + epsilon)
        i_candidate += 1
        assert i_candidate == curr.degree()
        # neighbors
        while i_candidate < len(candidates):
            weights[i_candidate] = 1 / (
                smoothness_values[i_candidate]**2 + epsilon)
            i_candidate += 1
        weights /= np.sum(weights)
        # Average the coeffs:
        new_coeff = np.zeros(curr.degree()+1)
        for i_candidate in range(len(candidates)):
            coeff = candidates[i_candidate].get_coeff()
            new_coeff[0:len(coeff)] += coeff * weights[i_candidate]
        return new_coeff


if __name__ == '__main__':
    pass
