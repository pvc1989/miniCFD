"""Concrete implementations of limiters.
"""
import abc
import numpy as np
from copy import deepcopy

import concept
import expansion


def _get_taylor_coeff(candidate: concept.Expansion):
    return expansion.Taylor.get_coeff_ref(candidate)


class CompactWENO(concept.Limiter):
    """A high-order WENO limiter, which is compact (using only immediate neighbors).
    """

    def reconstruct(self, troubled_cell_indices, grid: concept.Grid):
        new_taylor_coeffs = []
        # print('WENO on', troubled_cell_indices)
        for i_cell in troubled_cell_indices:
            cell_i = grid.get_element_by_index(i_cell)
            new_taylor_coeffs.append(self.get_new_taylor_coeff(cell_i))
        assert len(new_taylor_coeffs) == len(troubled_cell_indices)
        i_new = 0
        for i_cell in troubled_cell_indices:
            expansion_i = grid.get_element_by_index(i_cell).expansion()
            assert isinstance(expansion_i, expansion.Taylor)
            expansion_i.set_taylor_coeff(new_taylor_coeffs[i_new])
            i_new += 1
        assert i_new == len(new_taylor_coeffs)

    def get_new_taylor_coeff(self, curr: concept.Element) -> np.ndarray:
        candidates = self.get_candidates(curr)
        if curr.is_scalar():
            return self.get_scalar_taylor_coeff(candidates)
        else:
            assert curr.is_system()
            n_component = curr.equation().n_component()
            left, right = curr.get_convective_eigmats()
            vector_taylor_coeffs = []
            vector_candidates = candidates
            for vector_poly in vector_candidates:
                old_taylor_coeff = _get_taylor_coeff(vector_poly)
                new_taylor_coeff = np.ndarray(old_taylor_coeff.shape,
                                              old_taylor_coeff.dtype)
                for i_term in range(len(new_taylor_coeff)):
                    new_taylor_coeff[i_term] = left @ old_taylor_coeff[i_term]
                vector_taylor_coeffs.append(new_taylor_coeff)
            new_taylor_coeff = np.ndarray((curr.n_term(), n_component),
                curr.scalar_type())
            for i_comp in range(n_component):
                scalar_taylors = []
                for i_candidate in range(len(vector_candidates)):
                    vector_poly = vector_candidates[i_candidate]
                    assert isinstance(vector_poly, concept.Expansion)
                    vector_coeff = vector_taylor_coeffs[i_candidate]
                    n_term = len(vector_coeff)
                    scalar_coeff = np.ndarray(n_term, curr.scalar_type())
                    for i_term in range(n_term):
                        scalar_coeff[i_term] = vector_coeff[i_term][i_comp]
                    scalar_taylor = expansion.Taylor(vector_poly.degree(),
                        curr.coordinate(), curr.scalar_type())
                    scalar_taylor.set_coeff(scalar_coeff)
                    scalar_taylors.append(scalar_taylor)
                scalar_coeff = self.get_scalar_taylor_coeff(scalar_taylors)
                for i_term in range(len(scalar_coeff)):
                    new_taylor_coeff[i_term][i_comp] = scalar_coeff[i_term]
            for i_term in range(len(new_taylor_coeff)):
                new_taylor_coeff[i_term] = right @ new_taylor_coeff[i_term]
            return new_taylor_coeff

    @abc.abstractmethod
    def get_candidates(self, curr: concept.Element) -> list:
        """"""

    @abc.abstractmethod
    def get_scalar_taylor_coeff(self, candidates, curr: concept.Element) -> np.ndarray:
        """"""


class ZhongXingHui2013(CompactWENO):
    """A high-order compact WENO limiter, which is simple (just borrowing immediate neighbors' expansions).

    See [Zhong and Shu, "A simple weighted essentially nonoscillatory limiter for Runge–Kutta discontinuous Galerkin methods", Journal of Computational Physics 232, 1 (2013), pp. 397--415](https://doi.org/10.1016/j.jcp.2012.08.028) for details.
    """

    def __init__(self, epsilon=1e-6, w_small=0.001) -> None:
        self._epsilon = epsilon
        self._w_small = w_small

    def name(self, verbose=False):
        if verbose:
            return 'Zhong–Shu (2013)'
        else:
            return 'ZXH'

    def _borrow_expansion(self, curr: concept.Element,
            that: concept.Expansion) -> expansion.Legendre:
        this = curr.expansion()
        Type = type(this)
        borrowed = Type(this.degree(), this.coordinate(), this.value_type())
        this_average = this.average()
        that_average = this.integrator().average(
            lambda x: that.global_to_value(x))
        borrowed.approximate(lambda x: that.global_to_value(x)
            + this_average - that_average)
        return borrowed

    def _get_smoothness_value(self, taylor: expansion.Taylor):
        beta = 0.0
        def integrand(x_global):
            values = np.ndarray(taylor.degree(), taylor.value_type())
            for d in range(taylor.degree()):
                k = d + 1
                values[d] = taylor.global_to_derivatives(x_global, k)**2
            return values
        norms = taylor.integrator().integrate(integrand)
        for d in range(taylor.degree()):
            k = d + 1
            length = taylor.length()
            scale = length**(2*k-1) / expansion.Taylor._factorials[k]**2
            beta += norms[d] * scale
        return beta

    def get_candidates(self, curr: concept.Element) -> list:
        candidates = []
        candidates.append(curr.expansion())
        for neighbor in curr.neighbor_expansions():
            if neighbor:
                candidates.append(self._borrow_expansion(curr, neighbor))
        return candidates

    def get_scalar_taylor_coeff(self, candidates: list):
        # evaluate weights for each candidate
        weights = np.ndarray(len(candidates))
        for i in range(len(candidates)):
            if i == 0:
                linear_weight = 1 - (len(candidates)-1) * self._w_small
            else:
                linear_weight = self._w_small
            smoothness = self._get_smoothness_value(candidates[i])
            weights[i] = linear_weight / (self._epsilon + smoothness)**2
        weights /= np.sum(weights)
        # weight the coeffs
        coeff = _get_taylor_coeff(candidates[0]) * weights[0]
        for i in range(1, len(candidates)):
            coeff += _get_taylor_coeff(candidates[i]) * weights[i]
        return coeff


class LiWanAi2020(CompactWENO):
    """A high-order compact WENO limiter, whose linear weights are related to candidates' degrees.

    See [Li and Wang and Ren, "A p-weighted limiter for the discontinuous Galerkin method on one-dimensional and two-dimensional triangular grids", Journal of Computational Physics 407 (2020), pp. 109246](https://doi.org/10.1016/j.jcp.2020.109246) for details.
    """

    def __init__(self, epsilon=1e-16, k_epsilon=0.1, k_trunc=1.0) -> None:
        self._epsilon = epsilon
        self._k_epsilon = k_epsilon
        self._k_trunc = k_trunc

    def name(self, verbose=False):
        if verbose:
            return r'Li–Wang–Ren (2020), $K_\mathrm{trunc}=$'+f'{self._k_trunc:g}'
        else:
            return r'LWA($K_\mathrm{trunc}=$'+f'{self._k_trunc:g})'

    def _borrow_expansion(self, curr: concept.Element,
            that: concept.Expansion) -> expansion.Legendre:
        this = curr.expansion()
        borrowed = expansion.Legendre(1, this.coordinate(), this.value_type())
        coeff = np.ndarray(2, this.value_type())
        this_average = this.average()
        coeff[0] = this_average
        def psi(x_that):
            return that.global_to_value(x_that) - this_average
        phi = borrowed.get_basis(1)
        coeff[1] = (that.integrator().inner_product(phi, psi) /
                    that.integrator().norm_2(phi))
        borrowed.set_coeff(coeff)
        return borrowed

    def _get_derivative_norms(self, taylor: expansion.Taylor):
        def integrand(x_global):
            values = np.ndarray(taylor.degree(), taylor.value_type())
            for d in range(taylor.degree()):
                k = d + 1
                values[d] = taylor.global_to_derivatives(x_global, k)**2
            return values
        norms = taylor.integrator().integrate(integrand)
        for d in range(taylor.degree()):
            k = d + 1
            jacobian = taylor.length() / 2
            scale = jacobian**(2*k-1) / expansion.Taylor._factorials[k-1]**2
            norms[d] *= scale
        return norms

    def _get_smoothness_value(self, taylor: expansion.Taylor):
        norms = self._get_derivative_norms(taylor)
        return np.sum(norms[1:])

    def get_candidates(self, curr: concept.Element) -> list:
        candidates = []
        # Get candidates by projecting to lower-order spaces:
        curr_legendre = curr.expansion()
        if not isinstance(curr_legendre, expansion.Legendre):
            curr_legendre = expansion.Legendre(curr.degree(),
                curr.coordinate(), curr.value_type())
            curr_legendre.approximate(lambda x: curr.get_solution_value(x))
        for degree in range(curr.degree(), 0, -1):
            candidates.append(
                expansion.TruncatedLegendre(degree, curr_legendre))
        # Get candidates by borrowing from neighbors:
        for neighbor in curr.neighbor_expansions():
            if neighbor:
                candidates.append(self._borrow_expansion(curr, neighbor))
        return candidates

    def get_scalar_taylor_coeff(self, candidates: list):
        curr_degree = candidates[0].degree()
        # Get smoothness values: (TODO: use quadrature-free implementation)
        # process 1-degree candidates
        i_candidate = curr_degree - 1
        beta_curr = self._get_smoothness_value(candidates[i_candidate])
        assert i_candidate+2 <= len(candidates)  # at least 1 candidate
        beta_left = self._get_smoothness_value(candidates[i_candidate+1])
        beta_right = self._get_smoothness_value(candidates[-1])
        beta_mean = (beta_curr + beta_left + beta_right) / 3
        # Get non-linear weights:
        weights = []
        i_candidate = 0
        for degree in range(curr_degree, 1, -1):  # from p to 2
            norms = self._get_derivative_norms(candidates[i_candidate])
            beta = np.sum(norms[1:])
            linear_weight = 10**(degree+1)
            weights.append(linear_weight / (beta**2 + self._epsilon))
            i_candidate += 1
            if beta_mean > self._k_trunc / (degree-1) * (beta - norms[1]):
                break
        if i_candidate == curr_degree - 1:  # p = 1, no truncation
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
        new_taylor_coeff = np.zeros(curr_degree + 1)
        for i_candidate in range(len(weights)):
            coeff = _get_taylor_coeff(candidates[i_candidate])
            weight = weights[i_candidate] / weight_sum
            new_taylor_coeff[0:len(coeff)] += coeff * weight
        return new_taylor_coeff


class XuXiaoRui2023(CompactWENO):
    """A limiter using min-max compression.
    """

    def __init__(self, alpha=1.0) -> None:
        self._alpha = alpha

    def name(self, verbose=False):
        if verbose:
            return 'Xu (2023, ' + r'$\alpha=$' + f'{self._alpha:g})'
        else:
            return 'XXR'

    def get_new_coeff(self, curr: concept.Element) -> np.ndarray:
        curr_expansion = curr.expansion()
        curr_average = curr_expansion.average()
        u_max = curr_average
        u_min = u_max
        for neighbor in curr.neighbor_expansions():
            if neighbor:
                average = neighbor.average()
                u_max = max(u_max, average)
                u_min = min(u_min, average)
        big_a = self._alpha * min(u_max - curr_average, curr_average - u_min)
        if big_a == 0:
            old_coeff = deepcopy(curr_expansion.get_coeff_ref())
            curr_expansion.approximate(lambda x: curr_average)
            new_coeff = deepcopy(curr_expansion.get_coeff_ref())
            curr_expansion.set_coeff(old_coeff)
        else:
            def monotone(x):
                q = curr_expansion.global_to_value(x) - curr_average
                q /= big_a
                return np.tanh(q)
            monotone_average = curr.integrator().average(monotone, curr.degree())
            def new_expansion(x):
                dividend = (monotone(x) - monotone_average) * big_a
                divisor = 1 + np.abs(monotone_average)
                return curr_average + dividend / divisor
            old_coeff = deepcopy(curr_expansion.get_coeff_ref())
            curr_expansion.approximate(new_expansion)
            new_coeff = deepcopy(curr_expansion.get_coeff_ref())
            curr_expansion.set_coeff(old_coeff)
        return new_coeff


if __name__ == '__main__':
    pass
