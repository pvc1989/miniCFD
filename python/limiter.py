"""Concrete implementations of limiters.
"""
import numpy as np
from scipy import special

import concept
from spatial import PiecewiseContinuous
import expansion
import integrate


_eps = 1e-8


class SimpleWENO(concept.Limiter):

    def name(self):
        return 'Zhong (2013)'

    def _borrow_expansion(self, this: expansion.Legendre,
            that: expansion.Legendre) -> expansion.Legendre:
        borrowed = expansion.Legendre(this.degree(), this.x_left(),
            this.x_right(), this._value_type)
        that_average = integrate.fixed_quad_global(
            lambda x: that.get_function_value(x),
            this.x_left(), this.x_right(), this.degree()) / this.length()
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
            beta += norms[k] * taylor.length()**(2*k-1)
            beta /= (special.factorial(k))**2
        return beta

    def get_new_coeff(self, curr: concept.Element, neighbors) -> np.ndarray:
        candidates = []
        candidates.append(curr.get_expansion())
        for neighbor in neighbors:
            candidates.append(self._borrow_expansion(curr.get_expansion(),
                neighbor.get_expansion()))
        # evaluate weights for each candidate
        w_small = 0.001
        linear_weights = (1-w_small*2, w_small, w_small)
        weights = np.ndarray(3)
        weights_sum = 0.0
        for i in range(len(candidates)):
            smoothness = self._get_smoothness_value(candidates[i])
            weights[i] = linear_weights[i] / (1e-6 + smoothness)**2
            weights_sum += weights[i]
        weights /= weights_sum
        # weight the coeffs
        coeff  = candidates[0].get_coeff() * weights[0]
        for i in range(1, len(candidates)):
            coeff += candidates[i].get_coeff() * weights[i]
        return coeff

    def reconstruct(self, scheme: PiecewiseContinuous, troubled_cell_indices):
        new_coeffs = []
        for i_curr in range(len(troubled_cell_indices)):
            if not troubled_cell_indices[i_curr]:
                continue
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
        i_new = 0
        for i_curr in range(len(troubled_cell_indices)):
            if troubled_cell_indices[i_curr]:
                curr = scheme.get_element_by_index(i_curr)
                curr.set_solution_coeff(new_coeffs[i_new])
                i_new += 1
        assert i_new == len(new_coeffs)


if __name__ == '__main__':
    pass
