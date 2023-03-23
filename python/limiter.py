"""Concrete implementations of limiters.
"""
import numpy as np
from scipy import special

import concept
from spatial import PiecewiseContinuous
import expansion
import element
import integrate


_eps = 1e-8


def _norm_1(cell: concept.Element):
    value = integrate.fixed_quad_global(
        lambda x_global: np.abs(cell.get_solution_value(x_global)),
        cell.x_left(), cell.x_right(), cell.degree())
    return value


def _norm_infty(cell: concept.Element):
    value = 0.0
    points = cell.get_quadrature_points(cell.degree())
    for x_global in points:
        value = max(value, np.abs(cell.get_solution_value(x_global)))
    return value


class SimpleWENO:

    def name(self):
        return 'Zhong (2013)'

    def borrow_expansion(self, this: expansion.Legendre,
            that: expansion.Legendre) -> expansion.Legendre:
        borrowed = expansion.Legendre(this.degree(), this.x_left(),
            this.x_right(), this._value_type)
        that_average = integrate.fixed_quad_global(
            lambda x: that.get_function_value(x),
            this.x_left(), this.x_right(), this.degree()) / this.length()
        borrowed.approximate(lambda x: that.get_function_value(x)
            + this.get_average() - that_average)
        return borrowed

    def get_smoothness_value(self, taylor: expansion.Taylor):
        beta = 0.0
        def integrand(x_global):
            return taylor.get_derivative_values(x_global)**2
        norms = integrate.fixed_quad_global(integrand, taylor.x_left(),
            taylor.x_right(), n_point=taylor.degree())
        for k in range(1, taylor.degree()+1):
            beta += norms[k] * taylor.length()**(2*k-1)
            beta /= (special.factorial(k))**2
        return beta

    def reconstruct(self, scheme: PiecewiseContinuous, troubled_cell_indices):
        for i_curr in range(len(troubled_cell_indices)):
            if not troubled_cell_indices[i_curr]:
                continue
            candidates = []
            curr = scheme.get_element_by_index(i_curr)
            candidates.append(curr.get_expansion())
            # periodic BC
            i_prev = i_curr - 1
            prev = scheme.get_element_by_index(i_prev)
            candidates.append(self.borrow_expansion(curr.get_expansion(),
                prev.get_expansion()))
            i_next = (i_curr + 1) % scheme.n_element()
            next = scheme.get_element_by_index(i_next)
            candidates.append(self.borrow_expansion(curr.get_expansion(),
                next.get_expansion()))
            # evaluate weights for each candidate
            w_small = 0.001
            linear_weights = (1-w_small*2, w_small, w_small)
            weights = np.ndarray(3)
            weights_sum = 0.0
            for i in range(3):
                smoothness = self.get_smoothness_value(candidates[i])
                weights[i] = linear_weights[i] / (1e-6 + smoothness)**2
                weights_sum += weights[i]
            weights /= weights_sum
            # weight the coeffs
            coeff  = candidates[0].get_coeff() * weights[0]
            coeff += candidates[1].get_coeff() * weights[1]
            coeff += candidates[2].get_coeff() * weights[2]
            curr.set_solution_coeff(coeff)


if __name__ == '__main__':
    pass
