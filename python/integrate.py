"""Some frequently used functions for or using Gauss quadratures.
"""
import numpy as np
from scipy import special

import concept


def fixed_quad_global(function: callable, x_left, x_right, n_point=5):
    x_center = (x_right + x_left) / 2
    jacobian = (x_right - x_left) / 2
    def integrand(x_local):
        x_global = x_center + jacobian * x_local
        return function(x_global)
    return jacobian * fixed_quad_local(integrand, n_point)

def fixed_quad_local(function: callable, n_point=5):
    roots, weights = special.roots_legendre(n_point)
    value = weights[0] * function(roots[0])
    for i in range(1, len(roots)):
        value += weights[i] * function(roots[i])
    return value


def average(function: callable, cell: concept.Element):
    value = fixed_quad_global(lambda x_global: function(x_global),
        cell.x_left(), cell.x_right(), cell.degree())
    value /= cell.length()
    return value


def norm_1(function: callable, cell: concept.Element):
    value = fixed_quad_global(lambda x_global: np.abs(function(x_global)),
        cell.x_left(), cell.x_right(), cell.degree())
    return value


def norm_2(function: callable, cell: concept.Element):
    value = fixed_quad_global(lambda x_global: np.abs(function(x_global))**2,
        cell.x_left(), cell.x_right(), cell.degree())
    return np.sqrt(value)


def norm_infty(function: callable, cell: concept.Element):
    value = 0.0
    points = cell.get_quadrature_points(cell.degree())
    for x_global in points:
        value = max(value, np.abs(function(x_global)))
    return value


if __name__ == '__main__':
    pass
