"""Implementations of concrete integrators.
"""
import numpy as np
from scipy import special

import concept


class GaussLegendre(concept.Integrator):

    def get_quadrature_points(self, n_point: int) -> np.ndarray:
        roots, _ = special.roots_legendre(n_point)
        points = np.ndarray(n_point)
        for i in range(n_point):
            points[i] = self._coordinate.local_to_global(roots[i])
        return points

    def fixed_quad_local(self, function: callable, n_point: int):
        roots, weights = special.roots_legendre(n_point)
        value = weights[0] * function(roots[0])
        for i in range(1, len(roots)):
            value += weights[i] * function(roots[i])
        return value


if __name__ == '__main__':
    pass
