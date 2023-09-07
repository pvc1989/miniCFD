"""Implementations of concrete integrators.
"""
import numpy as np
from scipy import special

import concept


class GaussLegendre(concept.Integrator):

    @staticmethod
    def get_roots_and_weights(n_point):
        return special.roots_legendre(n_point)


class GaussLobatto(concept.Integrator):

    @staticmethod
    def get_roots_and_weights(n_point):
        roots = np.ndarray(n_point)
        weights = np.ndarray(n_point)
        roots[0] = -1
        roots[-1] = 1
        if n_point == 3:
            roots[1] = 0
            weights[1] = 4 / 3
            weights[0] = weights[-1] = 1 / 3
        elif n_point == 4:
            roots[1] = -np.sqrt(1 / 5)
            roots[2] = -roots[1]
            weights[1] = weights[2] = 5 / 6
            weights[0] = weights[-1] = 1 / 6
        elif n_point == 5:
            roots[2] = 0
            weights[2] = 32 / 45
            roots[1] = -np.sqrt(21) / 7
            roots[3] = -roots[1]
            weights[1] = weights[3] = 49 / 90
            weights[0] = weights[-1] = 1 / 10
        elif n_point == 6:
            roots[1] = -np.sqrt((7 - 2 * np.sqrt(7)) / 21)
            roots[4] = -roots[1]
            weights[1] = weights[4] = (14 + np.sqrt(7)) / 30
            roots[2] = -np.sqrt((7 + 2 * np.sqrt(7)) / 21)
            roots[3] = -roots[2]
            weights[2] = weights[3] = (14 - np.sqrt(7)) / 30
            weights[0] = weights[-1] = 1 / 15
        else:
            assert False, f'{n_point} not in (3, 4, 5, 6).'
        return roots, weights


if __name__ == '__main__':
    pass
