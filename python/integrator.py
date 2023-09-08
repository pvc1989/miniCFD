"""Implementations of concrete integrators.
"""
import numpy as np
from scipy import special

import concept


class GaussLegendre(concept.Integrator):

    @staticmethod
    def get_local_points(n_point):
        return special.roots_legendre(n_point)[0]

    @staticmethod
    def get_local_weights(n_point):
        return special.roots_legendre(n_point)[1]


class GaussLobatto(concept.Integrator):

    @staticmethod
    def get_local_points(n_point):
        points = np.ndarray(n_point)
        points[0] = -1
        points[-1] = 1
        if n_point == 3:
            points[1] = 0
        elif n_point == 4:
            points[1] = -np.sqrt(1 / 5)
            points[2] = -points[1]
        elif n_point == 5:
            points[1] = -np.sqrt(21) / 7
            points[2] = 0
            points[3] = -points[1]
        elif n_point == 6:
            points[1] = -np.sqrt((7 + 2 * np.sqrt(7)) / 21)
            points[2] = -np.sqrt((7 - 2 * np.sqrt(7)) / 21)
            points[3] = -points[2]
            points[4] = -points[1]
        else:
            assert False, f'{n_point} not in (3, 4, 5, 6).'
        return points

    @staticmethod
    def get_local_weights(n_point):
        weights = np.ndarray(n_point)
        if n_point == 3:
            weights[1] = 4 / 3
            weights[0] = weights[-1] = 1 / 3
        elif n_point == 4:
            weights[1] = weights[2] = 5 / 6
            weights[0] = weights[-1] = 1 / 6
        elif n_point == 5:
            weights[2] = 32 / 45
            weights[1] = weights[3] = 49 / 90
            weights[0] = weights[-1] = 1 / 10
        elif n_point == 6:
            weights[1] = weights[4] = (14 - np.sqrt(7)) / 30
            weights[2] = weights[3] = (14 + np.sqrt(7)) / 30
            weights[0] = weights[-1] = 1 / 15
        else:
            assert False, f'{n_point} not in (3, 4, 5, 6).'
        return weights


if __name__ == '__main__':
    pass
