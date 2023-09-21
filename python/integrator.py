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
        elif n_point == 7:
            points[1] = -0.83022389627856708
            points[2] = -0.46884879347071418
            points[3] = 0
            points[4] = -points[2]
            points[5] = -points[1]
        elif n_point == 8:
            points[1] = -0.87174014850960668
            points[2] = -0.59170018143314218
            points[3] = -0.20929921790247885
            points[4] = -points[3]
            points[5] = -points[2]
            points[6] = -points[1]
        elif n_point == 9:
            points[1] = -0.89975799541146018
            points[2] = -0.67718627951073762
            points[3] = -0.36311746382617827
            points[4] = 0
            points[5] = -points[3]
            points[6] = -points[2]
            points[7] = -points[1]
        elif n_point == 10:
            points[1] = -0.91953390816645864
            points[2] = -0.73877386510550491
            points[3] = -0.47792494981044459
            points[4] = -0.16527895766638698
            points[5] = -points[4]
            points[6] = -points[3]
            points[7] = -points[2]
            points[8] = -points[1]
        else:
            assert False, f'{n_point} not in range(3, 11).'
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
        elif n_point == 7:
            weights[0] = weights[6] = 0.0476190476190476
            weights[1] = weights[5] = 0.2768260473615660
            weights[2] = weights[4] = 0.4317453812098626
            weights[3] = 0.4876190476190476
        elif n_point == 8:
            weights[0] = weights[7] = 0.0357142857142857
            weights[1] = weights[6] = 0.2107042271435061
            weights[2] = weights[5] = 0.3411226924835044
            weights[3] = weights[4] = 0.4124587946587039
        elif n_point == 9:
            weights[0] = weights[8] = 0.0277777777777778
            weights[1] = weights[7] = 0.1654953615608056
            weights[2] = weights[6] = 0.2745387125001618
            weights[3] = weights[5] = 0.3464285109730465
            weights[4] = 0.3715192743764172
        elif n_point == 10:
            weights[0] = weights[9] = 0.0222222222222222
            weights[1] = weights[8] = 0.1333059908510700
            weights[2] = weights[7] = 0.2248893420631265
            weights[3] = weights[6] = 0.2920426836796839
            weights[4] = weights[5] = 0.3275397611838974
        else:
            assert False, f'{n_point} not in range(3, 11).'
        return weights


if __name__ == '__main__':
    pass
