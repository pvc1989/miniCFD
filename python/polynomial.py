"""Implement some special polynomials.
"""
import numpy as np
from scipy.special import legendre

from concept import Polynomial


class Radau(Polynomial):
    """The left- and right- Radau polynomials.
    """

    def __init__(self, degree: int) -> None:
        super().__init__()
        assert degree >= 1
        self._k = degree
        self._legendres = []
        for k in range(degree + 1):
            self._legendres.append(legendre(k))

    def get_function_value(self, x_local):
        legendre_value_curr = self._legendres[self._k](x_local)
        legendre_value_prev = self._legendres[self._k - 1](x_local)
        left = (legendre_value_curr + legendre_value_prev) / 2
        right = legendre_value_curr - legendre_value_prev
        right *= (-1)**self._k / 2
        return (left, right)

    def get_gradient_value(self, x_local):
        legendre_derivative_prev = 0.0
        legendre_derivative_curr = 0.0
        for k in range(1, self._k + 1):
            legendre_derivative_prev = legendre_derivative_curr
            # prev == k - 1, curr == k
            legendre_derivative_curr = k * self._legendres[k-1](x_local)
            legendre_derivative_curr += x_local * legendre_derivative_prev
        left = (legendre_derivative_curr + legendre_derivative_prev) / 2
        right = legendre_derivative_curr - legendre_derivative_prev
        right *= (-1)**self._k / 2
        return (left, right)


class Lagrange(Polynomial):
    """Lagrange polynomials defined on [-1, 1].
    """

    def __init__(self, n_point: int) -> None:
        super().__init__()
        assert n_point >= 2
        self._n_point = n_point
        self._local_coords = np.linspace(-1.0, 1.0, n_point)

    def get_function_value(self, x_local):
        values = np.zeros(self._n_point)
        for i in range(self._n_point):
            product = 1.0
            for j in range(self._n_point):
                if j == i:
                    continue
                dividend = x_local - self._local_coords[j]
                divisor = self._local_coords[i] - self._local_coords[j]
                product *= (dividend / divisor)
            values[i] = product
        return values

    def get_gradient_value(self, x_local):
        values = np.zeros(self._n_point)
        for i in range(self._n_point):
            for j in range(self._n_point):
                if j == i:
                    continue
                dividend = 1.0
                divisor = self._local_coords[i] - self._local_coords[j]
                for k in range(self._n_point):
                    if k in (i, j):
                        continue
                    dividend *= x_local - self._local_coords[k]
                    divisor *= self._local_coords[i] - self._local_coords[k]
                values[i] += dividend / divisor
        return values


if __name__ == '__main__':
    pass
