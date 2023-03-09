"""Implement some special polynomials.
"""
import numpy as np
from scipy.special import legendre

from concept import Polynomial


class Radau(Polynomial):
    """The left- and right- Radau polynomials.
    """
    _legendres = []
    for k in range(10):
        _legendres.append(legendre(k))

    def __init__(self, degree: int) -> None:
        super().__init__()
        assert 1 <= degree <= 9
        self._k = degree

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


class Vincent(Polynomial):
    """The left- and right- g(ξ) in Vincent's ESFR schemes.

    The right g(ξ) for a k-degree u^h is defined as
        (P_{k} + prev_ratio * P_{k-1} + next_ratio * P_{k+1}) / 2
    """
    _legendres = []
    for k in range(10):
        _legendres.append(legendre(k))

    @staticmethod
    def discontinuous_galerkin(k: int):
        return 1.0

    @staticmethod
    def spectral_difference(k: int):
        return (k + 1) / (2*k + 1)

    @staticmethod
    def huyhn_lump_lobatto(k: int):
        return k / (2*k + 1)

    def __init__(self, degree: int, next_ratio=huyhn_lump_lobatto) -> None:
        super().__init__()
        assert 0 <= degree <= 9
        self._k = degree  # degree of solution, not the polynomial
        self._next_ratio = next_ratio(degree)
        self._prev_ratio = 1 - self._next_ratio

    def get_function_value(self, x_local):
        def right(xi):
            val = self._legendres[self._k](xi)
            val += self._prev_ratio * self._legendres[self._k - 1](xi)
            val += self._next_ratio * self._legendres[self._k + 1](xi)
            return val / 2
        return (right(-x_local), right(x_local))

    def get_gradient_value(self, x_local):
        legendre_derivative_prev = 0.0
        legendre_derivative_curr = 0.0
        legendre_derivative_next = 0.0
        for curr in range(1 + self._k):
            if curr > 0:
                legendre_derivative_prev = legendre_derivative_curr
                legendre_derivative_curr = legendre_derivative_next
            next = curr + 1
            legendre_derivative_next = (next * self._legendres[curr](x_local)
                + x_local * legendre_derivative_curr)
        left = (-1)**self._k * (legendre_derivative_curr
            - self._prev_ratio * legendre_derivative_prev
            - self._next_ratio * legendre_derivative_next) / 2
        right = (legendre_derivative_curr
            + self._prev_ratio * legendre_derivative_prev
            + self._next_ratio * legendre_derivative_next) / 2
        return (left, right)


class IthLagrange(Polynomial):
    """The Ith Lagrange polynomial for N given points.
    """

    def __init__(self, index: int, points: np.ndarray) -> None:
        self._i = index
        assert 0 <= index < len(points)
        self._points = points

    def n_point(self):
        return len(self._points)

    def get_function_value(self, point: float):
        value = 1.0
        for j in range(self.n_point()):
            if j == self._i:
                continue
            dividend = point - self._points[j]
            divisor = self._points[self._i] - self._points[j]
            value *= (dividend / divisor)
        return value

    def get_gradient_value(self, point: float):
        value = 0.0
        for j in range(self.n_point()):
            if j == self._i:
                continue
            dividend = 1.0
            divisor = self._points[self._i] - self._points[j]
            for k in range(self.n_point()):
                if k in (self._i, j):
                    continue
                dividend *= point - self._points[k]
                divisor *= self._points[self._i] - self._points[k]
            value += dividend / divisor
        return value


class LagrangeBasis(Polynomial):
    """All Lagrange polynomials, which form a basis.
    """

    def __init__(self, points: np.ndarray) -> None:
        super().__init__()
        n_point = len(points)
        self._points = points
        self._lagranges = np.ndarray(n_point, IthLagrange)
        for i in range(n_point):
            self._lagranges[i] = IthLagrange(i, self._points)

    def n_term(self):
        return len(self._points)

    def get_function_value(self, x_local):
        values = np.ndarray(self.n_term())
        for i in range(self.n_term()):
            values[i] = self._lagranges[i].get_function_value(x_local)
        return values

    def get_gradient_value(self, x_local):
        values = np.ndarray(self.n_term())
        for i in range(self.n_term()):
            values[i] = self._lagranges[i].get_gradient_value(x_local)
        return values


if __name__ == '__main__':
    pass
