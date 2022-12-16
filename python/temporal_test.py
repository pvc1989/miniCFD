import unittest
import numpy as np

from concept import SemiDiscreteSystem
from temporal import ExplicitEuler, SspRungeKutta


class Constant(SemiDiscreteSystem):
    def __init__(self, a_mat) -> None:
        m, n = a_mat.shape
        assert(m == n)
        self._n = n
        self._a = a_mat.copy()
        self._u = np.zeros(n)


    def set_unknown(self, unknown_vec):
        for i in range(self._n):
            self._u[i] = unknown_vec[i]


    def get_unknown(self):
        return self._u


    def get_residual(self):
        residual = self._a.dot(self._u)
        return residual


class TestExplicitEuler(unittest.TestCase):

    def test_update(self):
        n = 10
        a = np.random.rand(n, n)
        sds = Constant(a)
        u_curr = np.random.rand(n)
        sds.set_unknown(u_curr)
        scheme = ExplicitEuler()
        delta_t = 0.01
        scheme.update(sds, delta_t)
        u_expect = u_curr + a.dot(u_curr) * delta_t
        u_actual = sds.get_unknown()
        self.assertFalse(np.linalg.norm(u_expect - u_actual))


class TestSspRungeKutta(unittest.TestCase):

    def euler(self, u, a, dt):
        return u + a.dot(u) * dt

    def test_rk1_update(self):
        n = 10
        a = np.random.rand(n, n)
        sds = Constant(a)
        u_curr = np.random.rand(n)
        sds.set_unknown(u_curr)
        scheme = SspRungeKutta(order=1)
        delta_t = 0.01
        scheme.update(sds, delta_t)
        u_actual = sds.get_unknown()
        u_expect = self.euler(u_curr, a, delta_t)
        self.assertFalse(np.linalg.norm(u_expect - u_actual))

    def test_rk2_update(self):
        n = 10
        a = np.random.rand(n, n)
        sds = Constant(a)
        u_curr = np.random.rand(n)
        sds.set_unknown(u_curr)
        scheme = SspRungeKutta(order=2)
        delta_t = 0.01
        scheme.update(sds, delta_t)
        u_actual = sds.get_unknown()
        u_frac12 = self.euler(u_curr, a, delta_t)
        u_expect = (u_curr + self.euler(u_frac12, a, delta_t)) / 2
        self.assertFalse(np.linalg.norm(u_expect - u_actual))

    def test_rk3_update(self):
        n = 10
        a = np.random.rand(n, n)
        sds = Constant(a)
        u_curr = np.random.rand(n)
        sds.set_unknown(u_curr)
        scheme = SspRungeKutta(order=3)
        delta_t = 0.01
        scheme.update(sds, delta_t)
        u_actual = sds.get_unknown()
        u_frac13 = self.euler(u_curr, a, delta_t)
        u_frac23 = (u_curr * 3 + self.euler(u_frac13, a, delta_t)) / 4
        u_expect = (u_curr + self.euler(u_frac23, a, delta_t) * 2) / 3
        self.assertFalse(np.linalg.norm(u_expect - u_actual))


if __name__ == '__main__':
    unittest.main()
