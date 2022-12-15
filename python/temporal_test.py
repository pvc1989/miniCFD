import unittest
import numpy as np

from concept import SemiDiscreteSystem
from temporal import ExplicitEuler


class Constant(SemiDiscreteSystem):
  def __init__(self, a_mat) -> None:
    m, n = a_mat.shape
    assert(m == n)
    self._n = n
    self._a = a_mat
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


if __name__ == '__main__':
    unittest.main()
