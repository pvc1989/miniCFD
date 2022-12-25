"""Tests for some temporal schemes.
"""
import unittest
import numpy as np

from concept import OdeSystem
from temporal import ExplicitEuler, SspRungeKutta


class Constant(OdeSystem):
    """The simplest implementation of OdeSystem.
    """
    def __init__(self, a_mat) -> None:
        n_row, n_col = a_mat.shape
        assert n_row == n_col
        self._n = n_col
        self._a = a_mat.copy()
        self._u = np.zeros(n_col)

    def set_solution_column(self, column):
        for i in range(self._n):
            self._u[i] = column[i]

    def get_solution_column(self):
        return self._u.copy()

    def get_residual_column(self):
        residual = self._a.dot(self._u)
        return residual

    def initialize(self, function):
        pass
        

class TestExplicitEuler(unittest.TestCase):
    """Test the ExplicitEuler class.
    """

    def test_update(self):
        """Test the update() method.
        """
        n_row = 10
        df_du = np.random.rand(n_row, n_row)
        ode_system = Constant(df_du)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = ExplicitEuler()
        delta_t = 0.01
        scheme.update(ode_system, delta_t)
        u_expect = u_curr + df_du.dot(u_curr) * delta_t
        u_actual = ode_system.get_solution_column()
        self.assertFalse(np.linalg.norm(u_expect - u_actual))


class TestSspRungeKutta(unittest.TestCase):
    """Test the SspRungeKutta class.
    """

    def euler(self, solution, df_du, delta_t):
        """Wrap the computation of the explicit Euler method.
        """
        return solution + df_du.dot(solution) * delta_t

    def test_rk1_update(self):
        """Test RK1's update() method.
        """
        n_row = 10
        df_du = np.random.rand(n_row, n_row)
        ode_system = Constant(df_du)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = SspRungeKutta(order=1)
        delta_t = 0.01
        scheme.update(ode_system, delta_t)
        u_actual = ode_system.get_solution_column()
        u_expect = self.euler(u_curr, df_du, delta_t)
        self.assertFalse(np.linalg.norm(u_expect - u_actual))

    def test_rk2_update(self):
        """Test RK2's update() method.
        """
        n_row = 10
        df_du = np.random.rand(n_row, n_row)
        ode_system = Constant(df_du)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = SspRungeKutta(order=2)
        delta_t = 0.01
        scheme.update(ode_system, delta_t)
        u_actual = ode_system.get_solution_column()
        u_frac12 = self.euler(u_curr, df_du, delta_t)
        u_expect = (u_curr + self.euler(u_frac12, df_du, delta_t)) / 2
        self.assertFalse(np.linalg.norm(u_expect - u_actual))

    def test_rk3_update(self):
        """Test RK3's update() method.
        """
        n_row = 10
        df_du = np.random.rand(n_row, n_row)
        ode_system = Constant(df_du)
        u_curr = np.random.rand(n_row)
        ode_system.set_solution_column(u_curr)
        scheme = SspRungeKutta(order=3)
        delta_t = 0.01
        scheme.update(ode_system, delta_t)
        u_actual = ode_system.get_solution_column()
        u_frac13 = self.euler(u_curr, df_du, delta_t)
        u_frac23 = (u_curr * 3 + self.euler(u_frac13, df_du, delta_t)) / 4
        u_expect = (u_curr + self.euler(u_frac23, df_du, delta_t) * 2) / 3
        self.assertFalse(np.linalg.norm(u_expect - u_actual))


if __name__ == '__main__':
    unittest.main()
