"""Test some Riemann solvers.
"""
import unittest

import numpy as np
from numpy.random import rand

import riemann
import equation


class TestLinearAdvection(unittest.TestCase):

    def test_get_value(self):
        u_left, u_right = rand(), rand()  # for any u_left, u_right
        solver = riemann.LinearAdvection(a_const=1)
        solver.set_initial(u_left, u_right)
        # x / t < a
        self.assertEqual(solver.get_value(x=-1, t=1), u_left)
        self.assertEqual(solver.get_value(x=0, t=1), u_left)
        # x / t > a
        self.assertEqual(solver.get_value(x=2, t=1), u_right)
        # on x-axis, where t == 0
        self.assertIn(solver.get_value(x=0, t=0), (u_left, u_right))
        self.assertEqual(solver.get_value(x=-1, t=0), u_left)
        self.assertEqual(solver.get_value(x=+1, t=0), u_right)

    def test_get_upwind_flux(self):
        u_left, u_right = rand(), rand()  # for any u_left, u_right
        # right running wave
        pde = equation.LinearAdvection(a_const=+1)
        solver = riemann.LinearAdvection(a_const=+1)
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_left))
        # left running wave
        pde = equation.LinearAdvection(a_const=-1)
        solver = riemann.LinearAdvection(a_const=-1)
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_right))
        # standing wave
        pde = equation.LinearAdvection(a_const=0)
        solver = riemann.LinearAdvection(a_const=0)
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_left))
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_right))


class TestInviscidBurgers(unittest.TestCase):

    def test_right_running_shock(self):
        u_left, u_right = 2, 0
        pde = equation.InviscidBurgers()
        solver = riemann.InviscidBurgers()
        solver.set_initial(u_left, u_right)
        self.assertEqual(solver.get_value(x=-1, t=2), u_left)
        self.assertEqual(solver.get_value(x=+1, t=2), u_left)
        self.assertEqual(solver.get_value(x=+2, t=1), u_right)
        # on x-axis, where t == 0
        self.assertEqual(solver.get_value(x=-1, t=0), u_left)
        self.assertEqual(solver.get_value(x=+1, t=0), u_right)
        self.assertIn(solver.get_value(x=0, t=0), (u_left, u_right))
        # on shock, where x/t == +1
        self.assertIn(solver.get_value(x=1, t=1), (u_left, u_right))
        self.assertIn(solver.get_value(x=2, t=2), (u_left, u_right))
        # on t-axis
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_left))

    def test_left_running_shock(self):
        u_left, u_right = 0, -2
        pde = equation.InviscidBurgers()
        solver = riemann.InviscidBurgers()
        solver.set_initial(u_left, u_right)
        self.assertEqual(solver.get_value(x=+1, t=2), u_right)
        self.assertEqual(solver.get_value(x=-1, t=2), u_right)
        self.assertEqual(solver.get_value(x=-2, t=1), u_left)
        # on x-axis, where t == 0
        self.assertEqual(solver.get_value(x=-1, t=0), u_left)
        self.assertEqual(solver.get_value(x=+1, t=0), u_right)
        self.assertIn(solver.get_value(x=0, t=0), (u_left, u_right))
        # on shock, where x/t == -1
        self.assertIn(solver.get_value(x=-1, t=1), (u_left, u_right))
        self.assertIn(solver.get_value(x=-2, t=2), (u_left, u_right))
        # on t-axis
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_right))

    def test_standing_shock(self):
        u_left, u_right = +1, -1
        pde = equation.InviscidBurgers()
        solver = riemann.InviscidBurgers()
        solver.set_initial(u_left, u_right)
        self.assertEqual(solver.get_value(x=-2, t=1), u_left)
        self.assertEqual(solver.get_value(x=-1, t=2), u_left)
        self.assertEqual(solver.get_value(x=+1, t=2), u_right)
        self.assertEqual(solver.get_value(x=+2, t=1), u_right)
        # on x-axis, where t == 0
        self.assertEqual(solver.get_value(x=-1, t=0), u_left)
        self.assertEqual(solver.get_value(x=+1, t=0), u_right)
        self.assertIn(solver.get_value(x=0, t=0), (u_left, u_right))
        # on shock, where x == 0
        self.assertIn(solver.get_value(x=0, t=1), (u_left, u_right))
        self.assertIn(solver.get_value(x=0, t=2), (u_left, u_right))
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_left))
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_right))

    def test_rarefaction(self):
        u_left, u_right = -1, +1
        solver = riemann.InviscidBurgers()
        solver.set_initial(u_left, u_right)
        # in left const region, where x/t < -1
        self.assertEqual(solver.get_value(x=-2, t=1), u_left)
        # in right const region, where x/t > +1
        self.assertEqual(solver.get_value(x=2, t=1), u_right)
        # on x-axis, where t == 0
        self.assertEqual(solver.get_value(x=-1, t=0), u_left)
        self.assertEqual(solver.get_value(x=+1, t=0), u_right)
        self.assertIn(solver.get_value(x=0, t=0), (u_left, u_right))
        # inside rarefaction, where -1 <= x/t= < +1
        self.assertEqual(solver.get_value(x=-1, t=1), u_left)
        self.assertEqual(solver.get_value(x=+1, t=1), u_right)
        x, t = rand(), 1
        self.assertEqual(solver.get_value(x, t), x/t)
        self.assertEqual(solver.get_upwind_flux(u_left, u_right), 0)

    def test_smooth_initial_condition(self):
        u_left = rand()
        u_right = u_left
        pde = equation.InviscidBurgers()
        solver = riemann.InviscidBurgers()
        self.assertEqual(solver.get_upwind_flux(u_left, u_right),
            pde.get_convective_flux(u_left))
        solver.set_initial(u_left, u_right)
        self.assertEqual(solver.get_value(x=-1, t=1), u_left)
        self.assertEqual(solver.get_value(x=+1, t=1), u_right)
        # on x-axis, where t == 0
        self.assertEqual(solver.get_value(x=-1, t=0), u_left)
        self.assertEqual(solver.get_value(x=+1, t=0), u_right)
        self.assertIn(solver.get_value(x=0, t=0), (u_left, u_right))


class TestCoupled(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_inviscid(self):
        v_0 = riemann.LinearAdvection(1.0)
        v_1 = riemann.InviscidBurgers(1.0)
        R = np.array([[1, 1], [-1, 1]])
        L = np.linalg.inv(R)
        u_solver = riemann.Coupled(v_0, v_1, R)
        for i in range(1000):
            u_left, u_right = rand(2), rand(2)
            v_left, v_right = L @ u_left, L @ u_right
            u_solved = u_solver.get_upwind_flux(u_left, u_right)
            v_solved = np.array([
                v_0.get_upwind_flux(v_left[0], v_right[0]),
                v_1.get_upwind_flux(v_left[1], v_right[1])])
            self.assertAlmostEqual(0, np.linalg.norm((u_solved - R @ v_solved)))


class TestEuler(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._solvers = [
            riemann.Euler(), riemann.Roe(), riemann.LaxFriedrichs()
        ]
        self._equation = equation.Euler()

    def test_consistency(self):
        np.random.seed(31415926)
        for i in range(1000):
            rho = np.random.rand()
            u = np.random.rand()
            p = np.random.rand()
            u_left = self._equation.primitive_to_conservative(rho, u, p)
            u_right = u_left
            for solver in self._solvers:
                assert isinstance(solver, riemann.Euler)
                self.assertAlmostEqual(0.0, np.linalg.norm(
                    solver.get_upwind_flux(u_left, u_right) -
                    solver.equation().get_convective_flux(u_left)))

    def test_upwind_value(self):
        solver = riemann.Roe()
        np.random.seed(31415926)
        for i in range(1000):
            rho = np.random.rand()
            u = np.random.rand()
            p = np.random.rand()
            u_left = self._equation.primitive_to_conservative(rho, u, p)
            f_left = self._equation.get_convective_flux(u_left)
            rho = np.random.rand()
            u = np.random.rand()
            p = np.random.rand()
            u_right = self._equation.primitive_to_conservative(rho, u, p)
            f_right = self._equation.get_convective_flux(u_right)
            u_upwind = solver.get_upwind_value(u_left, u_right)
            a_upwind = self._equation.get_convective_jacobian(u_upwind)
            self.assertAlmostEqual(0.0, np.linalg.norm(
                a_upwind @ (u_right - u_left) - (f_right - f_left)))
            L, R = self._equation.get_convective_eigmats(u_upwind)
            eigvals = self._equation.get_convective_eigvals(u_upwind)
            lambdas = L @ a_upwind @ R
            self.assertAlmostEqual(eigvals[0], lambdas[0][0])
            self.assertAlmostEqual(eigvals[1], lambdas[1][1])
            self.assertAlmostEqual(eigvals[2], lambdas[2][2])
            f_upwind = 0.5 * (f_left + f_right
                - R @ np.abs(lambdas) @ L @ (u_right - u_left))
            self.assertAlmostEqual(0.0, np.linalg.norm(
                f_upwind - solver.get_upwind_flux(u_left, u_right)))


if __name__ == '__main__':
    unittest.main()
