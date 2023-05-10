"""Test some Riemann solvers.
"""
import unittest

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

if __name__ == '__main__':
    unittest.main()
