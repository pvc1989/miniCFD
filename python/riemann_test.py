import unittest

from numpy.random import rand

import riemann


class TestLinearAdvection(unittest.TestCase):

    def test_U(self):
        U_L, U_R = rand(), rand()  # for any U_L, U_R
        solver = riemann.LinearAdvection(a_const=1)
        solver.set_initial(U_L, U_R)
        # x / t < a
        self.assertEqual(solver.U(x=-1, t=1), U_L)
        self.assertEqual(solver.U(x=0, t=1), U_L)
        # x / t > a
        self.assertEqual(solver.U(x=2, t=1), U_R)
        # on x-axis, where t == 0
        self.assertIn(solver.U(x=0, t=0), (U_L, U_R))
        self.assertEqual(solver.U(x=-1, t=0), U_L)
        self.assertEqual(solver.U(x=+1, t=0), U_R)

    def test_F(self):
        U_L, U_R = rand(), rand()  # for any U_L, U_R
        # right running wave
        solver = riemann.LinearAdvection(a_const=+1)
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_L))
        # left running wave
        solver = riemann.LinearAdvection(a_const=-1)
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_R))
        # standing wave
        solver = riemann.LinearAdvection(a_const=0)
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_L))
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_R))


class TestInviscidBurgers(unittest.TestCase):

    def test_right_running_shock(self):
        U_L, U_R = 2, 0
        solver = riemann.InviscidBurgers()
        solver.set_initial(U_L, U_R)
        self.assertEqual(solver.U(x=-1, t=2), U_L)
        self.assertEqual(solver.U(x=+1, t=2), U_L)
        self.assertEqual(solver.U(x=+2, t=1), U_R)
        # on x-axis, where t == 0
        self.assertEqual(solver.U(x=-1, t=0), U_L)
        self.assertEqual(solver.U(x=+1, t=0), U_R)
        self.assertIn(solver.U(x=0, t=0), (U_L, U_R))
        # on shock, where x/t == +1
        self.assertIn(solver.U(x=1, t=1), (U_L, U_R))
        self.assertIn(solver.U(x=2, t=2), (U_L, U_R))
        # on t-axis
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_L))

    def test_left_running_shock(self):
        U_L, U_R = 0, -2
        solver = riemann.InviscidBurgers()
        solver.set_initial(U_L, U_R)
        self.assertEqual(solver.U(x=+1, t=2), U_R)
        self.assertEqual(solver.U(x=-1, t=2), U_R)
        self.assertEqual(solver.U(x=-2, t=1), U_L)
        # on x-axis, where t == 0
        self.assertEqual(solver.U(x=-1, t=0), U_L)
        self.assertEqual(solver.U(x=+1, t=0), U_R)
        self.assertIn(solver.U(x=0, t=0), (U_L, U_R))
        # on shock, where x/t == -1
        self.assertIn(solver.U(x=-1, t=1), (U_L, U_R))
        self.assertIn(solver.U(x=-2, t=2), (U_L, U_R))
        # on t-axis
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_R))

    def test_standing_shock(self):
        U_L, U_R = +1, -1
        solver = riemann.InviscidBurgers()
        solver.set_initial(U_L, U_R)
        self.assertEqual(solver.U(x=-2, t=1), U_L)
        self.assertEqual(solver.U(x=-1, t=2), U_L)
        self.assertEqual(solver.U(x=+1, t=2), U_R)
        self.assertEqual(solver.U(x=+2, t=1), U_R)
        # on x-axis, where t == 0
        self.assertEqual(solver.U(x=-1, t=0), U_L)
        self.assertEqual(solver.U(x=+1, t=0), U_R)
        self.assertIn(solver.U(x=0, t=0), (U_L, U_R))
        # on shock, where x == 0
        self.assertIn(solver.U(x=0, t=1), (U_L, U_R))
        self.assertIn(solver.U(x=0, t=2), (U_L, U_R))
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_L))
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_R))

    def test_rarefaction(self):
        U_L, U_R = -1, +1
        solver = riemann.InviscidBurgers()
        solver.set_initial(U_L, U_R)
        # in left const region, where x/t < -1
        self.assertEqual(solver.U(x=-2, t=1), U_L)
        # in right const region, where x/t > +1
        self.assertEqual(solver.U(x=2, t=1), U_R)
        # on x-axis, where t == 0
        self.assertEqual(solver.U(x=-1, t=0), U_L)
        self.assertEqual(solver.U(x=+1, t=0), U_R)
        self.assertIn(solver.U(x=0, t=0), (U_L, U_R))
        # inside rarefaction, where -1 <= x/t= < +1
        self.assertEqual(solver.U(x=-1, t=1), U_L)
        self.assertEqual(solver.U(x=+1, t=1), U_R)
        x, t = rand(), 1
        self.assertEqual(solver.U(x, t), x/t)
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), 0)

    def test_smooth_initial_condition(self):
        U_L = rand()
        U_R = U_L
        solver = riemann.InviscidBurgers()
        self.assertEqual(solver.get_upwind_flux(U_L, U_R), solver.F(U_L))
        solver.set_initial(U_L, U_R)
        self.assertEqual(solver.U(x=-1, t=1), U_L)
        self.assertEqual(solver.U(x=+1, t=1), U_R)
        # on x-axis, where t == 0
        self.assertEqual(solver.U(x=-1, t=0), U_L)
        self.assertEqual(solver.U(x=+1, t=0), U_R)
        self.assertIn(solver.U(x=0, t=0), (U_L, U_R))

if __name__ == '__main__':
    unittest.main()
