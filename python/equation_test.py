"""Test various partial differential equations.
"""
import unittest
from numpy.random import rand

import equation


class TestEquations(unittest.TestCase):
    """Test methods for getting fluxes and Jacobians of various PDEs.
    """

    def test_linear_advection(self):
        """Test methods of a LinearAdvection object."""
        a_const = rand()
        advection = equation.LinearAdvection(a_const)
        unknown = rand()
        self.assertEqual(advection.F(unknown), a_const*unknown)
        self.assertEqual(advection.A(unknown), a_const)

    def test_inviscid_burgers(self):
        """Test methods of an InviscidBurgers object."""
        burgers = equation.InviscidBurgers()
        unknown = rand()
        self.assertEqual(burgers.F(unknown), unknown**2/2)
        self.assertEqual(burgers.A(unknown), unknown)

    def test_linear_system(self):
        """Test methods of a LinearSystem object."""
        a_const = rand(3, 3)
        system = equation.LinearSystem(a_const)
        unknown = rand(3, 1)
        self.assertEqual(system.F(unknown).all(), a_const.dot(unknown).all())
        self.assertEqual(system.A(unknown).all(), a_const.all())

    def test_euler_1d(self):
        """Test methods of an Euler1d object."""
        euler = equation.Euler1d(gamma=1.4)
        # If unknown is an array of ints, small number will be rounded to 0.
        u_given, p_given, rho_given = rand(), rand(), rand()
        unknown = euler.u_p_rho_to_U(u_given, p_given, rho_given)
        u_actual, p_actual, rho_actual = euler.U_to_u_p_rho(unknown)
        self.assertEqual(u_actual, u_given)
        self.assertEqual(p_actual, p_given)
        self.assertEqual(rho_actual, rho_given)
        # test unknown-property
        self.assertEqual(euler.A(unknown).dot(unknown).all(), euler.F(unknown).all())


if __name__ == '__main__':
    unittest.main()
