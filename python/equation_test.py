"""Test various partial differential equations.
"""
import unittest
import numpy as np
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
        self.assertEqual(advection.get_convective_flux(unknown), a_const*unknown)
        self.assertEqual(advection.get_convective_jacobian(unknown), a_const)

    def test_inviscid_burgers(self):
        """Test methods of an InviscidBurgers object."""
        burgers = equation.InviscidBurgers()
        unknown = rand()
        self.assertEqual(burgers.get_convective_flux(unknown), unknown**2/2)
        self.assertEqual(burgers.get_convective_jacobian(unknown), unknown)

    def test_linear_system(self):
        """Test methods of a LinearSystem object."""
        a_const = rand(3, 3)
        system = equation.LinearSystem(a_const)
        unknown = rand(3, 1)
        norm = np.linalg.norm(system.get_convective_flux(unknown)
            - a_const.dot(unknown))
        self.assertEqual(norm, 0.0)
        norm = np.linalg.norm(system.get_convective_jacobian(unknown) - a_const)
        self.assertEqual(norm, 0.0)

    def test_euler_1d(self):
        """Test methods of an Euler1d object."""
        euler = equation.Euler1d(gamma=1.4)
        # If unknown is an array of ints, small number will be rounded to 0.
        u_given, p_given, rho_given = rand(), rand(), rand()
        unknown = euler.u_p_rho_to_U(u_given, p_given, rho_given)
        u_actual, p_actual, rho_actual = euler.U_to_u_p_rho(unknown)
        self.assertAlmostEqual(u_actual, u_given)
        self.assertAlmostEqual(p_actual, p_given)
        self.assertAlmostEqual(rho_actual, rho_given)
        # test unknown-property
        norm = np.linalg.norm(euler.get_convective_flux(unknown)
            - euler.get_convective_jacobian(unknown).dot(unknown))
        self.assertAlmostEqual(norm, 0.0)


if __name__ == '__main__':
    unittest.main()
