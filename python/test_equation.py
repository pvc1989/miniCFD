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
        u_given = rand()
        self.assertEqual(advection.get_convective_flux(u_given), a_const*u_given)
        self.assertEqual(advection.get_convective_jacobian(u_given), a_const)
        self.assertEqual(advection.get_convective_eigvals(u_given)[0], a_const)
        self.assertEqual(advection.get_convective_speed(u_given), a_const)

    def test_inviscid_burgers(self):
        """Test methods of an InviscidBurgers object."""
        k_const = rand()
        burgers = equation.InviscidBurgers(k_const)
        u_given = rand()
        speed = k_const * u_given
        flux = k_const * u_given**2 / 2
        self.assertEqual(burgers.get_convective_flux(u_given), flux)
        self.assertEqual(burgers.get_convective_jacobian(u_given), speed)
        self.assertEqual(burgers.get_convective_eigvals(u_given)[0], speed)
        self.assertEqual(burgers.get_convective_speed(u_given), speed)

    def test_linear_system(self):
        """Test methods of a LinearSystem object."""
        a_const = rand(3, 3)
        system = equation.LinearSystem(a_const)
        u_given = rand(3, 1)
        norm = np.linalg.norm(system.get_convective_flux(u_given)
            - a_const.dot(u_given))
        self.assertEqual(norm, 0.0)
        norm = np.linalg.norm(system.get_convective_jacobian(u_given) - a_const)
        self.assertEqual(norm, 0.0)
        norm = np.linalg.norm(system.get_convective_eigvals(u_given)
                              - np.linalg.eigvals(a_const))
        self.assertEqual(norm, 0.0)

    def test_euler_1d(self):
        """Test methods of an Euler object."""
        euler = equation.Euler(gamma=1.4)
        rho_given, u_given, p_given = rand(), rand(), rand()
        given = euler.primitive_to_conservative(rho_given, u_given, p_given)
        rho_actual, u_actual, p_actual = euler.conservative_to_primitive(given)
        self.assertAlmostEqual(rho_actual, rho_given)
        self.assertAlmostEqual(u_actual, u_given)
        self.assertAlmostEqual(p_actual, p_given)
        # test F(U) == A(U) @ U
        jacobian = euler.get_convective_jacobian(given)
        norm = np.linalg.norm(euler.get_convective_flux(given)
            - jacobian.dot(given))
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(euler.get_convective_eigvals(given)
            - np.sort(np.linalg.eigvals(jacobian)))
        self.assertAlmostEqual(norm, 0.0)
        self.assertEqual(
            euler.get_convective_eigvals(given)[1],
            euler.get_convective_speed(given))


if __name__ == '__main__':
    unittest.main()
