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
        nu, du_dx = rand(), rand()
        self.assertEqual(nu * du_dx,
            advection.get_diffusive_flux(u_given, du_dx, nu))

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
        nu, du_dx = rand(), rand()
        self.assertEqual(nu * du_dx,
            burgers.get_diffusive_flux(u_given, du_dx, nu))

    def test_linear_system(self):
        """Test methods of a LinearSystem object."""
        a_const = rand(3, 3)
        a_const += a_const.transpose()  # gaurantee real eigvals
        system = equation.LinearSystem(a_const)
        u_given = rand(3, 1)
        norm = np.linalg.norm(system.get_convective_flux(u_given)
            - a_const.dot(u_given))
        self.assertEqual(norm, 0.0)
        norm = np.linalg.norm(system.get_convective_jacobian(u_given) - a_const)
        self.assertEqual(norm, 0.0)
        eigvals = system.get_convective_eigvals(u_given)
        norm = np.linalg.norm(eigvals - np.linalg.eigvals(a_const))
        self.assertEqual(norm, 0.0)
        # test A = R * lambdas * L
        lambdas = np.eye(3)
        lambdas[0][0] = eigvals[0]
        lambdas[1][1] = eigvals[1]
        lambdas[2][2] = eigvals[2]
        left, right = system.get_convective_eigmats(u_given)
        norm = np.linalg.norm(np.eye(3) - right @ left)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(np.eye(3) - left @ right)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(a_const @ right - right @ lambdas)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(left @ a_const - lambdas @ left)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(a_const - right @ lambdas @ left)
        self.assertAlmostEqual(norm, 0.0)

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
        eigvals = euler.get_convective_eigvals(given)
        norm = np.linalg.norm(eigvals - np.sort(np.linalg.eigvals(jacobian)))
        self.assertAlmostEqual(norm, 0.0)
        self.assertEqual(
            euler.get_convective_eigvals(given)[1],
            euler.get_convective_speed(given))
        # test A = R * lambdas * L
        lambdas = np.eye(3)
        lambdas[0][0] = eigvals[0]
        lambdas[1][1] = eigvals[1]
        lambdas[2][2] = eigvals[2]
        left, right = euler.get_convective_eigmats(given)
        norm = np.linalg.norm(np.eye(3) - right @ left)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(np.eye(3) - left @ right)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(jacobian @ right - right @ lambdas)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(left @ jacobian - lambdas @ left)
        self.assertAlmostEqual(norm, 0.0)
        norm = np.linalg.norm(jacobian - right @ lambdas @ left)
        self.assertAlmostEqual(norm, 0.0)


if __name__ == '__main__':
    unittest.main()
