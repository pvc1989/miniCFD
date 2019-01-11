import unittest

import numpy as np

import equation


class TestEquations(unittest.TestCase):

  def test_linear_advection(self):
    a = 0.618
    advection = equation.LinearAdvection(a_const=a)
    U = 0.618
    self.assertEqual(advection.F(U), a*U)
    self.assertEqual(advection.A(U), a)

  def test_inviscid_burgers(self):
    burgers = equation.InviscidBurgers()
    U = 0.618
    self.assertEqual(burgers.F(U), U**2/2)
    self.assertEqual(burgers.A(U), U)

  def test_linear_system(self):
    A = np.eye(3)
    system = equation.LinearSystem(A_const=A)
    U = np.random.rand(3, 1)
    self.assertEqual(system.F(U).all(), A.dot(U).all())
    self.assertEqual(system.A(U).all(), A.all())

  def test_euler_1d(self):
    euler = equation.Euler1d(gamma=1.4)
    # If U is an array of ints, small number will be rounded to 0.
    u_given, p_given, rho_given = 0, 0.01, 1
    U = euler.u_p_rho_to_U(u=u_given, p=p_given, rho=rho_given)
    u, p, rho = euler.U_to_u_p_rho(U)
    self.assertEqual(u, u_given)
    self.assertEqual(p, p_given)
    self.assertEqual(rho, rho_given)


if __name__ == '__main__':
    unittest.main()
