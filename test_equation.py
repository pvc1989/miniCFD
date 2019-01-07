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
    U = euler.primitive_to_conservative(u=0.1, p=0.2, rho=0.3)
    u, p, rho = euler.conservative_to_primitive(U)
    self.assertEqual(euler.primitive_to_conservative(u=u, p=p, rho=rho).all(),
                     U.all())
    self.assertEqual(euler.A(U).dot(U).all(), euler.F(U).all())


if __name__ == '__main__':
    unittest.main()
