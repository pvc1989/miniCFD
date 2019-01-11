import abc

import numpy as np

import gas


class ConservationLaw(abc.ABC):
  # \pdv{U}{t} + \pdv{F}{x} = 0
  
  @abc.abstractmethod
  def F(self, U):
    pass 

  @abc.abstractmethod
  def A(self, U):
    # A := \pdv{F}{U}
    pass 


class LinearAdvection(ConservationLaw):
  
  def __init__(self, a_const):
    self._a = a_const

  def F(self, U):
    return self._a * U 

  def A(self, U):
    return self._a


class InviscidBurgers(ConservationLaw):
  
  def __init__(self):
    pass

  def F(self, U):
    return U**2 / 2

  def A(self, U):
    return U


class LinearSystem(ConservationLaw):
  
  def __init__(self, A_const):
    assert A_const.shape[0] == A_const.shape[1]
    self._A = A_const

  def F(self, U):
    return self._A.dot(U) 

  def A(self, U):
    return self._A


class Euler1d(ConservationLaw):
  
  def __init__(self, gamma=1.4):
    self._gas = gas.Ideal(gamma)

  def u_p_rho_to_U(self, u, p, rho):
    U = np.array([rho, rho*u, 0])
    U[2] = u*U[1]/2 + p/self._gas.gamma_minus_1()
    return U

  def U_to_u_p_rho(self, U):
    rho = U[0]
    if rho == 0:
      assert U[1] == U[2] == 0, U
      u = 0
    else:
      u = U[1] / U[0]
    p = (U[2] - u*U[1]/2) * self._gas.gamma_minus_1()
    return u, p, rho

  def F(self, U):
    u, p, rho = self.U_to_u_p_rho(U)
    F = U * u
    F[1] += p
    F[2] += p*u
    return F

  def A(self, U):
    A = np.zeros((3, 3))
    A[0][1] = 1.0
    u = U[1] / U[0]
    uu = u**2
    e_kinetic = uu / 2
    A[1][0] = e_kinetic * self._gas.gamma_minus_3()
    A[1][1] = -u * self._gas.gamma_minus_3()
    A[1][2] = self._gas.gamma_minus_1()
    gamma_times_e_total = U[2] / U[0] * self._gas.gamma()
    A[2][0] = u * (uu*self._gas.gamma_minus_1() - gamma_times_e_total)
    A[2][1] = gamma_times_e_total - 3*e_kinetic*self._gas.gamma_minus_1()
    A[2][2] = u * self._gas.gamma()
    return A


if __name__ == '__main__':
  pass
