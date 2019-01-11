import abc

import numpy as np

import equation


class RiemannSolver(abc.ABC):

  def set_initial(self, U_L, U_R):
    self._U_L = U_L
    self._U_R = U_R
    self._solve()
  
  @abc.abstractmethod
  def _solve(self):
    # Determine boundaries of constant regions and elementary waves,
    # as well as the constant states.
    pass

  def U(self, x, t):
    if t == 0:
      if x <= 0:
        return self._U_L
      else:
        return self._U_R
    else:  # t > 0
      return self._U(v=x/t)

  @abc.abstractmethod
  def _U(self, v):
    # v = x / t
    # return the self-similar solution
    pass

  @abc.abstractmethod
  def F(self, U):
    pass

  def F_on_t_axis(self, U_L, U_R):
    self.set_initial(U_L=U_L, U_R=U_R)
    U_on_t_axis = self.U(x=0, t=1)
    # Actually, U(x=0, t=1) returns either U(x=-0, t=1) or U(x=+0, t=1).
    # If the speed of a shock is 0, then U(x=-0, t=1) != U(x=+0, t=1).
    # However, the jump condition guarantees F(U(x=-0, t=1)) == F(U(x=+0, t=1)).
    return self.F(U_on_t_axis)


class LinearAdvection(RiemannSolver):

  def __init__(self, a_const):
    self._a = a_const

  def _solve(self):
    pass

  def _U(self, v):
    if v <= self._a:
      return self._U_L
    else:
      return self._U_R

  def F(self, U):
    return U * self._a


class InviscidBurgers(RiemannSolver):

  def __init__(self):
    self._equation = equation.InviscidBurgers()

  def _solve(self):
    self._v_L, self._v_R = 0, 0
    if self._U_L <= self._U_R:
      # rarefaction
      self._v_L, self._v_R = self._U_L, self._U_R
    else:
      # shock
      v = (self._U_L + self._U_R) / 2
      self._v_L, self._v_R = v, v

  def _U(self, v):
    if v <= self._v_L:
      return self._U_L
    elif v >= self._v_R:
      return self._U_R
    else:  # v_L < v < v_R
      return v

  def F(self, U):
    return U**2 / 2


class Euler(RiemannSolver):

  def __init__(self, gamma=1.4):
    self._gas = gas.Ideal(gamma)
    self._equation = equation.Euler1d(gamma)

  def F(self, U):
    return self._equation.F(U)
     
  def _solve(self):
    pass

  def _U(self, v):
    pass


if __name__ == '__main__':
  euler = equation.Euler1d(gamma=1.4)  
  solver = Euler(gamma=1.4)

  settings = dict()
  # tests in Table 4.1 of Toro[2009], see https://doi.org/10.1007/b79761
  settings['Sod'] = (0.25,
    euler.u_p_rho_to_U(u=0, p=1.0, rho=1.0),
    euler.u_p_rho_to_U(u=0, p=0.1, rho=0.125))
  settings['AlmostVaccum'] = (0.15,
    euler.u_p_rho_to_U(u=-2, p=0.4, rho=1),
    euler.u_p_rho_to_U(u=+2, p=0.4, rho=1))
  settings['BlastWaveFromLeft'] = (0.12,
    euler.u_p_rho_to_U(u=0, p=1000,  rho=1),
    euler.u_p_rho_to_U(u=0, p=0.01, rho=1))
  settings['BlastWaveFromRight'] = (0.035,
    euler.u_p_rho_to_U(u=0, p=0.01, rho=1),
    euler.u_p_rho_to_U(u=0, p=100,  rho=1))
  settings['ShockCollision'] = (0.035, 
    euler.u_p_rho_to_U(u=19.5975,  p=460.894, rho=5.99924),
    euler.u_p_rho_to_U(u=-6.19633, p=46.0950, rho=5.99924))
  # other tests
  settings['Vaccum'] = (0.15,
    euler.u_p_rho_to_U(u=-4, p=0.4, rho=1),
    euler.u_p_rho_to_U(u=+4, p=0.4, rho=1))

  for name, setting in settings.items():
    t = setting[0]
    U_L = setting[1]
    U_R = setting[2]
    try:
      solver.set_initial(U_L, U_R)
    except AssertionError:
      raise
    finally:
      pass
