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


if __name__ == '__main__':
  pass
