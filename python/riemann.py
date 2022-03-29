import abc

import numpy as np
from scipy.optimize import fsolve

import equation
import gas


class RiemannSolver(abc.ABC):

  def set_initial(self, U_L, U_R):
    self._U_L = U_L
    self._U_R = U_R
    self._determine_wave_structure()
  
  @abc.abstractmethod
  def _determine_wave_structure(self):
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

  def _determine_wave_structure(self):
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

  def _determine_wave_structure(self):
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
     
  def _determine_wave_structure(self):
    # set states in unaffected regions
    u_L, p_L, rho_L = self._equation.U_to_u_p_rho(self._U_L)
    u_R, p_R, rho_R = self._equation.U_to_u_p_rho(self._U_R)
    assert p_L*p_R != 0, (p_L, p_R)
    self._u_L, self._u_R = u_L, u_R
    self._p_L, self._p_R = p_L, p_R
    self._rho_L, self._rho_R = rho_L, rho_R
    a_L = self._gas.p_rho_to_a(p_L, rho_L)
    a_R = self._gas.p_rho_to_a(p_R, rho_R)
    self._riemann_invariants_L = np.array([
      p_L / rho_L**self._gas.gamma(),
      u_L + 2*a_L/self._gas.gamma_minus_1()])
    self._riemann_invariants_R = np.array([
      p_R / rho_R**self._gas.gamma(),
      u_R - 2*a_R/self._gas.gamma_minus_1()])
    # determine wave heads and tails and states between 1-wave and 3-wave
    if self._exist_vacuum():
      self._p_2 = self._rho_2_L = self._rho_2_R = self._u_2 = np.nan
      self._v_1_L = u_L - a_L
      self._v_3_R = u_R + a_R
      self._v_1_R = self._riemann_invariants_L[1]
      self._v_3_L = self._riemann_invariants_R[1]
      self._v_2 = (self._v_1_R + self._v_3_L) / 2
      assert self._v_1_L < self._v_1_R <= self._v_2 <= self._v_3_L < self._v_3_R
    else:  # no vacuum
      # 2-field: always a contact
      du = u_R - u_L
      func = lambda p: (self._f(p, p_L, rho_L) +
                                  self._f(p, p_R, rho_R) + du)
      roots, infodict, ierror, message = fsolve(
        func,
        fprime=lambda p: (self._f_prime(p, p_L, rho_L) +
                          self._f_prime(p, p_R, rho_R)),
        x0=self._guess(p_L, func),
        full_output=True)
      assert ierror == 1, message
      p_2 = roots[0]
      u_2 = (u_L - self._f(p_2, p_L, rho_L) +
             u_R + self._f(p_2, p_R, rho_R)) / 2
      self._p_2 = p_2
      self._u_2 = u_2
      self._v_2 = u_2
      # 1-field: a left running wave
      rho_2, v_L, v_R = rho_L, u_L, u_L
      if p_2 > p_L:  # shock
        rho_2, v_L, v_R = self._shock(u_2, p_2, u_L, p_L, rho_L)
      elif p_2 < p_L:  # rarefraction
        # riemann-invariant = u + 2*a/(gamma-1)
        a_2 = a_L + (u_L-u_2)/2*self._gas.gamma_minus_1()
        assert a_2 >= 0, a_2
        rho_2 = p_2 / a_2**2 * self._gas.gamma()
        # eigenvalue = u - a
        v_L = u_L - a_L
        v_R = u_2 - a_2
      else:
        pass
      assert v_L <= v_R <= u_2, (p_2, p_L, v_L, v_R, u_2)
      self._rho_2_L = rho_2
      self._v_1_L = v_L
      self._v_1_R = v_R
      # 3-field: a right running wave
      rho_2, v_L, v_R = rho_R, u_R, u_R
      if p_2 > p_R:  # shock
        rho_2, v_L, v_R = self._shock(u_2, p_2, u_R, p_R, rho_R)
      elif p_2 < p_R:  # rarefraction
        # riemann-invariant = u - 2*a/(gamma-1)
        a_2 = a_R - (u_R-u_2)/2*self._gas.gamma_minus_1()
        assert a_2 >= 0, a_2
        rho_2 = p_2 / a_2**2 * self._gas.gamma()
        # eigenvalue = u + a
        v_L = u_2 + a_2
        v_R = u_R + a_R
      else:
        pass
      assert u_2 <= v_L <= v_R, (p_2, p_R, u_2, v_L, v_R)
      self._rho_2_R = rho_2
      self._v_3_L = v_L
      self._v_3_R = v_R
    print('p2 = {0:5f}, u2 = {1:5f}, rho2L = {2:5f}, rho2R = {3:5f}'.format(
      self._p_2, self._u_2, self._rho_2_L, self._rho_2_R))

  def _exist_vacuum(self):
    if self._riemann_invariants_L[1] <= self._riemann_invariants_R[1]:
      # print('Vacuum exists.')
      return True
    else:
      # print('No vacuum.')
      return False

  def _f(self, p_2, p_1, rho_1):
    f = 0.0
    if p_2 > p_1:
      f = (p_2 - p_1) / np.sqrt(rho_1 * self._P(p_1, p_2))
    elif p_2 < p_1:
      power = self._gas.gamma_minus_1() / self._gas.gamma() / 2
      assert p_2/p_1 >= 0, (p_2, p_1, p_2/p_1, p_1/p_2)
      f = (p_2/p_1)**power - 1
      f *= 2*self._gas.p_rho_to_a(p=p_1, rho=rho_1)
      f /= self._gas.gamma_minus_1()
    else:
      pass
    return f

  def _f_prime(self, p_2, p_1, rho_1):
    assert p_1 != 0 or p_2 != 0, (p_1, p_2)
    df = 1.0
    if p_2 > p_1:
      P = self._P(p_1=p_1, p_2=p_2)
      df -= (p_2-p_1) / P / 4 * self._gas.gamma_plus_1()
      df /= np.sqrt(rho_1 * P)
    else:
      df /= np.sqrt(rho_1 * p_1 * self._gas.gamma())
      if p_2 < p_1:
        power = self._gas.gamma_plus_1() / self._gas.gamma() / 2
        df *= (p_1/p_2)**power
      else:
        pass
    return df

  def _P(self, p_1, p_2):
    return (p_1 * self._gas.gamma_minus_1() +
            p_2 * self._gas.gamma_plus_1()) / 2

  def _guess(self, p, func):
    while (func(p) > 0):
      p *= 0.5
    return p

  @staticmethod
  def _shock(u_2, p_2, u_1, p_1, rho_1):
    assert u_2 != u_1
    v = u_1 + (p_2-p_1)/(u_2-u_1)/rho_1    
    assert u_2 != v
    rho_2 = rho_1 * (u_1 - v) / (u_2 - v)
    return rho_2, v, v

  def _U(self, v):
    u, p, rho = 0, 0, 0
    if v < self._v_2:
      if v <= self._v_1_L:
        u, p, rho = self._u_L, self._p_L, self._rho_L 
      elif v > self._v_1_R:
        u, p, rho = self._u_2, self._p_2, self._rho_2_L
        # print(p, rho)
      else:  # v_1_L < v < v_2_R
        a = self._riemann_invariants_L[1] - v
        a *= self._gas.gamma_minus_1_over_gamma_plus_1()
        assert a >= 0, a
        u = v + a
        rho = a**2 / self._gas.gamma() / self._riemann_invariants_L[0]
        rho = rho**self._gas.one_over_gamma_minus_1()
        p = a**2 * rho / self._gas.gamma()
    else:  # v > v_2
      if v >= self._v_3_R:
        u, p, rho = self._u_R, self._p_R, self._rho_R
      elif v <= self._v_3_L:
        u, p, rho = self._u_2, self._p_2, self._rho_2_R
        # print(p, rho)
      else:  # v_3_L < v < v_3_R
        a = v - self._riemann_invariants_R[1]
        a *= self._gas.gamma_minus_1_over_gamma_plus_1()
        assert a >= 0, a
        u = v - a
        rho = a**2 / self._gas.gamma() / self._riemann_invariants_R[0]
        rho = rho**self._gas.one_over_gamma_minus_1()
        p = a**2 * rho / self._gas.gamma()
    return self._equation.u_p_rho_to_U(u, p, rho)


if __name__ == '__main__':
  from matplotlib import pyplot as plt

  euler = equation.Euler1d(gamma=1.4)  
  solver = Euler(gamma=1.4)

  problems = dict()
  # tests in Table 4.1 of Toro[2009], see https://doi.org/10.1007/b79761
  problems['Sod'] = (0.25,
    euler.u_p_rho_to_U(u=0, p=1.0, rho=1.0),
    euler.u_p_rho_to_U(u=0, p=0.1, rho=0.125))
  problems['Lax'] = (0.15,
    euler.u_p_rho_to_U(u=0.698, p=3.528, rho=0.445),
    euler.u_p_rho_to_U(u=0.0, p=0.571, rho=0.5))
  problems['ShockCollision'] = (0.035, 
    euler.u_p_rho_to_U(u=19.5975,  p=460.894, rho=5.99924),
    euler.u_p_rho_to_U(u=-6.19633, p=46.0950, rho=5.99242))
  problems['BlastFromLeft'] = (0.012,
    euler.u_p_rho_to_U(u=0, p=1000,  rho=1),
    euler.u_p_rho_to_U(u=0, p=0.01, rho=1))
  problems['BlastFromRight'] = (0.035,
    euler.u_p_rho_to_U(u=0, p=0.01, rho=1),
    euler.u_p_rho_to_U(u=0, p=100,  rho=1))
  problems['AlmostVacuumed'] = (0.15,
    euler.u_p_rho_to_U(u=-2, p=0.4, rho=1),
    euler.u_p_rho_to_U(u=+2, p=0.4, rho=1))
  # other tests
  problems['Vacuumed'] = (0.1,
    euler.u_p_rho_to_U(u=-4, p=0.4, rho=1),
    euler.u_p_rho_to_U(u=+4, p=0.4, rho=1))

  # range for plot
  x_vec = np.linspace(start=-0.5, stop=0.5, num=1001)

  for name, problem in problems.items():
    U_L = problem[1]
    U_R = problem[2]
    try:
      solver.set_initial(U_L, U_R)
      u_vec = np.zeros(len(x_vec))
      p_vec = np.zeros(len(x_vec))
      rho_vec = np.zeros(len(x_vec))
      for i in range(len(x_vec)):
        x = x_vec[i]
        U = solver.U(x, t=problem[0])
        u, p, rho = euler.U_to_u_p_rho(U)
        u_vec[i], p_vec[i], rho_vec[i] = u, p, rho
    except AssertionError:
      raise
    finally:
      pass
    plt.figure(figsize=(4,5))
    # subplots = (311, 312, 313)
    l = 5.0
    np.savetxt(name+".csv", (x_vec*l+l/2, rho_vec), delimiter=',')
    titles = (r'$\rho(x)$', r'$p(x)$', r'$u(x)$')
    y_data = (rho_vec, p_vec, u_vec)
    for i in range(3):
      plt.subplot(3,1,i+1)
      plt.grid(True)
      plt.title(titles[i])
      plt.plot(x_vec*l+l/2, y_data[i], '.', markersize='3')
    plt.tight_layout()
    # plt.show()
    plt.savefig(name+'.pdf')
