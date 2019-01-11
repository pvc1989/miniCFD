import numpy as np


class Ideal(object):

  def __init__(self, gamma=1.4, R=8.3144598/29):
    self._R = R
    self._gamma = gamma
    self._gamma_plus_1 = gamma + 1
    self._gamma_minus_1 = gamma - 1
    self._gamma_minus_1_over_gamma_plus_1 = (gamma - 1) / (gamma + 1)
    self._gamma_minus_3 = gamma - 3
    self._1_over_gamma_minus_1 = 1 / self._gamma_minus_1

  def gamma(self):
    return self._gamma

  def gamma_plus_1(self):
    return self._gamma_plus_1

  def gamma_minus_1(self):
    return self._gamma_minus_1
    
  def gamma_minus_1_over_gamma_plus_1(self):
    return self._gamma_minus_1_over_gamma_plus_1

  def one_over_gamma_minus_1(self):
    return self._1_over_gamma_minus_1

  def gamma_minus_3(self):
    return self._gamma_minus_3

  def p_rho_to_aa(self, p, rho):
    if rho == 0:
      assert p == 0, (p, rho)
      return 0
    else:
      return (p / rho) * self.gamma()

  def p_rho_to_a(self, p, rho):
    return np.sqrt(self.p_rho_to_aa(p, rho))
