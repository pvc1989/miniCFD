class Ideal(object):

  def __init__(self, gamma=1.4, R=8.3144598/29):
    self._gamma = gamma
    self._gamma_plus_1 = gamma + 1
    self._gamma_minus_1 = gamma - 1
    self._gamma_minus_1_over_gamma_plus_1 = (gamma - 1) / (gamma + 1)
    self._gamma_minus_3 = gamma - 3
    self._R = R

  def gamma(self):
    return self._gamma

  def gamma_plus_1(self):
    return self._gamma_plus_1

  def gamma_minus_1(self):
    return self._gamma_minus_1
    
  def gamma_minus_1_over_gamma_plus_1(self):
    return self._gamma_minus_1_over_gamma_plus_1

  def gamma_minus_3(self):
    return self._gamma_minus_3

