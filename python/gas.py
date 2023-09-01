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
        aa = 0.0
        if rho == 0:
            assert p == 0, (p, rho)
        else:
            aa = (p / rho) * self.gamma()
        return aa

    def p_rho_to_a(self, p, rho):
        return np.sqrt(self.p_rho_to_aa(p, rho))

    def get_heat_flux(self, rho, dx_rho, u, dx_u, p, dx_e0, mu, prandtl):
        dx_kinetic = dx_rho * u * u / 2 + rho * u * dx_u
        temp = p / rho * dx_rho * self.one_over_gamma_minus_1()
        # Now, temp == R * T * dx_rho / (gamma - 1)
        temp = (dx_e0 - dx_kinetic - temp) / rho
        # Now, temp == C_v * dx_T / (gamma - 1)
        return -mu / prandtl * self.gamma() * temp
