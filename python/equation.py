import numpy as np

import concept
import gas


class ConservationLaw(concept.Equation):
    # \pdv{U}{t} + \pdv{F}{x} = 0

    def get_diffusive_coeff(self, u=0.0):
        return 0

    def get_diffusive_flux(self, u, du_dx):
        return self.get_diffusive_coeff(u) * du_dx

    def get_source(self, u):
        return u * 0


class LinearAdvection(ConservationLaw):
    # ∂u/∂t + a * ∂u/∂x = 0

    def __init__(self, a_const):
        self._a = a_const

    def name(self, verbose=True) -> str:
        my_name = r'$\partial u/\partial t$+' + f'{self._a} '
        my_name += r'$\partial u/\partial x=0$'
        return my_name

    def get_convective_flux(self, u):
        return self._a * u

    def get_convective_speed(self, u=0.0):
        return self._a


class LinearAdvectionDiffusion(LinearAdvection):
    # ∂u/∂t + a * ∂u/∂x = (∂/∂x)(b * ∂u/∂x)

    def __init__(self, a_const, b_const):
        super().__init__(a_const)
        self._b = b_const

    def name(self, verbose=True) -> str:
        my_name = r'$\partial u/\partial t+$' + f'{self._a} '
        my_name += r'$\partial u/\partial x=$' + f'{self._b} '
        my_name += r'$\partial^2 u/\partial x^2$'
        return my_name

    def get_diffusive_coeff(self, u=0.0):
        return self._b


class InviscidBurgers(ConservationLaw):
    # ∂u/∂t + k * u * ∂u/∂x = 0

    def __init__(self, k=1.0):
        assert k > 0.0
        self._k = k

    def name(self, verbose=True) -> str:
        my_name = r'$\partial u/\partial t+$' + f'{self._k} u '
        my_name += r'$\partial u/\partial x=0$'
        return my_name

    def get_convective_flux(self, u):
        return self._k * u**2 / 2

    def get_convective_speed(self, u):
        return self._k * u


class Burgers(InviscidBurgers):
    # ∂u/∂t + k * ∂u/∂x = (∂/∂x)(ν * ∂u/∂x)

    def __init__(self, k_const=1.0, nu_const=0.0):
        assert k_const > 0.0 and nu_const >= 0.0
        super().__init__(k_const)
        self._nu = nu_const

    def name(self, verbose=True) -> str:
        my_name = r'$\partial u/\partial t$+' + f'{self._k} u '
        my_name += r'$\partial u/\partial x=$' + f'{self._nu} '
        my_name += r'$\partial^2 u/\partial x^2$'
        return my_name

    def get_diffusive_coeff(self, u=0.0):
        return self._nu


class LinearSystem(ConservationLaw):

    def __init__(self, A_const):
        assert A_const.shape[0] == A_const.shape[1]
        self._A = A_const

    def name(self, verbose=True) -> str:
        return "LinearSystem"

    def get_convective_flux(self, U):
        return self._A.dot(U)

    def get_convective_speed(self, U):
        return self._A


class Euler(ConservationLaw):

    def __init__(self, gamma=1.4):
        self._gas = gas.Ideal(gamma)

    def name(self, verbose=True) -> str:
        return "Euler"

    def primitive_to_conservative(self, rho, u, p):
        U = np.array([rho, rho*u, 0.0])
        U[2] = u*U[1]/2 + p/self._gas.gamma_minus_1()
        # print(rho, u, p, '->', U)
        return U

    def conservative_to_primitive(self, U):
        rho = U[0]
        if rho == 0:
            assert U[1] == U[2] == 0, U
            u = 0
        else:
            u = U[1] / U[0]
        p = (U[2] - u*U[1]/2) * self._gas.gamma_minus_1()
        # print(U, '->', rho, u, p)
        return rho, u, p

    def get_convective_flux(self, U):
        rho, u, p = self.conservative_to_primitive(U)
        F = U * u
        F[1] += p
        F[2] += p*u
        return F

    def get_convective_speed(self, U):
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
