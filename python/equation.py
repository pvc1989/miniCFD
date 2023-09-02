import numpy as np

import concept
import gas


class Scalar(concept.Equation):

    def get_convective_jacobian(self, u_given) -> np.ndarray:
        speed = self.get_convective_speed(u_given)
        return np.array([speed])

    def get_convective_eigvals(self, u_given) -> tuple:
        return (self.get_convective_speed(u_given),)

    def component_names(self) -> tuple[str]:
        return ('U',)


class ConservationLaw(concept.Equation):
    # \pdv{U}{t} + \pdv{F}{x} = 0

    def __init__(self, value_type) -> None:
        concept.Equation.__init__(self, value_type)

    def get_diffusive_coeff(self, u=0.0):
        return 0

    def get_diffusive_flux(self, u, du_dx, nu_extra):
        return (self.get_diffusive_coeff(u) + nu_extra) * du_dx

    def get_source(self, u):
        return u * 0


class LinearAdvection(ConservationLaw, Scalar):
    # ∂u/∂t + a * ∂u/∂x = 0

    def __init__(self, a_const, value_type=float):
        ConservationLaw.__init__(self, value_type)
        self._a = a_const

    def n_component(self):
        return 1

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

    def __init__(self, a_const, b_const, value_type=float):
        LinearAdvection.__init__(self, a_const, value_type)
        self._b = b_const

    def name(self, verbose=True) -> str:
        my_name = r'$\partial u/\partial t+$' + f'{self._a} '
        my_name += r'$\partial u/\partial x=$' + f'{self._b} '
        my_name += r'$\partial^2 u/\partial x^2$'
        return my_name

    def get_diffusive_coeff(self, u=0.0):
        return self._b


class InviscidBurgers(ConservationLaw, Scalar):
    # ∂u/∂t + k * u * ∂u/∂x = 0

    def __init__(self, k=1.0):
        ConservationLaw.__init__(self, float)
        assert k > 0.0
        self._k = k

    def n_component(self):
        return 1

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
        InviscidBurgers.__init__(self, k_const, float)
        self._nu = nu_const

    def name(self, verbose=True) -> str:
        my_name = r'$\partial u/\partial t$+' + f'{self._k} u '
        my_name += r'$\partial u/\partial x=$' + f'{self._nu} '
        my_name += r'$\partial^2 u/\partial x^2$'
        return my_name

    def get_diffusive_coeff(self, u=0.0):
        return self._nu


class LinearSystem(ConservationLaw):

    def __init__(self, A_const: np.ndarray):
        ConservationLaw.__init__(self, np.ndarray)
        assert A_const.shape[0] == A_const.shape[1]
        self._A = A_const
        eigvals = np.linalg.eigvals(A_const)
        self._eigvals = (eigvals[0], eigvals[1], eigvals[2])
        names = []
        for i in range(self.n_component()):
            names.append(f'U{i}')
        self._component_names = tuple(names)

    def n_component(self):
        return len(self._A)

    def component_names(self) -> tuple[str]:
        return self._component_names

    def name(self, verbose=True) -> str:
        return "LinearSystem"

    def get_convective_flux(self, U: np.ndarray) -> np.ndarray:
        return self._A.dot(U)

    def get_convective_jacobian(self, U: np.ndarray) -> np.ndarray:
        return self._A

    def get_convective_eigvals(self, u_given) -> np.ndarray:
        return self._eigvals

    def get_convective_speed(self, u_given):
        assert False


class Euler(ConservationLaw):

    def __init__(self, gamma=1.4):
        ConservationLaw.__init__(self, np.ndarray)
        self._gas = gas.Ideal(gamma)

    def n_component(self):
        return 3

    def component_names(self) -> tuple[str]:
        return ('Density', 'MomentumX', 'EnergyStagnationDensity')

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
        assert rho >= 0 and p >= 0
        return rho, u, p

    def primitive_to_total_enthalpy(self, rho, u, p):
        aa = self._gas.gamma() * p / rho
        enthalpy = aa / self._gas.gamma_minus_1()
        return enthalpy + u*u/2

    def get_diffusive_flux(self, U, dx_U, nu_extra):
        """Get the diffusive flux caused by extra viscosity.
        """
        rho, u, p = self.conservative_to_primitive(U)
        dx_rho = dx_U[0]
        dx_u = (dx_U[1] - u * dx_rho) / rho
        mu = rho * nu_extra
        tau = (4.0/3) * mu * dx_u
        dx_e0 = dx_U[2]
        prandtl = 1e100
        q = self._gas.get_heat_flux(rho, dx_rho, u, dx_u, p, dx_e0, mu, prandtl)
        return np.array([0, tau, tau * u - q])

    def get_convective_flux(self, U):
        rho, u, p = self.conservative_to_primitive(U)
        F = U * u
        F[1] += p
        F[2] += p*u
        return F

    def get_convective_jacobian(self, U):
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

    def get_convective_eigvals(self, u_given) -> tuple:
        rho, u, p = self.conservative_to_primitive(u_given)
        a = np.sqrt(self._gas.gamma() * p / rho)
        return (u-a, u, u+a)

    def get_convective_eigmats(self, u_given) -> tuple[np.ndarray, np.ndarray]:
        rho, u, p = self.conservative_to_primitive(u_given)
        aa = self._gas.gamma() * p / rho
        a = np.sqrt(aa)
        right = np.ndarray((3, 3))
        right[0][0] = 1
        right[0][1] = 1
        right[0][2] = 1
        right[1][0] = u - a
        right[1][1] = u
        right[1][2] = u + a
        ua = u * a
        ke = 0.5 * u * u
        h0 = aa / self._gas.gamma_minus_1() + ke
        right[2][0] = h0 - ua
        right[2][1] = ke
        right[2][2] = h0 + ua
        b1 = self._gas.gamma_minus_1() / aa
        b2 = b1 * ke
        left = np.ndarray((3, 3))
        left[0][0] = (b2 + u / a) / 2
        left[0][1] = -(b1 * u + 1 / a) / 2
        left[0][2] = b1 / 2
        left[1][0] = 1 - b2
        left[1][1] = b1 * u
        left[1][2] = -b1
        left[2][0] = (b2 - u / a) / 2
        left[2][1] = -(b1 * u - 1 / a) / 2
        left[2][2] = b1 / 2
        return (left, right)

    def get_convective_speed(self, u_given):
        return u_given[1] / u_given[0]


if __name__ == '__main__':
    pass
