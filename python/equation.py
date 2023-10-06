import numpy as np

import concept
import gas


class LinearAdvection(concept.ScalarEquation):
    """ \f$ \partial_t u + a\,\partial_x u = 0 \f$
    """

    def __init__(self, a_const, value_type=float):
        concept.Equation.__init__(self, value_type)
        self._a = a_const

    def name(self, verbose=True) -> str:
        my_name = r'$\partial u/\partial t$+' + f'{self._a} '
        my_name += r'$\partial u/\partial x=0$'
        return my_name

    def get_convective_flux(self, u):
        return self._a * u

    def get_convective_speed(self, u=0.0):
        return self._a

    def inviscid(self):
        return True


class LinearAdvectionDiffusion(LinearAdvection):
    """ \f$ \partial_t u + a\,\partial_x u = \partial_x \left(b\,\partial_x\right) \f$
    """

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

    def inviscid(self):
        return self._b == 0


class InviscidBurgers(concept.ScalarEquation):
    """ \f$ \partial_t u + ku\,\partial_x u = 0 \f$
    """

    def __init__(self, k=1.0):
        concept.Equation.__init__(self, float)
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

    def inviscid(self):
        return True


class Burgers(InviscidBurgers):
    """ \f$ \partial_t u + ku\,\partial_x u = \partial_x \left(b\,\partial_x\right) \f$
    """

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

    def inviscid(self):
        return self._nu == 0


class Coupled(concept.EquationSystem):
    """A system of two linearly coupled scalar equations.
    """

    def __init__(self, v_0: concept.ScalarEquation, v_1: concept.ScalarEquation,
            R: np.ndarray):
        concept.EquationSystem.__init__(self)
        self._v_0 = v_0
        self._v_1 = v_1
        self._component_names = (v_0.name()+'0', v_1.name()+'1')
        assert R.shape == (2, 2)
        self._R = R
        self._L = np.linalg.inv(R)
        self._A_0 = np.tensordot(R[:, 0], self._L[0], 0)
        self._A_1 = np.tensordot(R[:, 1], self._L[1], 0)

    def n_component(self):
        return 2

    def component_names(self) -> tuple[str]:
        return self._component_names

    def name(self, verbose=True) -> str:
        return "Coupled"

    def get_convective_jacobian(self, U: np.ndarray) -> np.ndarray:
        lambdas = self.get_convective_eigvals(U)
        return lambdas[0] * self._A_0 + lambdas[1] * self._A_1

    def get_convective_eigmats(self, U: np.ndarray) -> tuple:
        return (self._L, self._R)

    def get_convective_eigvals(self, U: np.ndarray) -> np.ndarray:
        V = self.to_characteristics(U)
        lambda_0 = self._v_0.get_convective_speed(V[0])
        lambda_1 = self._v_1.get_convective_speed(V[1])
        return np.array([lambda_0, lambda_1])

    def get_convective_flux(self, U: np.ndarray) -> np.ndarray:
        V = self.to_characteristics(U)
        flux_0 = self._v_0.get_convective_flux(V[0])
        flux_1 = self._v_1.get_convective_flux(V[1])
        return self._R @ np.array([flux_0, flux_1])

    def get_diffusive_coeff(self, U: np.ndarray) -> np.ndarray:
        V = self.to_characteristics(U)
        nu_0 = self._v_0.get_diffusive_coeff(V[0])
        nu_1 = self._v_1.get_diffusive_coeff(V[1])
        return nu_0 * self._A_0 + nu_1 * self._A_1

    def get_diffusive_flux(self, U: np.ndarray, dU: np.ndarray,
            nu_extra: np.ndarray) -> np.ndarray:
        V = self.to_characteristics(U)
        dV = self.to_characteristics(dU)
        nu_0, nu_1 = nu_extra[0], nu_extra[1]
        flux_0 = self._v_0.get_diffusive_flux(V[0], dV[0], nu_0)
        flux_1 = self._v_1.get_diffusive_flux(V[1], dV[1], nu_1)
        return self._R @ np.array([flux_0, flux_1])

    def get_diffusive_radius(self, U, nu_extra, h_given):
        V = self.to_characteristics(U)
        nu_0, nu_1 = nu_extra[0], nu_extra[1]
        s_0 = self._v_0.get_diffusive_radius(V[0], nu_0, h_given)
        s_1 = self._v_1.get_diffusive_radius(V[1], nu_1, h_given)
        return max(s_0, s_1)

    def inviscid(self):
        return self._v_0.inviscid() and self._v_1.inviscid()


class Euler(concept.EquationSystem):

    def __init__(self, gamma=1.4):
        concept.EquationSystem.__init__(self)
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

    def _get_diffusive_flux_physically(self, U, dx_U, nu_extra):
        rho, u, p = self.conservative_to_primitive(U)
        dx_rho = dx_U[0]
        dx_u = (dx_U[1] - u * dx_rho) / rho
        mu = rho * nu_extra
        tau = (4.0/3) * mu * dx_u
        dx_e0 = dx_U[2]
        prandtl = 1e100
        q = self._gas.get_heat_flux(rho, dx_rho, u, dx_u, p, dx_e0, mu, prandtl)
        return np.array([0, tau, tau * u - q])

    def _get_diffusive_flux_eigenwisely(self, U, dx_U, nu_extra):
        L, R = self.get_convective_eigmats(U)
        B = np.zeros((3, 3))
        for i in range(3):
            B += nu_extra[i] * np.tensordot(R[:, i], L[i], 0)
        return B @ dx_U

    def get_diffusive_flux(self, U, dx_U, nu_extra):
        if type(nu_extra) is np.ndarray:
            return self._get_diffusive_flux_eigenwisely(U, dx_U, nu_extra)
        else:
            return self._get_diffusive_flux_physically(U, dx_U, nu_extra)

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

    def get_diffusive_radius(self, U, nu_extra, h_given):
        V = self.to_characteristics(U)
        if isinstance(nu_extra, np.ndarray):
            return np.max(nu_extra) / h_given
        else:
            assert isinstance(nu_extra, float)
            return nu_extra / h_given

    def inviscid(self):
        return True


if __name__ == '__main__':
    pass
