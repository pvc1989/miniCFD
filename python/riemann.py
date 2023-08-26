import abc

import numpy as np
from scipy.optimize import fsolve

import concept
import equation
import gas
import expansion


class Solver(concept.RiemannSolver):

    # DDG constants:
    _beta_0, _beta_1 = 3.0, 1.0 / 12

    def __init__(self, equation: concept.Equation) -> None:
        super().__init__(equation)
        self._value_left = None
        self._value_right = None

    def set_initial(self, u_left, u_right):
        self._value_left = u_left
        self._value_right = u_right
        self._determine_wave_structure()

    @abc.abstractmethod
    def _determine_wave_structure(self):
        # Determine boundaries of constant regions and elementary waves,
        # as well as the constant states.
        pass

    def get_value(self, x, t):
        if t == 0:
            if x <= 0:
                return self._value_left
            else:
                return self._value_right
        else:  # t > 0
            return self._get_value(slope=x/t)

    @abc.abstractmethod
    def _get_value(self, slope):
        """Get the self-similar solution on x / t.
        """

    def get_upwind_flux(self, u_left, u_right):
        self.set_initial(u_left, u_right)
        u_upwind = self.get_value(x=0, t=1)
        # Actually, get_value(x=0, t=1) returns either U(x=-0, t=1) or U(x=+0, t=1).
        # If the speed of a shock is 0, then U(x=-0, t=1) != U(x=+0, t=1).
        # However, the jump condition guarantees F(U(x=-0, t=1)) == F(U(x=+0, t=1)).
        return self.equation().get_convective_flux(u_upwind)

    def get_interface_gradient(self, h_left, h_right,
            u_jump, du_mean, ddu_jump):
        delta_x = (h_left + h_right) / 2
        du = self._beta_0 / delta_x * u_jump
        du += self._beta_1 * delta_x * ddu_jump
        du += du_mean
        return du

    def get_interface_flux_and_bjump(self, expansion_left: expansion.Taylor,
            expansion_right: expansion.Taylor, viscosity: float):
        # Get the convective flux on the interface:
        x_left = expansion_left.x_right()
        u_left = expansion_left.global_to_value(x_left)
        x_right = expansion_right.x_left()
        u_right = expansion_right.global_to_value(x_right)
        flux = self.get_upwind_flux(u_left, u_right)
        if viscosity == 0:
            return flux, u_left - u_left
        # Get the diffusive flux on the interface by the DDG method:
        du_left, ddu_left = 0, 0
        if expansion_left.degree() > 1:
            derivatives = expansion_left.global_to_derivatives(x_left)
            du_left, ddu_left = derivatives[1], derivatives[2]
        elif expansion_left.degree() == 1:
            du_left = expansion_left.global_to_gradient(x_left)
        else:
            pass
        du_right, ddu_right = 0, 0
        if expansion_right.degree() > 1:
            derivatives = expansion_right.global_to_derivatives(x_right)
            du_right, ddu_right = derivatives[1], derivatives[2]
        elif expansion_right.degree() == 1:
            du_right = expansion_right.global_to_gradient(x_right)
        else:
            pass
        u_jump = u_right - u_left
        du = self.get_interface_gradient(expansion_left.length(), 
            expansion_right.length(), u_jump,
            (du_left + du_right) / 2, ddu_right - ddu_left)
        return flux - viscosity * du, viscosity / 2 * u_jump


class LinearAdvection(Solver):

    def __init__(self, a_const: float, value_type=float):
        e = equation.LinearAdvection(a_const, value_type)
        Solver.__init__(self, e)

    def equation(self) -> concept.Equation:
        return self._equation

    def _determine_wave_structure(self):
        pass

    def _get_value(self, slope):
        if slope <= self.equation().get_convective_speed():
            return self._value_left
        else:
            return self._value_right


class LinearAdvectionDiffusion(LinearAdvection):

    def __init__(self, a_const, b_const):
        e = equation.LinearAdvectionDiffusion(a_const, b_const)
        Solver.__init__(self, e)


class InviscidBurgers(Solver):

    def __init__(self, k=1.0):
        Solver.__init__(self, equation.InviscidBurgers(k))

    def equation(self) -> equation.InviscidBurgers:
        return self._equation

    def _determine_wave_structure(self):
        self._slope_left, self._slope_right = 0, 0
        if self._value_left <= self._value_right:  # rarefaction
            self._slope_left = \
                self._equation.get_convective_speed(self._value_left)
            self._slope_right = \
                self._equation.get_convective_speed(self._value_right)
        else:  # shock
            u_mean = (self._value_left + self._value_right) / 2
            slope = self._equation.get_convective_speed(u_mean) 
            self._slope_left, self._slope_right = slope, slope

    def _get_value(self, slope):
        if slope <= self._slope_left:
            return self._value_left
        elif slope >= self._slope_right:
            return self._value_right
        else:  # slope_left < slope < slope_right, u = a^{-1}(slope)
            k = (self._slope_right - self._slope_left) / (self._value_right - self._value_left)
            return slope / k


class Burgers(InviscidBurgers):

    def __init__(self, k=1.0, nu=0.0):
        Solver.__init__(self, equation.Burgers(k, nu))


class Euler(Solver):

    def __init__(self, gamma=1.4):
        self._gas = gas.Ideal(gamma)
        Solver.__init__(self, equation.Euler(gamma))

    def equation(self) -> equation.Euler:
        return self._equation

    def _determine_wave_structure(self):
        # set states in unaffected regions
        rho_left, u_left, p_left = \
            self.equation().conservative_to_primitive(self._value_left)
        rho_right, u_right, p_right = \
            self.equation().conservative_to_primitive(self._value_right)
        assert p_left*p_right != 0, (p_left, p_right)
        self._u_left, self._u_right = u_left, u_right
        self._p_left, self._p_right = p_left, p_right
        self._rho_left, self._rho_right = rho_left, rho_right
        a_left = self._gas.p_rho_to_a(p_left, rho_left)
        a_right = self._gas.p_rho_to_a(p_right, rho_right)
        self._riemann_invariants_left = np.array([
            p_left / rho_left**self._gas.gamma(),
            u_left + 2*a_left/self._gas.gamma_minus_1()])
        self._riemann_invariants_right = np.array([
            p_right / rho_right**self._gas.gamma(),
            u_right - 2*a_right/self._gas.gamma_minus_1()])
        # determine wave heads and tails and states between 1-wave and 3-wave
        if self._exist_vacuum():
            self._p_2 = self._rho_2_left = self._rho_2_right = self._u_2 = np.nan
            self._slope_1_left = u_left - a_left
            self._slope_3_right = u_right + a_right
            self._slope_1_right = self._riemann_invariants_left[1]
            self._slope_3_left = self._riemann_invariants_right[1]
            self._slope_2 = (self._slope_1_right + self._slope_3_left) / 2
            assert (self._slope_1_left < self._slope_1_right
                <= self._slope_2 <= self._slope_3_left < self._slope_3_right)
        else:  # no vacuum
            # 2-field: always a contact
            du = u_right - u_left
            func = lambda p: (self._f(p, p_left, rho_left)
                + self._f(p, p_right, rho_right) + du)
            p_2 = self._guess(p_left, func)
            if np.abs(func(p_2)) > 1e-8:
                roots, infodict, ierror, message = fsolve(func,
                    fprime=lambda p: (self._f_prime(p, p_left, rho_left)
                        + self._f_prime(p, p_right, rho_right)),
                    x0=self._guess(p_left, func),
                    full_output=True)
                assert ierror == 1, message
                p_2 = roots[0]
            u_2 = (u_left - self._f(p_2, p_left, rho_left)
                + u_right + self._f(p_2, p_right, rho_right)) / 2
            self._p_2 = p_2
            self._u_2 = u_2
            self._slope_2 = u_2
            # 1-field: a left running wave
            rho_2, slope_left, slope_right = rho_left, u_left, u_left
            eps = 1e-10
            if p_2 > p_left + eps:  # shock
                rho_2, slope_left, slope_right = self._shock(u_2, p_2, u_left, p_left, rho_left)
            elif p_2 < p_left - eps:  # rarefraction
                # riemann-invariant = u + 2*a/(gamma-1)
                a_2 = a_left + (u_left-u_2)/2*self._gas.gamma_minus_1()
                assert a_2 >= 0, a_2
                rho_2 = p_2 / a_2**2 * self._gas.gamma()
                # eigenvalue = u - a
                slope_left = u_left - a_left
                slope_right = u_2 - a_2
            else:
                pass
            # assert slope_left <= slope_right <= u_2+eps, \
            #     (p_2, p_left, slope_left, slope_right, u_2)
            self._rho_2_left = rho_2
            self._slope_1_left = slope_left
            self._slope_1_right = slope_right
            # 3-field: a right running wave
            rho_2, slope_left, slope_right = rho_right, u_right, u_right
            if p_2 > p_right + 1e-8:  # shock
                rho_2, slope_left, slope_right = self._shock(u_2, p_2, u_right, p_right, rho_right)
            elif p_2 < p_right - 1e-8:  # rarefraction
                # riemann-invariant = u - 2*a/(gamma-1)
                a_2 = a_right - (u_right-u_2)/2*self._gas.gamma_minus_1()
                assert a_2 >= 0, a_2
                rho_2 = p_2 / a_2**2 * self._gas.gamma()
                # eigenvalue = u + a
                slope_left = u_2 + a_2
                slope_right = u_right + a_right
            else:
                pass
            # assert u_2-eps <= slope_left <= slope_right, \
            #     (p_2, p_right, u_2, slope_left, slope_right)
            self._rho_2_right = rho_2
            self._slope_3_left = slope_left
            self._slope_3_right = slope_right
        # print(f'p2 = {self._p_2:5f}, u2 = {self._u_2:5f},',
        #     f'rho2L = {self._rho_2_left:5f}, rho2R = {self._rho_2_right:5f}')

    def _exist_vacuum(self):
        exist = False
        if self._riemann_invariants_left[1] <= self._riemann_invariants_right[1]:
            # print('Vacuum exists.')
            exist = True
        else:
            # print('No vacuum.')
            exist = False
        return exist

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
        while func(p) > 0:
            p *= 0.5
        return p

    @staticmethod
    def _shock(u_2, p_2, u_1, p_1, rho_1):
        assert u_2 != u_1
        slope = u_1 + (p_2-p_1)/(u_2-u_1)/rho_1
        assert u_2 != slope
        rho_2 = rho_1 * (u_1 - slope) / (u_2 - slope)
        return rho_2, slope, slope

    def _get_value(self, slope):
        u, p, rho = 0, 0, 0
        if slope < self._slope_2:
            if slope <= self._slope_1_left:
                u, p, rho = self._u_left, self._p_left, self._rho_left
            elif slope > self._slope_1_right:
                u, p, rho = self._u_2, self._p_2, self._rho_2_left
                # print(p, rho)
            else:  # slope_1_left < slope < slope_2_right
                a = self._riemann_invariants_left[1] - slope
                a *= self._gas.gamma_minus_1_over_gamma_plus_1()
                assert a >= 0, a
                u = slope + a
                rho = a**2 / self._gas.gamma() / self._riemann_invariants_left[0]
                rho = rho**self._gas.one_over_gamma_minus_1()
                p = a**2 * rho / self._gas.gamma()
        else:  # slope > slope_2
            if slope >= self._slope_3_right:
                u, p, rho = self._u_right, self._p_right, self._rho_right
            elif slope <= self._slope_3_left:
                u, p, rho = self._u_2, self._p_2, self._rho_2_right
                # print(p, rho)
            else:  # slope_3_left < slope < slope_3_right
                a = slope - self._riemann_invariants_right[1]
                a *= self._gas.gamma_minus_1_over_gamma_plus_1()
                assert a >= 0, a
                u = slope - a
                rho = a**2 / self._gas.gamma() / self._riemann_invariants_right[0]
                rho = rho**self._gas.one_over_gamma_minus_1()
                p = a**2 * rho / self._gas.gamma()
        return self.equation().primitive_to_conservative(rho, u, p)


class ApproximateEuler(Euler):

    def _determine_wave_structure(self):
        assert False

    def _get_value(self, slope):
        assert False


class Roe(ApproximateEuler):

    def __init__(self, gamma=1.4):
        Euler.__init__(self, gamma)

    def get_upwind_flux(self, value_left, value_right):
        # algebraic averaging
        eq = self.equation()
        flux = eq.get_convective_flux(value_left)
        flux += eq.get_convective_flux(value_right)
        flux /= 2
        # Roe averaging
        rho_left, u_left, p_left = eq.conservative_to_primitive(value_left)
        rho_right, u_right, p_right = eq.conservative_to_primitive(value_right)
        h0_left = eq.primitive_to_total_enthalpy(rho_left, u_left, p_left)
        h0_right = eq.primitive_to_total_enthalpy(rho_right, u_right, p_right)
        rho_left_sqrt = np.sqrt(rho_left)
        rho_right_sqrt = np.sqrt(rho_right)
        rho_sqrt_sum = rho_left_sqrt + rho_right_sqrt
        rho = rho_left_sqrt * rho_right_sqrt
        u = (rho_left_sqrt * u_left + rho_right_sqrt * u_right) / rho_sqrt_sum
        ke = u * u / 2  # kinetic energy
        h0 = (rho_left_sqrt * h0_left + rho_right_sqrt * h0_right) / rho_sqrt_sum
        aa = eq._gas.gamma_minus_1() * (h0 - ke)
        a = np.sqrt(aa)
        ua = u * a
        # 1st wave
        p_jump = p_right - p_left
        u_jump = u_right - u_left
        alpha = (p_jump - rho * a * u_jump) / (2 * aa)
        flux -= 0.5 * alpha * np.abs(u - a) * \
            np.array([1, u - a, h0 - ua])
        # 2nd wave
        alpha = (rho_right - rho_left) - p_jump / aa
        flux -= 0.5 * alpha * np.abs(u) * np.array([1, u, ke])
        # 3rd wave
        alpha = (p_jump + rho * a * u_jump) / (2 * aa)
        flux -= 0.5 * alpha * np.abs(u + a) * \
            np.array([1, u + a, h0 + ua])
        return flux


class LaxFriedrichs(ApproximateEuler):

    def __init__(self, gamma=1.4):
        Euler.__init__(self, gamma)

    def get_upwind_flux(self, value_left, value_right):
        eq = self.equation()
        flux = eq.get_convective_flux(value_right)
        flux += eq.get_convective_flux(value_left)
        flux /= 2
        lambdas = eq.get_convective_eigvals(value_left)
        lambda_max = max(np.abs(lambdas[0]), np.abs(lambdas[2]))
        lambdas = eq.get_convective_eigvals(value_right)
        lambda_max = max(lambda_max, np.abs(lambdas[0]), np.abs(lambdas[2]))
        flux -= lambda_max / 2 * (value_right - value_left)
        return flux


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    euler = equation.Euler(gamma=1.4)
    solver = Euler(gamma=1.4)

    problems = dict()
    # tests in Table 4.1 of Toro[2009], see https://doi.org/10.1007/b79761
    problems['Sod'] = (0.25,
        euler.primitive_to_conservative(rho=1.0, u=0, p=1.0),
        euler.primitive_to_conservative(rho=0.125, u=0, p=0.1))
    problems['Lax'] = (0.15,
        euler.primitive_to_conservative(rho=0.445, u=0.698, p=3.528),
        euler.primitive_to_conservative(rho=0.5, u=0.0, p=0.571))
    problems['ShockCollision'] = (0.035,
        euler.primitive_to_conservative(rho=5.99924, u=19.5975, p=460.894),
        euler.primitive_to_conservative(rho=5.99242, u=-6.19633, p=46.0950))
    problems['BlastFromLeft'] = (0.012,
        euler.primitive_to_conservative(rho=1, u=0, p=1000),
        euler.primitive_to_conservative(rho=1, u=0, p=0.01))
    problems['BlastFromRight'] = (0.035,
        euler.primitive_to_conservative(rho=1, u=0, p=0.01),
        euler.primitive_to_conservative(rho=1, u=0, p=100))
    problems['AlmostVacuumed'] = (0.15,
        euler.primitive_to_conservative(rho=1, u=-2, p=0.4),
        euler.primitive_to_conservative(rho=1, u=+2, p=0.4))
    # other tests
    problems['Vacuumed'] = (0.1,
        euler.primitive_to_conservative(rho=1, u=-4, p=0.4),
        euler.primitive_to_conservative(rho=1, u=+4, p=0.4))

    # range for plot
    x_vec = np.linspace(start=-0.5, stop=0.5, num=1001)

    for name, problem in problems.items():
        u_left = problem[1]
        u_right = problem[2]
        try:
            solver.set_initial(u_left, u_right)
            rho_vec = np.zeros(len(x_vec))
            u_vec = np.zeros(len(x_vec))
            p_vec = np.zeros(len(x_vec))
            for i in range(len(x_vec)):
                x = x_vec[i]
                U = solver.get_value(x, t=problem[0])
                rho, u, p = euler.conservative_to_primitive(U)
                rho_vec[i], u_vec[i], p_vec[i] = rho, u, p
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
        plt.savefig(name+'.svg')
