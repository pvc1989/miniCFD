"""Concrete implementations of temporal schemes.
"""
from concept import OdeSolver


class ExplicitEuler(OdeSolver):
    """The explicit Euler method.
    """

    def update(self, ode_system, delta_t):
        u_curr = ode_system.get_solution_column()
        residual = ode_system.get_residual_column()
        u_next = u_curr
        u_next += residual * delta_t
        ode_system.set_solution_column(u_next)


class RungeKutta(OdeSolver):
    """The explicit Runge--Kutta methods.
    """

    _euler = ExplicitEuler()

    def __init__(self, order: int):
        assert 1 <= order <= 4
        self._order = order

    def update(self, ode_system, delta_t):
        if self._order == 1:
            self._rk1_update(ode_system, delta_t)
        elif self._order == 2:
            self._rk2_update(ode_system, delta_t)
        elif self._order == 3:
            self._rk3_update(ode_system, delta_t)
        elif self._order == 4:
            self._rk4_update(ode_system, delta_t)
        else:
            raise NotImplementedError("Only RK[1-4] are implemented.")

    @staticmethod
    def _rk1_update(ode_system, delta_t):
        RungeKutta._euler.update(ode_system, delta_t)

    @staticmethod
    def _rk2_update(ode_system, delta_t):
        u_curr = ode_system.get_solution_column()  # u_curr == U_{n}
        RungeKutta._euler.update(ode_system, delta_t)
        # Now, ode_system holds U_{n + 1/2}
        RungeKutta._euler.update(ode_system, delta_t)
        u_next = ode_system.get_solution_column()
        # Now, u_next == U_{n + 1/2} + R_{n + 1/2} * delta_t
        u_next += u_curr
        u_next /= 2
        # Now, u_next == U_{n + 1}
        ode_system.set_solution_column(u_next)

    @staticmethod
    def _rk3_update(ode_system, delta_t):
        u_curr = ode_system.get_solution_column()  # u_curr == U_{n}
        RungeKutta._euler.update(ode_system, delta_t)
        # Now, ode_system holds U_{n + 1/3}
        RungeKutta._euler.update(ode_system, delta_t)
        u_next = ode_system.get_solution_column()
        # Now, u_next == U_{n + 1/3} + R_{n + 1/3} * dt
        u_next += u_curr * 3
        u_next /= 4
        # Now, u_next == U_{n + 2/3}.
        ode_system.set_solution_column(u_next)
        RungeKutta._euler.update(ode_system, delta_t)
        u_next = ode_system.get_solution_column()
        # Now, u_next == U_{n + 2/3} + R_{n + 2/3} * dt
        u_next *= 2
        u_next += u_curr
        u_next /= 3
        # Now, u_next == U_{n + 1}.
        ode_system.set_solution_column(u_next)

    @staticmethod
    def _rk4_update(ode_system, delta_t):
        u_frac04 = ode_system.get_solution_column()  # := U_{n}
        r_frac04 = ode_system.get_residual_column()  # := R_{n}
        u_frac14 = u_frac04 + r_frac04 * delta_t / 2
        ode_system.set_solution_column(u_frac14)
        # Now, ode_system holds U_{n + 1/4} and R_{n + 1/4}
        r_frac14 = ode_system.get_residual_column()
        u_frac24 = u_frac04 + r_frac14 * delta_t / 2
        ode_system.set_solution_column(u_frac24)
        # Now, ode_system holds U_{n + 2/4} and R_{n + 2/4}
        r_frac24 = ode_system.get_residual_column()
        u_frac34 = u_frac04 + r_frac24 * delta_t
        ode_system.set_solution_column(u_frac34)
        # Now, ode_system holds U_{n + 3/4} and R_{n + 3/4}
        r_frac34 = ode_system.get_residual_column()
        # delta_U = (R_{n} + 2 * R_{n+1/4} + 2 * R_{n+2/4} + R_{n+3/4}) * dt / 6
        delta_u = r_frac04 + r_frac34
        delta_u += (r_frac14 + r_frac24) * 2
        delta_u *= (delta_t / 6)
        # U_{n+1} == U_{n} + delta_U
        ode_system.set_solution_column(u_frac04 + delta_u)


if __name__ == '__main__':
    pass
