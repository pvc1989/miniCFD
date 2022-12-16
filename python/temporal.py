"""Concrete implementations of temporal schemes.
"""
from concept import TemporalScheme


class ExplicitEuler(TemporalScheme):
    """The explicit Euler method.
    """

    def __init__(self):
        pass

    def update(self, semi_discrete_system, delta_t):
        u_curr = semi_discrete_system.get_unknown().copy()
        residual = semi_discrete_system.get_residual()
        u_next = u_curr
        u_next += residual * delta_t
        semi_discrete_system.set_unknown(u_next)


class SspRungeKutta(TemporalScheme):
    """The explicit Runge--Kutta methods.
    """

    def __init__(self, order: int):
        assert 1 <= order <= 3
        self._order = order
        self._euler = ExplicitEuler()


    def update(self, semi_discrete_system, delta_t):
        if self._order == 1:
            self._rk1_update(semi_discrete_system, delta_t)
        elif self._order == 2:
            self._rk2_update(semi_discrete_system, delta_t)
        elif self._order == 3:
            self._rk3_update(semi_discrete_system, delta_t)
        else:
            raise NotImplementedError("Only 1st-, 2nd- and 3rd-order are implemented.")


    def _rk1_update(self, semi_discrete_system, delta_t):
        self._euler.update(semi_discrete_system, delta_t)


    def _rk2_update(self, semi_discrete_system, delta_t):
        u_curr = semi_discrete_system.get_unknown().copy()  # u_curr == U_{n}
        self._euler.update(semi_discrete_system, delta_t)
        # Now, semi_discrete_system holds U_{n + 1/2}
        self._euler.update(semi_discrete_system, delta_t)
        u_next = semi_discrete_system.get_unknown().copy()
        # Now, u_next == U_{n + 1/2} + R_{n + 1/2} * delta_t
        u_next += u_curr
        u_next /= 2
        # Now, u_next == U_{n + 1}
        semi_discrete_system.set_unknown(u_next)


    def _rk3_update(self, semi_discrete_system, delta_t):
        u_curr = semi_discrete_system.get_unknown().copy()  # u_curr == U_{n}
        self._euler.update(semi_discrete_system, delta_t)
        # Now, semi_discrete_system holds U_{n + 1/3}
        self._euler.update(semi_discrete_system, delta_t)
        u_next = semi_discrete_system.get_unknown().copy()
        # Now, u_next == U_{n + 1/3} + R_{n + 1/3} * dt
        u_next += u_curr * 3
        u_next /= 4
        # Now, u_next == U_{n + 2/3}.
        semi_discrete_system.set_unknown(u_next)
        self._euler.update(semi_discrete_system, delta_t)
        u_next = semi_discrete_system.get_unknown().copy()
        # Now, u_next == U_{n + 2/3} + R_{n + 2/3} * dt
        u_next *= 2
        u_next += u_curr
        u_next /= 3
        # Now, u_next == U_{n + 1}.
        semi_discrete_system.set_unknown(u_next)


if __name__ == '__main__':
    pass
