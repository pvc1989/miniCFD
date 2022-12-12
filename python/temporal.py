import abc


class TemporalScheme(abc.ABC):
  # Interface to solve an ODE system M \dv{U}{t} = R, in which
  # M (the mass matrix) and R (the residual) are provided by a SemiDiscreteSystem object.

  @abc.abstractmethod
  def update(self, semi_discrete_system, delta_t):
    pass


class SspRungeKutta(TemporalScheme):

  def __init__(self, order):
    assert isinstance(order, int) and 1 <= order <= 3
    self._order = order


  def update(self, semi_discrete_system, delta_t):
    if self._order == 1:
      self._rk1_update(semi_discrete_system, delta_t)
    elif self._order == 2:
      self._rk2_update(semi_discrete_system, delta_t)
    elif self._order == 3:
      self._rk3_update(semi_discrete_system, delta_t)
    else:
      raise "unsupported order"


  def _rk1_update(self, semi_discrete_system, delta_t):
    u_curr = semi_discrete_system.get_unknown()
    residual = semi_discrete_system.get_redisual(u_curr)
    u_next = u_curr
    u_next += residual * delta_t
    semi_discrete_system.set_unknown(u_next)


  def _rk2_update(self, semi_discrete_system, delta_t):
    u_curr = semi_discrete_system.get_unknown()  # u_curr == U_{n}
    residual = semi_discrete_system.get_redisual(u_curr)
    u_next = u_curr + residual * delta_t  # u_next == U_{n + 1/2}
    residual = semi_discrete_system.get_redisual(u_next)
    u_next += residual * delta_t  # u_next == U_{n + 1/2} + R_{n + 1/2} * dt
    u_next += u_curr  # u_next == U_{n} + (U_{n + 1/2} + R_{n + 1/2} * dt)
    u_next /= 2
    semi_discrete_system.set_unknown(u_next)


  def _rk3_update(self, semi_discrete_system, delta_t):
    u_curr = semi_discrete_system.get_unknown()  # u_curr == U_{n}
    residual = semi_discrete_system.get_redisual(u_curr)
    u_next = u_curr + residual * delta_t
    # Now, u_next == U_{n + 1/3}.
    residual = semi_discrete_system.get_redisual(u_next)
    u_next += residual * delta_t  # u_next == U_{n + 1/3} + R_{n + 1/3} * dt
    u_next *= 3/4
    u_next += u_curr / 4  # u_next == (3/4)U_{n} + (1/4)(U_{n + 1/3} + R_{n + 1/3} * dt)
    # Now, u_next == U_{n + 2/3}.
    residual = semi_discrete_system.get_redisual(u_next)
    u_next += residual * delta_t  # u_next == U_{n + 2/3} + R_{n + 2/3} * dt
    u_next *= 2/3
    u_next += u_curr / 3
    # Now, u_next == U_{n + 1}.
    semi_discrete_system.set_unknown(u_next)


if __name__ == '__main__':
  pass
