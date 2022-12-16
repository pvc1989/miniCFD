import abc


class SemiDiscreteSystem(abc.ABC):

    @abc.abstractmethod
    def set_unknown(self, unknown):
        pass


    @abc.abstractmethod
    def get_unknown(self):
        pass


    @abc.abstractmethod
    def get_residual(self):
        pass


class TemporalScheme(abc.ABC):
    # Interface to solve an ODE system \dv{U}{t} = R, in which
    # the residual matrix R is provided by a SemiDiscreteSystem object.

    @abc.abstractmethod
    def update(self, semi_discrete_system: SemiDiscreteSystem, delta_t: float):
        assert isinstance(semi_discrete_system, SemiDiscreteSystem)


if __name__ == '__main__':
    pass
