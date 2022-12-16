"""Define the abstract interfaces that must be defined in concrete implementations.
"""
import abc


class SemiDiscreteSystem(abc.ABC):
    """The ODE system given by some spatial discretization.
    """

    @abc.abstractmethod
    def set_unknown(self, unknown):
        """Overwrite the values in the matrix of unknowns.
        """

    @abc.abstractmethod
    def get_unknown(self):
        """Get a reference to the matrix of unknowns.
        """

    @abc.abstractmethod
    def get_residual(self):
        """Get a reference to the residual matrix.
        """


class TemporalScheme(abc.ABC):
    """A solver for the standard ODE system dU/dt = R.

    The matrix of unknowns U and the residual matrix R is provided by a SemiDiscreteSystem object.
    """

    @abc.abstractmethod
    def update(self, semi_discrete_system: SemiDiscreteSystem, delta_t: float):
        """Update the given SemiDiscreteSystem object to the next time step.
        """


if __name__ == '__main__':
    pass
