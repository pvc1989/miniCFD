"""Implement some element for spatial discretization.
"""


class FluxReconstruction(object):
    """Element for implement flux reconstruction schemes.
    """

    def __init__(self, degree: int, x_min, x_max) -> None:
        self._degree = degree
        self._x_min = x_min
        self._x_max = x_max

    def approximate(self, function):
        """
        """

    def get_unknown_values(self, point):
        """
        """
        return 0.0

    def get_discontinuous_flux(self, point):
        """
        """
        return 0.0

    def get_continuous_flux(self, point, upwind_flux_left, upwind_flux_right):
        """
        """
        return 0.0

    def get_flux_gradient(self, point, upwind_flux_left, upwind_flux_right):
        """
        """
        return 0.0


if __name__ == '__main__':
    pass
