"""Implementations of concrete coordinate transforms.
"""
import concept


class Linear(concept.Coordinate):
    """A linear map between local and global coordinates.
    """

    def __init__(self, x_left: float, x_right: float):
        self._x_left = x_left
        self._x_right = x_right
        self._x_center = (x_left + x_right) / 2
        self._jacobian = (x_right - x_left) / 2

    def jacobian_degree(self) -> int:
        return 0

    def local_to_global(self, x_local: float) -> float:
        return self._x_center + x_local * self._jacobian

    def global_to_local(self, x_global: float) -> float:
        return (x_global - self._x_center) / self._jacobian

    def local_to_jacobian(self, x_local: float) -> float:
        return self._jacobian

    def global_to_jacobian(self, x_global: float) -> float:
        return self._jacobian

    def x_left(self):
        return self._x_left

    def x_right(self):
        return self._x_right

    def x_center(self):
        return self._x_center


if __name__ == '__main__':
    pass
