"""Implementations of concrete coordinate transforms.
"""
import concept


class Shifted(concept.Coordinate):
    """An wrapper that acts as if a given coordinate is shifted along x-axis by a given amount.
    """

    def __init__(self, coordinate: concept.Coordinate, x_shift: float):
        self._unshifed_coordinate = coordinate
        self._x_shift = x_shift

    def jacobian_degree(self) -> int:
        return self._unshifed_coordinate.jacobian_degree()

    def local_to_global(self, x_local: float) -> float:
        x_unshifted = self._unshifed_coordinate.local_to_global(x_local)
        return x_unshifted + self._x_shift

    def global_to_local(self, x_global: float) -> float:
        x_unshifed = x_global - self._x_shift
        return self._unshifed_coordinate.global_to_local(x_unshifed)

    def local_to_jacobian(self, x_local: float) -> float:
        return self._unshifed_coordinate.local_to_jacobian(x_local)

    def global_to_jacobian(self, x_global: float) -> float:
        x_unshifed = x_global - self._x_shift
        return self._unshifed_coordinate.global_to_jacobian(x_unshifed)

    def x_left(self):
        x_unshifted = self._unshifed_coordinate.x_left()
        return x_unshifted + self._x_shift

    def x_right(self):
        x_unshifted = self._unshifed_coordinate.x_right()
        return x_unshifted + self._x_shift

    def x_center(self):
        x_unshifted = self._unshifed_coordinate.x_center()
        return x_unshifted + self._x_shift

    def length(self):
        return self._unshifed_coordinate.length()


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
