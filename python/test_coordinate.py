"""Tests for coordinate transforms.
"""
import unittest

import numpy as np

from concept import ShiftedCoordinate
from coordinate import LinearCoordinate


class TestLinearCoordinate(unittest.TestCase):

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = np.random.rand()
        self._x_right = 1 + np.random.rand()
        self._coordinate = LinearCoordinate(self._x_left, self._x_right)

    def test_consistency_of_transforms(self):
        """Test consistency of coordinate transforms between local and global.
        """
        points = np.linspace(self._x_left, self._x_right, num=201)
        for x_global in points:
            x_local = self._coordinate.global_to_local(x_global)
            self.assertAlmostEqual(x_global,
                self._coordinate.local_to_global(x_local))

    def test_consistency_with_shifted(self):
        """Test consistency with a shifted coordinate.
        """
        x_shift = np.random.rand()
        shifted = ShiftedCoordinate(self._coordinate, x_shift)
        expected = LinearCoordinate(self._x_left+x_shift, self._x_right+x_shift)
        self.assertEqual(shifted.jacobian_degree(), expected.jacobian_degree())
        self.assertAlmostEqual(shifted.x_left(), expected.x_left())
        self.assertAlmostEqual(shifted.x_right(), expected.x_right())
        self.assertAlmostEqual(shifted.x_center(), expected.x_center())
        self.assertAlmostEqual(shifted.length(), expected.length())
        points = np.linspace(-1.0, 1.0, num=201)
        for x_local in points:
            self.assertAlmostEqual(shifted.local_to_global(x_local),
                expected.local_to_global(x_local))
            self.assertAlmostEqual(shifted.local_to_jacobian(x_local),
                expected.local_to_jacobian(x_local))
            x_global = shifted.local_to_global(x_local)
            self.assertAlmostEqual(shifted.global_to_local(x_global),
                expected.global_to_local(x_global))
            self.assertAlmostEqual(shifted.global_to_jacobian(x_global),
                expected.global_to_jacobian(x_global))


if __name__ == '__main__':
    unittest.main()
