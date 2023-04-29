"""Tests for coordinate transforms.
"""
import unittest

import numpy as np

from coordinate import LinearCoordinate


class TestLinearCoordinate(unittest.TestCase):

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._x_left = np.random.rand()
        self._x_right = 1 + np.random.rand()
        self._coordinate = LinearCoordinate(self._x_left, self._x_right)

    def test_consistency(self):
        """Test consistency of coordinate transforms between local and global.
        """
        points = np.linspace(self._x_left, self._x_right, num=201)
        for x_global in points:
            x_local = self._coordinate.global_to_local(x_global)
            self.assertAlmostEqual(x_global,
                self._coordinate.local_to_global(x_local))


if __name__ == '__main__':
    unittest.main()
