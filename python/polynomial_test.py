"""Tests for some temporal schemes.
"""
import unittest
import numpy as np

from polynomial import Lagrange


class TestLagrange(unittest.TestCase):
    """Test the Lagrange class.
    """

    def __init__(self, method_name: str = "") -> None:
        super().__init__(method_name)
        self._n_point = 5
        self._sample_points = np.linspace(0.0, 10.0, self._n_point)
        self._lagrange = Lagrange(self._sample_points)

    def test_at_sample_points(self):
        """Test the update() method.
        """
        my_function = np.sin
        self._lagrange.approximate(my_function)
        for x in self._sample_points:
            self.assertEqual(my_function(x), self._lagrange.get_function_value(x))


if __name__ == '__main__':
    unittest.main()
