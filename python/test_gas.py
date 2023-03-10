"""Test various gas models.
"""
import unittest

import gas


class TestIdealGas(unittest.TestCase):
    """Test the IdealGas class.
    """

    def test_gamma(self):
        """Test constants related to gamma.
        """
        gamma = 1.4
        ideal = gas.Ideal(gamma=1.4)
        self.assertEqual(ideal.gamma(), gamma)
        self.assertEqual(ideal.gamma_minus_1(), gamma-1)
        self.assertEqual(ideal.gamma_plus_1(), gamma+1)
        self.assertEqual(ideal.gamma_minus_3(), gamma-3)
        self.assertEqual(ideal.gamma_minus_1_over_gamma_plus_1(),
                                         (gamma-1)/(gamma+1))
        self.assertEqual(ideal.one_over_gamma_minus_1(), 1/(gamma-1))


if __name__ == '__main__':
    unittest.main()
