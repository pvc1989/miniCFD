"""Concrete implementations of smoothness indicators.
"""
import numpy as np

from concept import Smoothness
from spatial import PiecewiseContinuous
import integrate


class LiAndRen2011(Smoothness):
    """A smoothness indicator for high-order finite volume schemes.

    See Li Wanai and Ren Yu-Xin, "High-order k-exact WENO finite volume schemes for solving gas dynamic Euler equations on unstructured grids", International Journal for Numerical Methods in Fluids 70, 6 (2011), pp. 742--763.
    """

    def name(self):
        return 'Li and Ren (2011)'

    def get_smoothness_values(self, scheme: PiecewiseContinuous) -> np.ndarray:
        n_cell = scheme.n_element()
        averages = np.ndarray(n_cell)  # could be easier for some schemes
        for i_cell in range(n_cell):
            cell = scheme.get_element_by_index(i_cell)
            averages[i_cell] = integrate.fixed_quad_global(
                lambda x_global: np.abs(cell.get_solution_value(x_global)),
                cell.x_left(), cell.x_right(), cell.degree())
        ratio = 2 * scheme.delta_x()**((cell.degree() + 1) / 2)
        smoothness = np.ndarray(n_cell)
        for i_curr in range(n_cell):
            curr = scheme.get_element_by_index(i_curr)
            curr_solution = curr.get_solution_value(curr.x_center())
            # apply periodic BCs
            i_prev = i_curr - 1
            prev = scheme.get_element_by_index(i_prev)
            prev_solution = prev.get_solution_value(curr.x_center()
                + scheme.length() * (i_curr == 0))
            i_next = (i_curr + 1) % n_cell
            next = scheme.get_element_by_index(i_next)
            next_solution = next.get_solution_value(curr.x_center()
                - scheme.length() * (i_next == 0))
            # evaluate smoothness
            dividend = max(np.abs(curr_solution - next_solution),
                np.abs(curr_solution - prev_solution))
            divisor = ratio * max(averages[i_curr], averages[i_prev],
                averages[i_next])
            # print(dividend, divisor)
            smoothness[i_curr] = dividend / divisor
        return smoothness


if __name__ == '__main__':
    pass
