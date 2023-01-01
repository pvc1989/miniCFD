import numpy as np
from matplotlib import pyplot as plt

import equation
import riemann
import spatial
import temporal


class Solver:

    def __init__(self) -> None:
        self._n_element = 4
        self._a_const = 10.0
        self._spatial = spatial.DGwithLagrangeFR(
            equation.LinearAdvection(self._a_const),
            riemann.LinearAdvection(self._a_const),
            degree=2, n_element=self._n_element,
            x_min=0.0, x_max=10.0)
        self._temporal = temporal.SspRungeKutta(3)
        self._delta_x = self._spatial.length() / self._n_element

    def plot(self, exact, t_curr, n_point: int):
        points = np.linspace(self._spatial.x_left(), self._spatial.x_right(), n_point)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = exact(point_i, t_curr)
        plt.clf()
        plt.plot(points, expect_solution, 'r--', label='Exact Solution')
        plt.plot(points, approx_solution, 'b.', label='Approximate Solution')
        plt.ylim([-1.1, 1.1])
        plt.title(f't = {t_curr:.2f}')
        plt.legend()
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            self._n_element + 1))
        plt.grid()
        plt.show()

    def run(self):
        cfl_number = 0.1  # a * dt / dx
        t_start = 0.0
        t_stop = 1.0
        delta_t = cfl_number * self._delta_x / self._a_const
        print(f"delta_t <= {delta_t}")
        n_step = int((t_stop - t_start) / delta_t) + 1
        delta_t = (t_stop - t_start) / n_step
        print(f"delta_t == {delta_t}")
        def u_init(x_global):
            value = np.sin(x_global * np.pi * 2 / self._spatial.length())
            return value  # np.sign(value)
        def u_exact(x_global, t_curr):
            return u_init(x_global - self._a_const * t_curr)
        def plot(t_curr):
            self.plot(u_exact, t_curr, n_point=101)
        self._spatial.initialize(u_init)
        self._temporal.solve(self._spatial, plot, t_start, t_stop, delta_t)


if __name__ == '__main__':
    solver = Solver()
    solver.run()
