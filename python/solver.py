import numpy as np
from matplotlib import pyplot as plt

import equation
import riemann
import spatial
import temporal


class RungeKuttaFluxReconstruction:

    def __init__(self) -> None:
        self._n_element = 5
        self._a_const = 10.0
        self._spatial = spatial.FluxReconstruction(
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
            expect_solution[i] = exact(point_i)
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
        cfl_number = 0.05  # a * dt / dx
        t_start = 0.0
        t_stop = 1.0
        delta_t = cfl_number * self._delta_x / self._a_const
        print(f"delta_t <= {delta_t}")
        n_step = int((t_stop - t_start) / delta_t) + 1
        delta_t = (t_stop - t_start) / n_step
        print(f"delta_t == {delta_t}")
        def u_init(x_global):
            return np.sin(x_global * np.pi * 2 / self._spatial.length())
        self._spatial.initialize(u_init)
        plt.figure()
        self.plot(exact=u_init, t_curr=0.0, n_point=101)
        plot_steps = range(0, n_step+1, n_step//4)
        for i_step in range(1, n_step + 1):
            t_curr = t_start + i_step * delta_t
            self._temporal.update(self._spatial, delta_t)
            if i_step not in plot_steps:
                continue
            print(f'step {i_step}, t = {t_curr}')
            def u_exact(point):
                return u_init(point - self._a_const * t_curr)
            self.plot(u_exact, t_curr, 101)


if __name__ == '__main__':
    solver = RungeKuttaFluxReconstruction()
    solver.run()
