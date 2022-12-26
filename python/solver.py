import numpy as np
from matplotlib import pyplot as plt

import equation
import riemann
import spatial
import temporal


class RungeKuttaFluxReconstruction:

    def __init__(self) -> None:
        self._x_left = 0.0
        self._x_right = np.pi * 2
        self._n_element = 5
        self._delta_x = (self._x_right - self._x_left) / self._n_element
        self._a_const = np.pi
        self._spatial = spatial.FluxReconstruction(
              self._x_left, self._x_right, self._n_element, degree=1,
              equation=equation.LinearAdvection(self._a_const),
              riemann=riemann.LinearAdvection(self._a_const))
        self._temporal = temporal.SspRungeKutta(1)

    def plot(self, exact, t_curr, n_point: int):
        points = np.linspace(self._x_left, self._x_right, n_point)
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
        plt.title(f't = {t_curr}')
        plt.legend()
        plt.show()

    def run(self):
        cfl_number = 0.5  # a * dt / dx
        t_start = 0.0
        t_stop = 1.0
        delta_t = cfl_number * self._delta_x / self._a_const
        print(f"delta_t <= {delta_t}")
        n_step = int((t_stop - t_start) / delta_t) + 1
        delta_t = (t_stop - t_start) / n_step
        print(f"delta_t == {delta_t}")
        function = np.sin
        self._spatial.initialize(function)
        plt.figure()
        self.plot(exact=function, t_curr=0.0, n_point=101)
        for t_curr in np.linspace(t_start + delta_t, t_stop, n_step):
            self._temporal.update(self._spatial, delta_t)
            def exact(point):
                return function(point - self._a_const * t_curr)
            self.plot(exact, t_curr, 101)
            break


if __name__ == '__main__':
    solver = RungeKuttaFluxReconstruction()
    solver.run()
