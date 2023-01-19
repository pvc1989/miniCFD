import abc
import numpy as np
from matplotlib import pyplot as plt
from sys import argv

import concept
import equation
import riemann
import spatial
import temporal


class SolverBase(abc.ABC):

    def plot(self, t_curr, n_point: int):
        points = np.linspace(self._spatial.x_left(), self._spatial.x_right(), n_point)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = self.u_exact(point_i, t_curr)
        plt.clf()
        plt.plot(points, approx_solution, 'b+', label='Approximate Solution')
        plt.plot(points, expect_solution, 'r-', label='Exact Solution')
        plt.ylim([-1.1, 1.1])
        plt.title(f't = {t_curr:.2f}')
        plt.legend()
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            self._spatial.n_element() + 1))
        plt.grid()
        plt.show()

    @abc.abstractmethod
    def a_max(self):
        pass

    def run(self, t_start: float, t_stop: float,  n_step: int):
        delta_t = (t_stop - t_start) / n_step
        cfl = self.a_max() * delta_t / self._spatial.delta_x()
        print(f"n_step = {n_step}, delta_t = {delta_t}, cfl = {cfl}")
        self._spatial.initialize(lambda x_global: self.u_init(x_global))
        def plot(t_curr):
            self.plot(t_curr, n_point=101)
        self._ode_solver.solve(self._spatial, plot, t_start, t_stop, delta_t)


class LinearAdvection(SolverBase):

    def __init__(self, a_const: float,
            spatial_scheme: concept.SpatialScheme,
            ode_solver: concept.OdeSolver) -> None:
        self._a_const = a_const
        self._spatial = spatial_scheme
        self._ode_solver = ode_solver

    def a_max(self):
        return np.abs(self._a_const)

    def u_init(self, x_global):
        value = np.sin(x_global * np.pi * 2 / self._spatial.length())
        return value  # np.sign(value)

    def u_exact(self, x_global, t_curr):
        return self.u_init(x_global - self._a_const * t_curr)


class InviscidBurgers(SolverBase):

    def __init__(self,
            spatial_scheme: concept.SpatialScheme,
            ode_solver: concept.OdeSolver) -> None:
        self._spatial = spatial_scheme
        self._ode_solver = ode_solver

    def a_max(self):
        return 1.0

    def u_init(self, x_global):
        value = np.sin(x_global * np.pi * 2 / self._spatial.length())
        return value  # np.sign(value)

    def u_exact(self, x_global, t_curr):
        return self.u_init(x_global - 0 * t_curr)


if __name__ == '__main__':
    if len(argv) < 9:
        print("Usage: \n  python3 solver.py <degree> <n_element> <x_left>",
            "<x_right> <rk_order> <t_start> <t_stop> <delta_t>")
        exit(-1)
    solver = InviscidBurgers(
        spatial_scheme=spatial.LagrangeDG(
            equation.InviscidBurgers(),
            riemann.InviscidBurgers(),
            degree=int(argv[1]), n_element=int(argv[2]),
            x_left=float(argv[3]), x_right=float(argv[4])),
        ode_solver = temporal.RungeKutta(order=int(argv[5])))
    solver.run(t_start=float(argv[6]), t_stop=float(argv[7]),
        n_step=int(argv[8]))
    exit(0)
    a_const = 10.0
    solver = LinearAdvection(a_const,
        spatial_scheme=spatial.LagrangeDG(
            equation.LinearAdvection(a_const),
            riemann.LinearAdvection(a_const),
            degree=int(argv[1]), n_element=int(argv[2]),
            x_left=float(argv[3]), x_right=float(argv[4])),
        ode_solver = temporal.RungeKutta(order=int(argv[5])))
    solver.run(t_start=float(argv[6]), t_stop=float(argv[7]),
        n_step=int(argv[8]))
