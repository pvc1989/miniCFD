import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as mpla
from sys import argv

import concept
import equation
import riemann
import spatial
import temporal



class LinearAdvection(object):

    def __init__(self, a_const: float,
            degree: int, n_element: int,
            x_left: float, x_right: float,
            ode_solver: concept.OdeSolver) -> None:
        self._a_const = a_const
        self._degree = degree
        self._n_element = n_element
        self._x_left = x_left
        self._x_right = x_right
        equation_object = equation.LinearAdvection(a_const)
        riemann_object = riemann.LinearAdvection(a_const)
        self._dg = spatial.LagrangeDG(equation_object, riemann_object,
            degree, n_element, x_left, x_right)
        self._fr = spatial.LagrangeFR(equation_object, riemann_object,
            degree, n_element, x_left, x_right)
        self._dgfr = spatial.DGwithLagrangeFR(equation_object, riemann_object,
            degree, n_element, x_left, x_right)
        self._length = self._dg.length()
        self._delta_x = self._dg.delta_x()
        self._ode_solver = ode_solver

    def a_max(self):
        return np.abs(self._a_const)

    def get_exact_ydata(self, t_curr, points):
        n_point = len(points)
        expect_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            expect_solution[i] = self.u_exact(point_i, t_curr)
        return expect_solution

    def get_approx_ydata(self, t_curr, points):
        n_point = len(points)
        dg_solution = np.ndarray(n_point)
        fr_solution = np.ndarray(n_point)
        dgfr_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            dg_solution[i] = self._dg.get_solution_value(point_i)
            fr_solution[i] = self._fr.get_solution_value(point_i)
            dgfr_solution[i] = self._dgfr.get_solution_value(point_i)
        return dg_solution, fr_solution, dgfr_solution

    def animate(self, t_start: float, t_stop: float,  n_step: int):
        delta_t = (t_stop - t_start) / n_step
        cfl = self.a_max() * delta_t / self._delta_x
        print(f"n_step = {n_step}, delta_t = {delta_t}, cfl = {cfl}")
        # general plot setting
        plt.figure(figsize=(6, 3))
        plt.ylim([-1.4, 1.4])
        plt.xticks(np.linspace(self._x_left, self._x_right,
            self._n_element + 1))
        plt.grid()
        # initialize line-plot objects
        exact_points = np.linspace(self._x_left, self._x_right, 1001)
        approx_points = np.linspace(self._x_left, self._x_right, 101)
        expect_line, = plt.plot([], [], 'r-', label='Exact Solution')
        dg_line, = plt.plot([], [], '1', label='DG Solution')
        fr_line, = plt.plot([], [], '2', label='FR Solution')
        dgfr_line, = plt.plot([], [], '3', label='DGFR Solution')
        # initialize animation
        def init_func():
            u_init = lambda x_global: self.u_init(x_global)
            self._dg.initialize(u_init)
            self._fr.initialize(u_init)
            self._dgfr.initialize(u_init)
            expect_line.set_xdata(exact_points)
            dg_line.set_xdata(approx_points)
            fr_line.set_xdata(approx_points)
            dgfr_line.set_xdata(approx_points)
        # update data for the next frame
        def func(t_curr):
            expect_solution = self.get_exact_ydata(t_curr, exact_points)
            expect_line.set_ydata(expect_solution)
            dg_solution, fr_solution, dgfr_solution = self.get_approx_ydata(t_curr, approx_points)
            dg_line.set_ydata(dg_solution)
            fr_line.set_ydata(fr_solution)
            dgfr_line.set_ydata(dgfr_solution)
            plt.title(f't = {t_curr:.2f}')
            plt.legend(loc='upper right')
            self._ode_solver.update(self._dg, delta_t)
            self._ode_solver.update(self._fr, delta_t)
            self._ode_solver.update(self._dgfr, delta_t)
        frames = np.linspace(t_start, t_stop, n_step)
        anim = mpla.FuncAnimation(
            plt.gcf(), func, frames, init_func, interval=5)
        plt.show()

    def u_init(self, x_global):
        # x_global = x_global - self._spatial.x_left()
        # value = np.sin(x_global * np.pi * 2 / self._spatial.length())
        while x_global > self._x_right:
            x_global -= self._length
        while x_global < self._x_left:
            x_global += self._length
        k1 = 1 * np.pi / self._delta_x
        k2 = k1 * 4
        quad_length = self._length / 4
        gauss_width = self._delta_x / 2
        value =  np.exp(-((x_global - quad_length) / gauss_width)**2 / 2) * np.sin(k1 * x_global)
        value += np.exp(-((x_global + quad_length) / gauss_width)**2 / 2) * np.sin(k2 * x_global)
        return value

    def u_exact(self, x_global, t_curr):
        return self.u_init(x_global - self._a_const * t_curr)


if __name__ == '__main__':
    if len(argv) < 9:
        print("Usage: \n  python3 solver.py <degree> <n_element> <x_left>",
            "<x_right> <rk_order> <t_start> <t_stop> <n_step>")
        exit(-1)
    solver = LinearAdvection(a_const=1.0,
        degree=int(argv[1]), n_element=int(argv[2]),
        x_left=float(argv[3]), x_right=float(argv[4]),
        ode_solver=temporal.RungeKutta(order=int(argv[5])))
    solver.animate(t_start=float(argv[6]), t_stop=float(argv[7]),
        n_step=int(argv[8]))
