"""Put separated modules together.
"""
import abc
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as mpla
from scipy import optimize

import concept
import equation
import riemann
import spatial, detector, limiter
import temporal


class SolverBase(abc.ABC):
    """Define common methods for all solvers.
    """

    def __init__(self, spatial_scheme: concept.SpatialScheme,
            detector: concept.JumpDetector, limiter: concept.Limiter,
            ode_solver: concept.OdeSolver):
        self._spatial = spatial_scheme
        self._detector = detector
        self._limiter = limiter
        self._spatial.set_detector_and_limiter(detector, limiter)
        self._ode_solver = ode_solver
        self._animation = None

    @abc.abstractmethod
    def u_init(self, x_global):
        """Initial condition of the problem."""

    @abc.abstractmethod
    def u_exact(self, x_global, t_curr):
        """Exact solution of the problem."""

    def plot(self, t_curr, n_point: int):
        """Plot solution curves at a given moment.
        """
        points = np.linspace(self._spatial.x_left(), self._spatial.x_right(), n_point)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = self.u_exact(point_i, t_curr)
        plt.figure(figsize=(6, 3))
        plt.plot(points, approx_solution, 'b.', label='Approximate Solution')
        plt.plot(points, expect_solution, 'r-', label='Exact Solution')
        plt.ylim([-1.4, 1.4])
        plt.title(f't = {t_curr:.2f}')
        plt.legend(loc='upper right')
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            self._spatial.n_element() + 1), minor=True)
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            3), minor=False)
        plt.grid(which='both')
        # plt.show()
        plt.savefig(f't={t_curr:.2f}.pdf')

    @abc.abstractmethod
    def a_max(self):
        """The maximum phase speed.
        """

    def run(self, t_start: float, t_stop: float,  n_step: int):
        """Solve the problem in a given time range and plot the results.
        """
        delta_x = self._spatial.delta_x()
        delta_t = (t_stop - t_start) / n_step
        cfl = self.a_max() * delta_t / delta_x
        print(f"delta_x = {delta_x}, delta_t = {delta_t}, cfl = {cfl}")
        self._spatial.initialize(lambda x_global: self.u_init(x_global))
        def plot(t_curr):
            self.plot(t_curr, n_point=101)
        self._ode_solver.solve(self._spatial, plot, t_start, t_stop, delta_t)

    def get_ydata(self, t_curr, points):
        """Get the y-data for updating the solution curvess.
        """
        n_point = len(points)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = self.u_exact(point_i, t_curr)
        return expect_solution, approx_solution

    def animate(self, t_start: float, t_stop: float,  n_step: int):
        """Solve the problem in a given time range and animate the results.
        """
        degree = self._spatial.degree()
        delta_x = self._spatial.delta_x()
        delta_t = (t_stop - t_start) / n_step
        cfl = self.a_max() * delta_t / delta_x
        print(f"delta_x = {delta_x}, delta_t = {delta_t}, cfl = {cfl}")
        # general plot setting
        plt.figure('scheme='+self._spatial.name()+', detector='
            +self._detector.name())
        plt.ylim([-1.4, 1.6])
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            self._spatial.n_element() + 1), minor=True)
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            2), minor=False)
        plt.grid(which='both')
        # initialize line-plot objects
        approx_line, = plt.plot([], [], 'b--',
            label=r'$p=$'+f'{degree}'+r', $h=$'+f'{delta_x:.2f}'
                +', limiter='+self._limiter.name())
        expect_line, = plt.plot([], [], 'r-', label='Exact Solution')
        points = np.linspace(self._spatial.x_left(), self._spatial.x_right(), 101)
        # initialize animation
        def init_func():
            self._spatial.initialize(lambda x_global: self.u_init(x_global))
            expect_line.set_xdata(points)
            approx_line.set_xdata(points)
        # update data for the next frame
        def func(t_curr):
            expect_solution, approx_solution = self.get_ydata(t_curr, points)
            expect_line.set_ydata(expect_solution)
            approx_line.set_ydata(approx_solution)
            plt.title(f't = {t_curr:.2f}')
            plt.legend(loc='upper right')
            self._ode_solver.update(self._spatial, delta_t)
        frames = np.linspace(t_start, t_stop, n_step)
        self._animation = mpla.FuncAnimation(
            plt.gcf(), func, frames, init_func, interval=5)
        plt.show()


class LinearAdvection(SolverBase):
    """Demo the usage of LinearAdvection related classes.
    """

    def __init__(self, a_const: float, k_const: float,
            spatial_scheme: concept.SpatialScheme,
            detector: concept.JumpDetector, limiter: concept.Limiter,
            ode_solver: concept.OdeSolver) -> None:
        super().__init__(spatial_scheme, detector, limiter, ode_solver)
        self._a_const = a_const
        self._wave_number = k_const * np.pi * 2 / self._spatial.length()

    def a_max(self):
        return np.abs(self._a_const)

    def u_init(self, x_global):
        x_global = x_global - self._spatial.x_left()
        value = np.sin(x_global * self._wave_number)
        return np.sign(value)

    def u_exact(self, x_global, t_curr):
        return self.u_init(x_global - self._a_const * t_curr)


class InviscidBurgers(SolverBase):
    """Demo the usage of InviscidBurgers related classes.
    """

    def __init__(self, a_const: float, k_const: float,
            spatial_scheme: concept.SpatialScheme,
            detector: concept.JumpDetector, limiter: concept.Limiter,
            ode_solver: concept.OdeSolver) -> None:
        super().__init__(spatial_scheme, detector, limiter, ode_solver)
        self._k = a_const
        self._wave_number = k_const * np.pi * 2 / self._spatial.length()
        self._u_prev = 0.0
        self._x_mid = (self._spatial.x_left() + self._spatial.x_right()) / 2

    def a_max(self):
        return self._k

    def u_init(self, x_global):
        x_global = x_global - self._spatial.x_left()
        value = np.sin(x_global * self._wave_number)
        return value

    def u_exact(self, x_global, t_curr):
        # Solve u from u_curr = u_init(x - a(u_curr) * t_curr).
        return np.nan
        def func(u_curr):
            return u_curr - self.u_init(x_global - self._k * u_curr * t_curr)
        if np.abs(x_global - self._x_mid) < 1e-6:
            root = 0.0
        else:
            if x_global > self._x_mid and self._u_prev > 0:
                self._u_prev = -self._u_prev
            a = self._u_prev - 0.2
            b = self._u_prev + 0.2
            # print(x_global, self._x_mid, self._u_prev, func(a), func(b))
            root = optimize.bisect(func, a, b)
            self._u_prev = root
        return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python3 solver.py',
        description = 'What the program does',
        epilog = 'Text at the bottom of help')
    parser.add_argument('-n', '--n_element',
        default=23, type=int,
        help='number of elements')
    parser.add_argument('-l', '--x_left',
        default=0.0, type=float,
        help='coordinate of the left end of the domain')
    parser.add_argument('-r', '--x_right',
        default=10.0, type=float,
        help='coordinate of the right end of the domain')
    parser.add_argument('-o', '--rk_order',
        default=3, type=int,
        help='order of Runge--Kutta scheme')
    parser.add_argument('-b', '--t_begin',
        default=0.0, type=float,
        help='time to start')
    parser.add_argument('-e', '--t_end',
        default=10.0, type=float,
        help='time to stop')
    parser.add_argument('-s', '--n_step',
        default=100, type=int,
        help='number of time steps')
    parser.add_argument('-m', '--method',
        choices=['LagrangeDG', 'LagrangeFR', 'LegendreDG', 'LegendreFR',
            'DGwithFR'],
        default='LegendreFR',
        help='method for spatial discretization')
    parser.add_argument('--detector',
        choices=['ReportAll', 'Krivodonova2004', 'LiRen2011', 'ZhuShuQiu2021'],
        default='LiRen2011',
        help='method for detecting jumps')
    parser.add_argument('--limiter',
        choices=['LiWangRen2020', 'ZhongShu2013'],
        default='LiWangRen2020',
        help='method for limiting numerical oscillations')
    parser.add_argument('-d', '--degree',
        default=2, type=int,
        help='degree of polynomials for approximation')
    parser.add_argument('-p', '--problem',
        choices=['Linear', 'Burgers'],
        default='Linear',
        help='problem to be solved')
    parser.add_argument('-a', '--phase_speed',
        default=1.0, type=float,
        help='phase speed of the wave')
    parser.add_argument('-k', '--wave_number',
        default=1.0, type=float,
        help='number of waves in the domain')
    args = parser.parse_args()
    print(args)
    if args.method == 'LagrangeDG':
        SpatialClass = spatial.LagrangeDG
    elif args.method == 'LagrangeFR':
        SpatialClass = spatial.LagrangeFR
    elif args.method == 'LegendreDG':
        SpatialClass = spatial.LegendreDG
    elif args.method == 'LegendreFR':
        SpatialClass = spatial.LegendreFR
    elif args.method == 'DGwithFR':
        SpatialClass = spatial.DGwithFR
    else:
        assert False
    if args.detector == 'ReportAll':
        DetectorClass = detector.ReportAll
    elif args.detector == 'Krivodonova2004':
        DetectorClass = detector.Krivodonova2004
    elif args.detector == 'LiRen2011':
        DetectorClass = detector.LiRen2011
    elif args.detector == 'ZhuShuQiu2021':
        DetectorClass = detector.ZhuShuQiu2021
    else:
        assert False
    if args.limiter == 'LiWangRen2020':
        LimiterClass = limiter.LiWangRen2020
    elif args.limiter == 'ZhongShu2013':
        LimiterClass = limiter.ZhongShu2013
    else:
        assert False
    if args.problem == 'Linear':
        SolverClass = LinearAdvection
        EquationClass = equation.LinearAdvection
        RiemannClass = riemann.LinearAdvection
    elif args.problem == 'Burgers':
        SolverClass = InviscidBurgers
        EquationClass = equation.InviscidBurgers
        RiemannClass = riemann.InviscidBurgers
    else:
        assert False
    solver = SolverClass(args.phase_speed, args.wave_number,
        spatial_scheme=SpatialClass(
            EquationClass(args.phase_speed), RiemannClass(args.phase_speed),
            args.degree, args.n_element, args.x_left, args.x_right),
        detector=DetectorClass(), limiter=LimiterClass(),
        ode_solver=temporal.RungeKutta(args.rk_order))
    solver.animate(args.t_begin, args.t_end, args.n_step)
