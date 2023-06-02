"""Put separated modules together.
"""
import abc
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as mpla
from scipy import optimize
import vtk

import concept
import riemann
import spatial, detector, limiter, viscous
import temporal


class SolverBase(abc.ABC):
    """Define common methods for all solvers.
    """

    def __init__(self, spatial_scheme: concept.SpatialScheme,
            d: concept.Detector, l: concept.Limiter, v: concept.Viscous,
            ode_solver: concept.OdeSolver):
        self._spatial = spatial_scheme
        self._spatial.set_detector_and_limiter(d, l, v)
        self._ode_solver = ode_solver
        self._solver_name = f'spatial.{self._spatial.name()}'
        if not isinstance(d, detector.Off):
            self._solver_name += f', detector.{d.name()}'
        if not isinstance(l, limiter.Off):
            self._solver_name += f', limiter.{l.name()}'
        if not isinstance(v, viscous.Off):
            self._solver_name += f', viscous.{v.name(True)}'
        self._animation = None
        self._output_points = np.linspace(self._spatial.x_left(),
            self._spatial.x_right(), 201)

    @abc.abstractmethod
    def problem_name(self) -> str:
        """Get a string representation of the problem."""

    def solver_name(self) -> str:
        return self._solver_name

    @abc.abstractmethod
    def u_init(self, x_global):
        """Initial condition of the problem."""

    @abc.abstractmethod
    def u_exact(self, x_global, t_curr):
        """Exact solution of the problem."""

    def _get_figure(self):
        fig = plt.figure(figsize=(8, 6))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$')
        plt.ylim([-1.4, 1.6])
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            self._spatial.n_element() + 1), minor=True)
        plt.xticks(np.linspace(self._spatial.x_left(), self._spatial.x_right(),
            5), minor=False)
        plt.grid(which='minor')
        return fig

    def _get_ydata(self, t_curr, points):
        n_point = len(points)
        expect_solution = np.ndarray(n_point)
        approx_solution = np.ndarray(n_point)
        for i in range(n_point):
            point_i = points[i]
            approx_solution[i] = self._spatial.get_solution_value(point_i)
            expect_solution[i] = self.u_exact(point_i, t_curr)
        return expect_solution, approx_solution

    def _measure_errors(self, t_curr):
        error_1, error_2, error_infty = 0, 0, 0
        n_element = self._spatial.n_element()
        for i_element in range(n_element):
            element_i = self._spatial.get_element_by_index(i_element)
            def error(x_global):
                value = self.u_exact(x_global, t_curr)
                value -= element_i.get_solution_value(x_global)
                return value
            n_point = element_i.n_term()
            error_1 += element_i.integrator().norm_1(error, n_point)
            error_2 += element_i.integrator().norm_2(error, n_point)**2
            error_infty = max(error_infty,
                element_i.integrator().norm_infty(error, n_point*2))
        return error_1, np.sqrt(error_2), error_infty

    def _write_to_pdf(self, filename: str, t_curr: float):
        """Solve the problem in a given time range and write to pdf files.
        """
        points = self._output_points
        expect_solution, approx_solution = self._get_ydata(t_curr, points)
        fig = self._get_figure()
        plt.plot(points, approx_solution, 'b-',
            label=self.solver_name())
        plt.plot(points, expect_solution, 'r--',
            label=f'Exact Solution of {self.problem_name()}')
        plt.title(r'$t$'+f' = {t_curr:.2f}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        # plt.show()
        plt.savefig(filename)
        error_1, error_2, error_infty = self._measure_errors(t_curr)
        print('t_curr, error_1, error_2, error_∞ =',
            f'[ {t_curr}, {error_1:6e}, {error_2:6e}, {error_infty:6e} ],')
        plt.close(fig)

    def _write_to_vtu(self, filename: str, binary=True):
        grid = vtk.vtkUnstructuredGrid()
        vtk_points = vtk.vtkPoints()
        scalar_on_points = vtk.vtkFloatArray()
        scalar_on_points.SetName("U")
        scalar_on_points.SetNumberOfComponents(1)
        for x in self._output_points:
            vtk_points.InsertNextPoint((x, 0, 0))
            value = self._spatial.get_solution_value(x)
            scalar_on_points.InsertNextValue(value)
        grid.SetPoints(vtk_points)
        grid.GetPointData().SetScalars(scalar_on_points)
        writer = vtk.vtkXMLDataSetWriter()
        writer.SetInputData(grid)
        writer.SetFileName(filename)
        if binary:
            writer.SetDataModeToBinary()
        else:
            writer.SetDataModeToAscii()
        writer.Write()

    def solve_and_write(self, t_start: float, t_stop: float,  n_step: int,
            n_frame: int, output: str):
        """Solve the problem in a given time range and write to vtu/pdf files.
        """
        dt_max = (t_stop - t_start) / n_step
        dt_per_frame = (t_stop - t_start) / n_frame
        self._spatial.initialize(lambda x_global: self.u_init(x_global))
        t_curr = t_start
        for i_frame in range(n_frame+1):
            print(f'i_frame = {i_frame}, t = {t_curr:.2f}')
            if output == 'pdf':
                self._write_to_pdf(f'Frame{i_frame}.pdf', t_curr)
            elif output == 'vtu':
                self._write_to_vtu(f'Frame{i_frame}.vtu')
            else:
                assert False
            if i_frame == n_frame:
                break
            t_next = t_curr + dt_per_frame
            while t_curr < t_next:
                dt_suggested = self._spatial.suggest_delta_t(t_next - t_curr)
                dt_actual = min(dt_max, dt_suggested)
                print(f't = {t_curr:.3f}, dt_max = {dt_max:.2e}',
                    f', dt_suggested = {dt_suggested:.2e}',
                    f', dt_actual = {dt_actual:.2e}')
                self._ode_solver.update(self._spatial, dt_actual, t_curr)
                t_curr += dt_actual

    def animate(self, t_start: float, t_stop: float,  n_step: int,
            n_frame: int):
        """Solve the problem in a given time range and animate the results.
        """
        dt_max = (t_stop - t_start) / n_step
        dt_per_frame = (t_stop - t_start) / n_frame
        fig = self._get_figure()
        # initialize line-plot objects
        approx_line, = plt.plot([], [], 'b-', label=self.solver_name())
        expect_line, = plt.plot([], [], 'r-.',
            label=f'Exact Solution of {self.problem_name()}')
        points = self._output_points
        # initialize animation
        def init_func():
            self._spatial.initialize(lambda x_global: self.u_init(x_global))
            expect_line.set_xdata(points)
            approx_line.set_xdata(points)
        # update data for the next frame
        def update_func(t_curr):
            expect_solution, approx_solution = self._get_ydata(t_curr, points)
            expect_line.set_ydata(expect_solution)
            approx_line.set_ydata(approx_solution)
            print(f't = {t_curr:.3f}')
            plt.title(r'$t=$'+f'{t_curr:.2f}')
            plt.legend(loc='upper right')
            t_next = min(t_stop, t_curr + dt_per_frame)
            while t_curr < t_next:
                dt_suggested = self._spatial.suggest_delta_t(t_next - t_curr)
                dt_actual = min(dt_max, dt_suggested)
                print(f't = {t_curr:.3f}, dt_max = {dt_max:.2e}',
                    f', dt_suggested = {dt_suggested:.2e}',
                    f', dt_actual = {dt_actual:.2e}')
                self._ode_solver.update(self._spatial, dt_actual, t_curr)
                t_curr += dt_actual
        frames = np.linspace(t_start, t_stop, n_frame+1)
        self._animation = mpla.FuncAnimation(
            plt.gcf(), update_func, frames, init_func, interval=5)
        plt.show()


class LinearSmooth(SolverBase):
    """Demo the usage of LinearAdvection related classes for smooth IC.
    """

    def __init__(self, wave_number: int,
            spatial_scheme: concept.SpatialScheme,
            detector: concept.Detector, limiter: concept.Limiter,
            viscous: concept.Viscous,
            ode_solver: concept.OdeSolver) -> None:
        super().__init__(spatial_scheme, detector, limiter, viscous, ode_solver)
        self._a_const = self._spatial.equation().get_convective_speed()
        self._b_const = self._spatial.equation().get_diffusive_coeff()
        self._k_const = wave_number
        self._wave_number = self._k_const * np.pi * 2 / self._spatial.length()

    def problem_name(self):
        my_name = self._spatial.equation().name() + r', $u(x,t=0)=\sin($'
        length = self._spatial.length() / 2
        my_name += f'{self._k_const:g}' + r'$\pi x/$' + f'{length:g}' + r'$)$'
        return my_name

    def u_init(self, x_global):
        x_global = x_global - self._spatial.x_left()
        value = np.sin(x_global * self._wave_number)
        return value

    def u_exact(self, x_global, t_curr):
        ratio = np.exp(-t_curr * self._b_const * self._wave_number**2)
        return ratio * self.u_init(x_global - self._a_const * t_curr)


class LinearJumps(LinearSmooth):
    """Demo the usage of LinearAdvection related classes for IC with jumps.
    """

    def __init__(self, wave_number: int,
            spatial_scheme: concept.SpatialScheme,
            detector: concept.Detector, limiter: concept.Limiter,
            viscous: concept.Viscous,
            ode_solver: concept.OdeSolver) -> None:
        LinearSmooth.__init__(self, wave_number, spatial_scheme,
            detector, limiter, viscous, ode_solver)

    def u_init(self, x_global):
        value = LinearSmooth.u_init(self, x_global)
        return np.sign(value)

    def problem_name(self):
        my_name = self._spatial.equation().name() + r', $u(x,t=0)=$sign$(\sin($'
        length = self._spatial.length() / 2
        my_name += f'{self._k_const:g}' + r'$\pi x/$' + f'{length:g}' + r'$))$'
        return my_name


class InviscidBurgers(SolverBase):
    """Demo the usage of InviscidBurgers related classes.
    """

    def __init__(self, wave_number: int,
            spatial_scheme: concept.SpatialScheme,
            detector: concept.Detector, limiter: concept.Limiter,
            viscous: concept.Viscous,
            ode_solver: concept.OdeSolver) -> None:
        super().__init__(spatial_scheme, detector, limiter, viscous, ode_solver)
        self._k_const = wave_number
        self._wave_number = self._k_const * np.pi * 2 / self._spatial.length()
        self._u_prev = dict()

    def problem_name(self):
        my_name = self._spatial.equation().name() + r', $u(x,t=0)=\sin($'
        length = self._spatial.length() / 2
        my_name += f'{self._k_const:g}' + r'$\pi x/$' + f'{length:g}' + r'$)$'
        return my_name

    def u_init(self, x_global):
        x_global = x_global - self._spatial.x_left()
        value = np.sin(x_global * self._wave_number)
        return value

    def u_exact(self, x_global, t_curr):
        """ Solve u_curr from u_curr = u_init(x - a(u_curr) * t_curr).

        Currently, the solution is only correct for standing waves in t <= 10.
        """
        def func(u_curr):
            ku = self._spatial.equation().get_convective_speed(u_curr)
            return u_curr - self.u_init(x_global - ku * t_curr)
        if x_global in self._u_prev:
            u_prev = self._u_prev[x_global]
        else:
            u_prev = self.u_init(x_global)
        delta = 0.02
        u_min = u_prev - delta
        u_max = u_prev + delta
        while func(u_min) * func(u_max) > 0:
            u_min -= delta
            u_max += delta
        root = optimize.bisect(func, u_min, u_max)
        self._u_prev[x_global] = root
        return root


class EulerRiemann(SolverBase):
    """Demo the usage of Euler related classes.
    """

    def __init__(self,
            spatial_scheme: concept.SpatialScheme,
            detector: concept.Detector, limiter: concept.Limiter,
            viscous: concept.Viscous,
            ode_solver: concept.OdeSolver) -> None:
        super().__init__(spatial_scheme, detector, limiter, viscous, ode_solver)
        self._x_mid = (self._spatial.x_left() + self._spatial.x_right()) / 2
        self._riemann = riemann.Euler(gamma=1.4)
        self._value_left = self._riemann.equation().primitive_to_conservative(
              rho=1.0, u=0, p=1.0)
        self._value_right = self._riemann.equation().primitive_to_conservative(
              rho=0.125, u=0, p=0.1)
        self._riemann.set_initial(self._value_left, self._value_right)

    def u_init(self, x_global):
        if x_global < self._x_mid:
            return self._value_left
        else:
            return self._value_right

    def u_exact(self, x_global, t_curr):
        return self._riemann.U(x_global - self._x_mid, t_curr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python3 solver.py',
        description = 'What the program does',
        epilog = 'Text at the bottom of help')
    parser.add_argument('-n', '--n_element',
        default=23, type=int,
        help='number of elements')
    parser.add_argument('-l', '--x_left',
        default=-1.0, type=float,
        help='coordinate of the left end of the domain')
    parser.add_argument('-r', '--x_right',
        default=+1.0, type=float,
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
        default=500, type=int,
        help='number of time steps')
    parser.add_argument('-m', '--method',
        choices=['LagrangeDG', 'LagrangeFR', 'LegendreDG', 'LegendreFR'],
        default='LagrangeFR',
        help='method for spatial discretization')
    parser.add_argument('--detector',
        choices=['Off', 'All', 'KXRCF', 'LWA', 'LYH', 'ZJ', 'Persson'],
        default='Off',
        help='method for detecting jumps')
    parser.add_argument('--limiter',
        choices=['Off', 'LWA', 'ZXH', 'XXR'],
        default='Off',
        help='method for limiting numerical oscillations')
    parser.add_argument('--viscous_model',
        choices=['Off', 'Constant', 'Persson', 'Energy'],
        default='Off',
        help='method for adding artificial viscosity')
    parser.add_argument('--viscous_model_const',
        default=0.0, type=float,
        help='constant in viscous model')
    parser.add_argument('-d', '--degree',
        default=2, type=int,
        help='degree of polynomials for approximation')
    parser.add_argument('-p', '--problem',
        choices=['Smooth', 'Jumps', 'Burgers', 'Euler'],
        default='Smooth',
        help='problem to be solved')
    parser.add_argument('-a', '--convection_speed',
        default=1.0, type=float,
        help='phase speed of the wave')
    parser.add_argument('-v', '--physical_viscosity',
        default=0.0, type=float,
        help='physical viscous coeff of the equation')
    parser.add_argument('-k', '--wave_number',
        default=1, type=int,
        help='number of waves in the domain')
    parser.add_argument('--output',
        choices=['fig', 'pdf', 'vtu'],
        default='fig',
        help='type of output')
    parser.add_argument('--n_frame',
        default=100, type=int,
        help='number of frames to be written')
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
    else:
        assert False
    if args.detector == 'Off':
        the_detector = detector.Off()
    elif args.detector == 'All':
        the_detector = detector.All()
    elif args.detector == 'KXRCF':
        the_detector = detector.Krivodonova2004()
    elif args.detector == 'LWA':
        the_detector = detector.LiWanAi2011()
    elif args.detector == 'LYH':
        the_detector = detector.LiYanHui2022()
    elif args.detector == 'ZJ':
        the_detector = detector.ZhuJun2021()
    elif args.detector == 'Persson':
        the_detector = detector.Persson2006()
    else:
        assert False
    if args.limiter == 'LWA':
        the_limiter = limiter.LiWanAi2020()
    elif args.limiter == 'ZXH':
        the_limiter = limiter.ZhongXingHui2013()
    elif args.limiter == 'XXR':
        the_limiter = limiter.XuXiaoRui2023()
    elif args.limiter == 'Off':
        the_limiter = limiter.Off()
    else:
        assert False
    if args.viscous_model == 'Off':
        the_viscous = viscous.Off()
    elif args.viscous_model == 'Constant':
        the_viscous = viscous.Constant(args.viscous_model_const)
    elif args.viscous_model == 'Persson':
        the_viscous = viscous.Persson2006(args.viscous_model_const)
    elif args.viscous_model == 'Energy':
        the_viscous = viscous.Energy(args.viscous_model_const)
    else:
        assert False
    if args.problem == 'Smooth':
        SolverClass = LinearSmooth
        the_riemann = riemann.LinearAdvectionDiffusion(args.convection_speed,
            args.physical_viscosity)
    elif args.problem == 'Jumps':
        SolverClass = LinearJumps
        the_riemann = riemann.LinearAdvection(args.convection_speed)
    elif args.problem == 'Burgers':
        SolverClass = InviscidBurgers
        the_riemann = riemann.InviscidBurgers(args.convection_speed)
    elif args.problem == 'Euler':
        SolverClass = EulerRiemann
        the_riemann = riemann.Euler(gamma=1.4)
    else:
        assert False
    solver = SolverClass(args.wave_number,
        SpatialClass(the_riemann,
            args.degree, args.n_element, args.x_left, args.x_right),
        the_detector, the_limiter, the_viscous,
        ode_solver=temporal.RungeKutta(args.rk_order))
    if args.output == 'fig':
        solver.animate(args.t_begin, args.t_end, args.n_step, args.n_frame)
    else:
        solver.solve_and_write(args.t_begin, args.t_end, args.n_step,
            args.n_frame, args.output)

