"""Analyze modified wavenumbers for various spatial schemes.
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt

import spatial
import equation
import riemann
import temporal


class WaveNumberDisplayerOnDFT:
    """Plot modified-wavenumbers for various spatial schemes using DFT.
    """

    def __init__(self, x_left, x_right, n_element, tau,
            n_sample_per_element) -> None:
        self._a = 1.0
        self._equation = equation.LinearAdvection(self._a)
        self._riemann = riemann.LinearAdvection(self._a)
        self._tau = tau
        self._x_left = x_left
        self._x_right = x_right
        self._n_element = n_element
        delta_x = (x_right - x_left) / n_element
        n_sample = self._n_element * n_sample_per_element
        self._w = np.exp(1.0j * 2 * np.pi / n_sample)
        half_sample_gap = (delta_x / n_sample_per_element) / 2
        self._sample_points = np.linspace(self._x_left+half_sample_gap,
            self._x_right-half_sample_gap, n_sample)

    def get_kth_fourier_coeff(self, k, u):
        """Get the kth fourier coefficient of a scalar-valued function u(x).
        """
        n_sample = len(self._sample_points)
        kth_fourier_coeff = 0.0
        for j in range(n_sample):
            x_j = self._sample_points[j]
            u_j = u(x_j)
            kth_fourier_coeff += u_j * self._w**(-j*k)
        return kth_fourier_coeff / n_sample

    def get_wavenumbers(self, spatial_scheme: spatial.PiecewiseContinuous):
        """Get the reduced and modified wavenumbers of a PiecewiseContinuous scheme.
        """
        k_max = spatial_scheme.n_element() // 2
        k_max *= spatial_scheme.get_element(0.0).n_term()
        reduced_wavenumbers = np.ndarray(k_max)
        modified_wavenumbers = np.ndarray(k_max, complex)
        ode_solver = temporal.RungeKutta(3)
        for k in range(1, 1 + k_max):
            kappa = k * np.pi / (spatial_scheme.length() / 2)
            reduced_wavenumbers[k-1] = kappa * spatial_scheme.delta_x()
            def u_init(x):
                return np.exp(1.0j * kappa * x)
            kth_fourier_of_u_init = 0.0j
            kth_fourier_of_u_tau = 0.0j
            # solve the real part
            spatial_scheme.initialize(lambda x: u_init(x).real)
            kth_fourier_of_u_init += self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            ode_solver.update(spatial_scheme, delta_t=self._tau)
            kth_fourier_of_u_tau += self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            # solve the imag part
            spatial_scheme.initialize(lambda x: u_init(x).imag)
            kth_fourier_of_u_init += 1j * self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            ode_solver.update(spatial_scheme, delta_t=self._tau)
            kth_fourier_of_u_tau += 1j * self.get_kth_fourier_coeff(k,
                lambda x: spatial_scheme.get_solution_value(x))
            # put together
            modified_wavenumbers[k-1] = (1.0j * spatial_scheme.delta_x()
                / (self._a * self._tau)
                * np.log(kth_fourier_of_u_tau / kth_fourier_of_u_init))
        return reduced_wavenumbers, modified_wavenumbers

    def plot(self, schemes, labels, xticks_ticks, xticks_labels):
        linestyles = [
            ('densely dotted',        (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('dashed',                (0, (5, 5))),
            ('densely dashed',        (0, (5, 1))),
            ('dashdotted',            (0, (3, 5, 1, 5))),
            ('densely dashdotted',    (0, (3, 1, 1, 1))),
            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        kh_max = xticks_ticks[-1]
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot([0, kh_max], [0, kh_max], '-', label='Exact')
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.plot([0, kh_max], [0, 0], '-', label='Exact')
        for i in range(len(labels)):
            reduced, modified = self.get_wavenumbers(schemes[i])
            # reduced /= (i + 2)
            # modified /= (i + 2)
            plt.subplot(2,1,1)
            plt.plot(reduced, modified.real, label=labels[i],
                linestyle=linestyles[i][1])
            plt.subplot(2,1,2)
            plt.plot(reduced, modified.imag, label=labels[i],
                linestyle=linestyles[i][1])
        plt.subplot(2,1,1)
        plt.legend()
        plt.subplot(2,1,2)
        plt.legend()
        plt.tight_layout()

    def compare_degrees(self, method: spatial.PiecewiseContinuous):
        """Compare spatial schemes using the same method but different degrees.
        """
        degrees = np.arange(0, 4, step=1, dtype=int)
        schemes = []
        labels = []
        xticks_ticks = [0.0]
        xticks_labels = ['0']
        for degree in degrees:
            schemes.append(method(self._equation, self._riemann,
                degree, self._n_element, self._x_left, self._x_right))
            order = degree + 1
            labels.append(f'{method.name()}{order}')
            xticks_ticks.append(order * np.pi)
            xticks_labels.append(f'${order}\pi$')
        self.plot(schemes, labels, xticks_ticks, xticks_labels)
        # plt.show()
        plt.savefig(f'compare_{method.name()}.pdf')

    def compare_methods(self, degree: int):
        """Compare spatial schemes using the same degree but different methods.
        """
        methods = [
            spatial.LegendreDG,
            spatial.LagrangeDG,
            spatial.LegendreFR,
            spatial.LagrangeFR,
            spatial.DGwithFR,
        ]
        order = degree + 1
        xticks_ticks = [0.0]
        xticks_labels = ['0']
        for p in range(1, order + 1):
            xticks_ticks.append(p * np.pi)
            xticks_labels.append(f'${p}\pi$')
        schemes = []
        labels = []
        for method in methods:
            schemes.append(method(self._equation, self._riemann,
                degree, self._n_element, self._x_left, self._x_right))
            labels.append(f'{method.name()}{order}')
        self.plot(schemes, labels, xticks_ticks, xticks_labels)
        # plt.show()
        plt.savefig(f'compare_{degree}-degree_methods.pdf')

    def compare_all_degrees(self):
        self.compare_degrees(spatial.LegendreDG)
        self.compare_degrees(spatial.LagrangeDG)
        self.compare_degrees(spatial.LegendreFR)
        self.compare_degrees(spatial.LagrangeFR)
        self.compare_degrees(spatial.DGwithFR)

    def compare_all_methods(self):
        self.compare_methods(2)
        self.compare_methods(4)


class WaveNumberDisplayer:
    """Plot modified-wavenumbers for various spatial schemes.
    """

    def __init__(self, x_left, x_right, n_element) -> None:
        self._a = 1.0
        self._equation = equation.LinearAdvection(self._a)
        self._riemann = riemann.LinearAdvection(self._a)
        self._x_left = x_left
        self._x_right = x_right
        self._n_element = n_element

    def build_scheme(self, method: spatial.PiecewiseContinuous, degree: int):
        scheme = method(self._equation, self._riemann,
            degree, self._n_element, self._x_left, self._x_right, complex)
        return scheme

    def get_spatial_matrix(self, scheme: spatial.PiecewiseContinuous,
            kappa_h: float):
        """Get the spatial matrix of a PiecewiseContinuous scheme.
        """
        # kappa_h = k_int * 2 * np.pi / scheme.length() * scheme.delta_x()
        n_term = scheme.degree() + 1
        matrices = np.zeros((self._n_element, n_term, n_term), dtype=complex)
        for col in range(n_term):
            u_tilde = np.zeros(n_term)
            u_tilde[col] = 1
            global_column = np.ndarray(n_term * scheme.n_element(),
                dtype=complex)
            for i_element in range(scheme.n_element()):
                first = i_element * n_term
                last = first + n_term
                global_column[first:last] = u_tilde * np.exp(1j * i_element * kappa_h)
            scheme.set_solution_column(global_column)
            global_column = scheme.get_residual_column()
            for i_element in range(scheme.n_element()):
                matrix = matrices[i_element]
                assert matrix.shape == (n_term, n_term)
                first = i_element * n_term
                last = first + n_term
                matrix[:, col] = global_column[first:last]
                matrix[:, col] /= np.exp(1j * i_element * kappa_h)
        return matrices[-1]

    def get_modified_wavenumbers(self, method: spatial.PiecewiseContinuous,
            degree: int, sampled_wavenumbers: np.ndarray):
        """Get the eigenvalues of a scheme at a given set of wavenumbers.
        """
        n_sample = len(sampled_wavenumbers)
        n_term = degree + 1
        modified_wavenumbers = np.ndarray((n_sample, n_term), dtype=complex)
        scheme = self.build_scheme(method, degree)
        for i_sample in range(n_sample):
            kappa_h = sampled_wavenumbers[i_sample]
            matrix = self.get_spatial_matrix(scheme, kappa_h)
            matrix *= 1j * scheme.delta_x() / self._a
            modified_wavenumbers[i_sample, :] = np.linalg.eigvals(matrix)
        return modified_wavenumbers

    def get_physical_mode(self, sampled_wavenumbers: np.ndarray,
            modified_wavenumbers: np.ndarray):
        n_sample, n_term = modified_wavenumbers.shape
        assert n_sample == len(sampled_wavenumbers)
        physical_eigvals = np.ndarray(n_sample, dtype=complex)
        for i_sample in range(n_sample):
            eigvals = modified_wavenumbers[i_sample]
            # sort the eigvals by their norms
            norms = eigvals.real**2 + eigvals.imag**2
            pairs = np.ndarray(n_term, dtype=[('x', complex), ('y', float)])
            for i_term in range(n_term):
                pairs[i_term] = (eigvals[i_term], norms[i_term])
            pairs.sort(order='y')
            i_interval = int(np.floor(sampled_wavenumbers[i_sample] / np.pi))
            while i_interval > n_term:
                i_interval -= n_term * 2
            while i_interval <= -n_term:
                i_interval += n_term * 2
            assert -n_term < i_interval <= n_term
            if i_interval < 0:
                i_mode = -1 - i_interval
            else:
                i_mode = i_interval - (i_interval == n_term)
            assert 0 <= i_mode < n_term
            physical_eigvals[i_sample] = pairs[i_mode][0]
        return physical_eigvals

    def plot_modified_wavenumbers(self, method: spatial.PiecewiseContinuous,
            degree: int, n_sample: int):
        """Plot the tilde-kappa_h - kappa_h curves for a given scheme.
        """
        xticks_labels = np.linspace(-degree-1, degree+1, 2*degree+3, dtype=int)
        # xticks_labels = np.linspace(0, degree+1, degree+2, dtype=int)
        xticks_ticks = xticks_labels * np.pi
        kh_min, kh_max = xticks_ticks[0], xticks_ticks[-1]
        sampled_wavenumbers = np.linspace(kh_min, kh_max, n_sample)
        modified_wavenumbers = self.get_modified_wavenumbers(method,
            degree, sampled_wavenumbers)
        physical_eigvals = self.get_physical_mode(sampled_wavenumbers,
            modified_wavenumbers)
        plt.figure(figsize=(6,9))
        plt.subplot(3,1,1)
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h/\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.real, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.real, 'r+', label='Physical')
        plt.plot([kh_min, kh_max], [kh_min, kh_max], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.subplot(3,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h/\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.imag, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.imag, 'r+', label='Physical')
        plt.plot([kh_min, kh_max], [0, 0], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.subplot(3,1,3)
        plt.ylabel(r'$|\tilde{\kappa}h|$')
        plt.xlabel(r'$\kappa h/\pi$')
        norms = np.sqrt(modified_wavenumbers.real**2 + modified_wavenumbers.imag**2)
        plt.plot(sampled_wavenumbers, norms, 'k.')
        norms = np.sqrt(physical_eigvals.real**2 + physical_eigvals.imag**2)
        plt.plot(sampled_wavenumbers, norms, 'r+', label='Physical')
        plt.plot([kh_min, 0, kh_max], [-kh_min, 0, kh_max], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'all_modes_of_{method.name()}{degree+1}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python3 wave_number.py')
    parser.add_argument('-n', '--n_element',
        default=10, type=int,
        help='number of elements')
    parser.add_argument('-s', '--n_sample',
        default=50, type=int,
        help='number of sample points')
    parser.add_argument('-l', '--x_left',
        default=0.0, type=float,
        help='coordinate of the left end of the domain')
    parser.add_argument('-r', '--x_right',
        default=10.0, type=float,
        help='coordinate of the right end of the domain')
    parser.add_argument('-m', '--method',
        choices=['LagrangeDG', 'LagrangeFR', 'LegendreDG', 'LegendreFR',
            'DGwithFR'],
        default='LagrangeDG',
        help='method for spatial discretization')
    parser.add_argument('-d', '--degree',
        default=2, type=int,
        help='degree of polynomials for approximation')
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
    wnd = WaveNumberDisplayer(args.x_left, args.x_right, args.n_element)
    wnd.plot_modified_wavenumbers(SpatialClass, args.degree, args.n_sample)
    exit(0)
    wnd = WaveNumberDisplayerOnDFT(x_left=0.0, x_right=2000.0, n_element=20,
        tau=0.0001, n_sample_per_element = 10)
    wnd.compare_all_methods()
    wnd.compare_all_degrees() # time-consuming for large degree
