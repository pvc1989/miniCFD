"""Analyze modified wavenumbers for various spatial schemes.
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt

import spatial
import riemann


class WaveNumberDisplayer:
    """Plot modified-wavenumbers for various spatial schemes.
    """

    def __init__(self, x_left, x_right, n_element) -> None:
        self._a = 1.0
        self._riemann = riemann.LinearAdvection(self._a, complex)
        self._x_left = x_left
        self._x_right = x_right
        self._n_element = n_element

    def build_scheme(self, Method, degree: int):
        assert issubclass(Method, spatial.FiniteElement)
        scheme = Method(self._riemann,
            degree, self._n_element, self._x_left, self._x_right)
        return scheme

    def get_spatial_matrix(self, scheme: spatial.FiniteElement, kappa_h: float):
        """Get the spatial matrix of a FiniteElement scheme.
        """
        assert isinstance(scheme, spatial.FiniteElement)
        # kappa_h = k_int * 2 * np.pi / scheme.length() * scheme.delta_x(0)
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

    def get_modified_wavenumbers(self, Method, degree: int,
            sampled_wavenumbers: np.ndarray):
        """Get the eigenvalues of a scheme at a given set of wavenumbers.
        """
        n_sample = len(sampled_wavenumbers)
        n_term = degree + 1
        modified_wavenumbers = np.ndarray((n_sample, n_term), dtype=complex)
        scheme = self.build_scheme(Method, degree)
        for i_sample in range(n_sample):
            kappa_h = sampled_wavenumbers[i_sample]
            matrix = self.get_spatial_matrix(scheme, kappa_h)
            matrix *= 1j * scheme.delta_x(0) / self._a
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

    def plot_modified_wavenumbers(self, Method, degree: int, n_sample: int):
        """Plot the tilde-kappa_h - kappa_h curves for a given scheme.
        """
        xticks_labels = np.linspace(-degree-1, degree+1, 2*degree+3, dtype=int)
        # xticks_labels = np.linspace(0, degree+1, degree+2, dtype=int)
        xticks_ticks = xticks_labels * np.pi
        kh_min, kh_max = xticks_ticks[0], xticks_ticks[-1]
        sampled_wavenumbers = np.linspace(kh_min, kh_max, n_sample)
        modified_wavenumbers = self.get_modified_wavenumbers(Method,
            degree, sampled_wavenumbers)
        physical_eigvals = self.get_physical_mode(sampled_wavenumbers,
            modified_wavenumbers)
        plt.figure(figsize=(6,9))
        plt.subplot(3,1,1)
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.real, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.real, 'ro', label='Physical')
        plt.plot([kh_min, kh_max], [kh_min, kh_max], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.subplot(3,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        plt.plot(sampled_wavenumbers, modified_wavenumbers.imag, 'k.')
        plt.plot(sampled_wavenumbers, physical_eigvals.imag, 'ro', label='Physical')
        plt.plot([kh_min, kh_max], [0, 0], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.subplot(3,1,3)
        plt.ylabel(r'$|\tilde{\kappa}h|$')
        plt.xlabel(r'$\kappa h\,/\,\pi$')
        norms = np.sqrt(modified_wavenumbers.real**2 + modified_wavenumbers.imag**2)
        plt.plot(sampled_wavenumbers, norms, 'k.')
        norms = np.sqrt(physical_eigvals.real**2 + physical_eigvals.imag**2)
        plt.plot(sampled_wavenumbers, norms, 'ro', label='Physical')
        plt.plot([kh_min, 0, kh_max], [-kh_min, 0, kh_max], '-', label='Exact')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        scheme = self.build_scheme(Method, degree)
        plt.savefig(f'all_modes_of_{scheme.name(False)}_p={degree}.pdf')

    def compare_wave_numbers(self, methods, degrees, n_sample: int,
            compressed=False):
        linestyles = [
            ('dotted',                (0, (1, 1))),
            ('loosely dotted',        (0, (1, 4))),
            ('dashed',                (0, (4, 4))),
            ('densely dashed',        (0, (4, 1))),
            ('loosely dashed',        (0, (4, 8))),
            ('long dash with offset', (4, (8, 2))),
            ('dashdotted',            (0, (2, 4, 1, 4))),
            ('densely dashdotted',    (0, (2, 1, 1, 1))),
            ('loosely dashdotted',    (0, (2, 8, 1, 8))),
            ('dashdotdotted',         (0, (2, 4, 1, 4, 1, 4))),
            ('densely dashdotdotted', (0, (2, 1, 1, 1, 1, 1))),
            ('loosely dashdotdotted', (0, (2, 8, 1, 8, 1, 8))),]
        if compressed:
            divisor = r'$(N\pi)$'
        else:
            divisor = r'$\pi$'
        plt.figure(figsize=(6,9))
        plt.subplot(2,1,1)
        plt.ylabel(r'$\Re(\tilde{\kappa}h)\,/\,$'+divisor)
        plt.xlabel(r'$\kappa h\,/\,$'+divisor)
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)\,/\,$'+divisor)
        plt.xlabel(r'$\kappa h\,/\,$'+divisor)
        i = 0
        for degree in degrees:
            kh_max = (degree + 1) * np.pi
            sampled_wavenumbers = np.linspace(0, kh_max, n_sample)
            scale = (degree * compressed + 1) * np.pi
            for method in methods:
                scheme = self.build_scheme(method, degree)
                modified_wavenumbers = self.get_modified_wavenumbers(method,
                    degree, sampled_wavenumbers)
                physical_eigvals = self.get_physical_mode(sampled_wavenumbers,
                    modified_wavenumbers)
                plt.subplot(2,1,1)
                plt.plot(sampled_wavenumbers/scale, physical_eigvals.real/scale,
                    label=scheme.name(), linestyle=linestyles[i][1])
                plt.subplot(2,1,2)
                plt.plot(sampled_wavenumbers/scale, physical_eigvals.imag/scale,
                    label=scheme.name(), linestyle=linestyles[i][1])
                i += 1
        x_max = np.max(degrees) * (not compressed) + 1
        plt.subplot(2,1,1)
        plt.plot([0, x_max], [0, x_max], '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.subplot(2,1,2)
        plt.plot([0, x_max], [0, 0], '-', label='Exact')
        plt.grid()
        plt.legend(handlelength=4)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'compare_wave_numbers.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python3 wave_number.py')
    parser.add_argument('-m', '--method',
        choices=['LagrangeDG', 'LagrangeFR', 'LegendreDG', 'LegendreFR'],
        default='LagrangeFR',
        help='method for spatial discretization')
    parser.add_argument('-d', '--degree',
        default=2, type=int,
        help='degree of polynomials for approximation')
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
    parser.add_argument('-c', '--compressed',
        action='store_true',
        help='whether the range of input wavenumbers is compressed')
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
    wnd = WaveNumberDisplayer(args.x_left, args.x_right, args.n_element)
    wnd.plot_modified_wavenumbers(SpatialClass, args.degree, args.n_sample)
    wnd.compare_wave_numbers(methods=[spatial.LegendreDG, spatial.LagrangeFR,
        spatial.LegendreFR], degrees=[1, 3, 5], n_sample=args.n_sample,
        compressed=args.compressed)
