"""Analyze modified wavenumbers for various spatial schemes.
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt

import spatial
import equation
import riemann
import temporal


class PlotModifiedWavenumbers(unittest.TestCase):
    """Plot modified-wavenumbers for various spatial schemes.
    """

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self._a = 1.0
        self._equation = equation.LinearAdvection(self._a)
        self._riemann = riemann.LinearAdvection(self._a)
        self._tau = 0.0001
        self._n_element = 20
        delta_x = 100.0
        self._x_left = 0.0
        self._x_right = delta_x * self._n_element
        n_sample_per_element = 10
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

    def compare_degrees(self, method: str):
        """Compare spatial schemes using the same method but different degrees.
        """
        if method == 'DG':
            SpatialScheme = spatial.LagrangeDG
        elif method == 'FR':
            SpatialScheme = spatial.LagrangeFR
        elif method == 'DGFR':
            SpatialScheme = spatial.DGwithLagrangeFR
        else:
            assert False
        linestyles = [
            ('dotted',                (0, (1, 1))),
            ('densely dotted',        (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('dashed',                (0, (5, 5))),
            ('densely dashed',        (0, (5, 1))),
            ('dashdotted',            (0, (3, 5, 1, 5))),
            ('densely dashdotted',    (0, (3, 1, 1, 1))),
            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        degrees = np.arange(0, 5, step=1, dtype=int)
        spatials = []
        labels = []
        xticks_ticks = [0.0]
        xticks_labels = ['0']
        for degree in degrees:
            spatials.append(SpatialScheme(self._equation, self._riemann,
                degree, self._n_element, self._x_left, self._x_right))
            order = degree + 1
            labels.append(f'{method}{order}')
            xticks_ticks.append(order * np.pi)
            xticks_labels.append(f'${order}\pi$')
        kh_max = (degrees[-1] + 1) * np.pi
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot([0, kh_max], [0, kh_max], 'r-', label='Exact')
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.plot([0, kh_max], [0, 0], 'r-', label='Exact')
        for i in range(len(labels)):
            reduced, modified = self.get_wavenumbers(spatials[i])
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
        # plt.show()
        plt.savefig(f'compare_{method}.pdf')

    def compare_methods(self, degree: int):
        """Compare spatial schemes using the same degree but different methods.
        """
        linestyles = [
            ('dotted',                (0, (1, 1))),
            ('densely dotted',        (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('dashed',                (0, (5, 5))),
            ('densely dashed',        (0, (5, 1))),
            ('dashdotted',            (0, (3, 5, 1, 5))),
            ('densely dashdotted',    (0, (3, 1, 1, 1))),
            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        methods = [
            spatial.LagrangeDG,
            spatial.LagrangeFR,
            spatial.DGwithLagrangeFR,
        ]
        order = degree + 1
        xticks_ticks = [0.0]
        xticks_labels = ['0']
        for p in range(1, order + 1):
            xticks_ticks.append(p * np.pi)
            xticks_labels.append(f'${p}\pi$')
        spatials = []
        labels = []
        for method in methods:
            spatials.append(method(self._equation, self._riemann,
                degree, self._n_element, self._x_left, self._x_right))
            labels.append(f'{method.name()}{order}')
        kh_max = order * np.pi
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot([0, kh_max], [0, kh_max], 'r-', label='Exact')
        plt.ylabel(r'$\Re(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.subplot(2,1,2)
        plt.ylabel(r'$\Im(\tilde{\kappa}h)$')
        plt.xlabel(r'$\kappa h$')
        plt.xticks(xticks_ticks, xticks_labels)
        plt.grid()
        plt.plot([0, kh_max], [0, 0], 'r-', label='Exact')
        for i in range(len(labels)):
            reduced, modified = self.get_wavenumbers(spatials[i])
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
        # plt.show()
        plt.savefig(f'compare_{degree}-degree_methods.pdf')

    def test_all_degrees(self):
        self.compare_degrees('DG')
        self.compare_degrees('FR')
        self.compare_degrees('DGFR')

    def test_all_methods(self):
        self.compare_methods(2)
        self.compare_methods(4)

if __name__ == '__main__':
    unittest.main()
