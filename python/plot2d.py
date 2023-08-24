"""Plot solutions or errors in the x-t plane.
"""
import sys
import numpy as np
import vtk
from matplotlib import pyplot as plt
from matplotlib import colors


class Viewer:

    def __init__(self, expect, actual) -> None:
        self._expect = self.load(expect, 'U')
        self._actual = self.load(actual, 'U')
        self._viscosity = self.load(actual, 'Viscosity')
        self._actual_path = actual

    def load(self, path, array_name):
        reader = vtk.vtkXMLUnstructuredGridReader()
        n_point = 201
        n_frame = 101
        udata = np.ndarray((n_frame, n_point))
        nu_data = np.ndarray((n_frame, n_point))
        for i_frame in range(n_frame):
            reader.SetFileName(f"{path}/Frame{i_frame}.vtu")
            reader.Update()
            pdata = reader.GetOutput().GetPointData()
            assert isinstance(pdata, vtk.vtkPointData)
            u = pdata.GetAbstractArray(array_name)
            assert isinstance(u, vtk.vtkFloatArray)
            for i_point in range(n_point):
                udata[i_frame][i_point] = u.GetTuple1(i_point)
        return udata

    def plot_error(self):
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$')
        zdata = np.abs(self._expect - self._actual)
        mappable = ax.imshow(zdata, origin='lower', cmap='coolwarm',
            norm=colors.LogNorm(vmin=1e-6, vmax=1e1, clip=True))
        fig.colorbar(mappable, location='top', label='Pointwise Errors')
        plt.tight_layout()
        plt.savefig(f'{self._actual_path}/error2d.svg')

    def plot_viscosity(self):
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$')
        zdata = self._viscosity
        mappable = ax.imshow(zdata, origin='lower', cmap='coolwarm')
        fig.colorbar(mappable, location='top', label='Viscosity')
        plt.tight_layout()
        plt.savefig(f'{self._actual_path}/viscosity.svg')


if __name__ == '__main__':
    viewer = Viewer(sys.argv[1], sys.argv[2])
    viewer.plot_error()
    viewer.plot_viscosity()
