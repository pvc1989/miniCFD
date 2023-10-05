import sys
import numpy as np
from scipy.io import loadmat
from vtk import *
import equation


if __name__ == '__main__':
    filename = sys.argv[1]
    mat_contents = loadmat(f'{filename}.mat')
    print(mat_contents.keys())
    rho = mat_contents['rhoR'][0]
    u = mat_contents['veloR'][0]
    p = mat_contents['pR'][0]
    n_point = len(rho)
    x_left = 0
    x_right = 10
    delta_x = (x_right - x_left) / n_point
    output_points = np.arange(x_left, x_right, delta_x) + delta_x
    grid = vtkUnstructuredGrid()
    vtk_points = vtkPoints()
    euler = equation.Euler(1.4)
    n_component = euler.n_component()
    component_names = euler.component_names()
    solutions = []
    for i in range(n_component):
        solutions.append(vtkFloatArray())
        vtkFloatArray.SetName(solutions[i], component_names[i])
        vtkFloatArray.SetNumberOfComponents(solutions[i], 1)
    i_point = 0
    for x in output_points:
        vtk_points.InsertNextPoint((x, 0, 0))
        value = euler.primitive_to_conservative(rho[i_point], u[i_point], p[i_point])
        for i in range(n_component):
            vtkFloatArray.InsertNextValue(solutions[i], value[i])
        i_point += 1
    assert i_point == n_point
    grid.SetPoints(vtk_points)
    for i_component in range(n_component):
        grid.GetPointData().SetActiveScalars(component_names[i_component])
        grid.GetPointData().SetScalars(solutions[i_component])
    writer = vtkXMLDataSetWriter()
    writer.SetInputData(grid)
    writer.SetFileName(f'{filename}.vtu')
    writer.SetDataModeToBinary()
    writer.Write()
