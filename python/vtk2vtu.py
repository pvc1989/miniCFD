#!/usr/bin/env /Applications/ParaView-5.10.0-RC1.app/Contents/bin/pvpython

#### import the simple module from the paraview
from sys import argv
from typing import AsyncIterator
from paraview.simple import *

if __name__ == "__main__":
  import sys, os
  if len(sys.argv) < 7:
    print('usage:\n  {0} <case_name> <n_parts> <first_step> <last_step> <n_steps_per_frame> <n_fields>'.format(sys.argv[0]))
    exit()
  case_name = sys.argv[1]
  n_parts = int(sys.argv[2])
  first = int(sys.argv[3])
  last = int(sys.argv[4])
  step = int(sys.argv[5])
  n_fields = int(sys.argv[6])

  steps = list(range(first, last + step, step))

  for i_step in steps:
    step_name = '{0}/{1}/Step{2}'.format(os.getcwd(), case_name, i_step)
    print('Write data in {0}.pvtu'.format(step_name))
    with open('{0}.pvtu'.format(step_name), mode='w') as f:
      f.write('<VTKFile type="PUnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
      f.write('  <PUnstructuredGrid GhostLevel="1">\n')
      f.write('    <PPointData Scalars="Field[1]">\n')
      for i_field in range(n_fields):
        f.write('      <PDataArray type="Float64" Name="Field[{0}]"/>\n'.format(i_field + 1))
      f.write('    </PPointData>\n')
      f.write('    <PPoints>\n')
      f.write('      <PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>\n')
      f.write('    </PPoints>\n')
      for i_part in range(n_parts):
        f.write('    <Piece Source="./Step{0}/{1}.vtu"/>\n'.format(i_step, i_part))
      f.write('  </PUnstructuredGrid>\n')
      f.write('</VTKFile>\n')
    for i_part in range(n_parts):
      filename = '{0}/{1}'.format(step_name, i_part)
      # create a new 'Legacy VTK Reader'
      vtk_obj = LegacyVTKReader(registrationName='{0}.vtk'.format(i_part), FileNames=[filename + '.vtk'])
      # save data
      point_data_arrays = []
      for i_field in range(n_fields):
        point_data_arrays.append('Field[{0}]'.format(i_field + 1))
      SaveData(filename + '.vtu', proxy=vtk_obj, PointDataArrays=point_data_arrays)
