add_executable(test_mesh_dim1 dim1.cpp)
set_target_properties(test_mesh_dim1 PROPERTIES OUTPUT_NAME dim1)
add_test(NAME TestMeshDim1 COMMAND dim1)

add_executable(test_mesh_dim2 dim2.cpp)
set_target_properties(test_mesh_dim2 PROPERTIES OUTPUT_NAME dim2)
add_test(NAME TestMeshDim2 COMMAND dim2)

if (${PROJECT_NAME}_ENABLE_VTK)
  link_libraries(${VTK_LIBRARIES})

  add_executable(test_mesh_vtk1d vtk1d.cpp)
  set_target_properties(test_mesh_vtk1d PROPERTIES OUTPUT_NAME vtk1d)
  add_test(NAME TestMeshVtk1d COMMAND vtk1d)

  add_executable(test_mesh_vtk2d vtk2d.cpp)
  set_target_properties(test_mesh_vtk2d PROPERTIES OUTPUT_NAME vtk2d)
  add_test(NAME TestMeshVtk2d COMMAND vtk2d)
endif (${PROJECT_NAME}_ENABLE_VTK)

include_directories(${CGNS_INC} ${MPI_INC})
link_libraries(cgns_shared)

add_executable(test_mesh_cgns cgns.cpp)
set_target_properties(test_mesh_cgns PROPERTIES OUTPUT_NAME cgns)
add_test(NAME TestMeshCgns COMMAND cgns)
