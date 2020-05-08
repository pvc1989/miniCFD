add_executable(test_mesh_dim1 dim1.cpp)
set_target_properties(test_mesh_dim1 PROPERTIES OUTPUT_NAME dim1)
target_link_libraries(test_mesh_dim1 gtest_main)
add_test(NAME TestMeshDim1 COMMAND dim1)

add_executable(test_mesh_dim2 dim2.cpp)
set_target_properties(test_mesh_dim2 PROPERTIES OUTPUT_NAME dim2)
target_link_libraries(test_mesh_dim2 gtest_main)
add_test(NAME TestMeshDim2 COMMAND dim2)

add_executable(test_mesh_vtk1d vtk1d.cpp)
set_target_properties(test_mesh_vtk1d PROPERTIES OUTPUT_NAME vtk1d)
target_link_libraries(test_mesh_vtk1d ${VTK_LIBRARIES} gtest_main)
add_test(NAME TestMeshVtk1d COMMAND vtk1d)

add_executable(test_mesh_vtk2d vtk2d.cpp)
set_target_properties(test_mesh_vtk2d PROPERTIES OUTPUT_NAME vtk2d)
target_link_libraries(test_mesh_vtk2d ${VTK_LIBRARIES} gtest_main)
add_test(NAME TestMeshVtk2d COMMAND vtk2d)
