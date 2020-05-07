add_executable(test_mesh_mesh1d mesh1d.cpp)
set_target_properties(test_mesh_mesh1d PROPERTIES OUTPUT_NAME mesh1d)
target_link_libraries(test_mesh_mesh1d gtest_main)

add_executable(test_mesh_dim2 dim2.cpp)
set_target_properties(test_mesh_dim2 PROPERTIES OUTPUT_NAME dim2)
target_link_libraries(test_mesh_dim2 gtest_main)
add_test(NAME TestMeshDim2 COMMAND dim2)

add_executable(test_mesh_vtk1d vtk1d.cpp)
set_target_properties(test_mesh_vtk1d PROPERTIES OUTPUT_NAME vtk1d)
target_link_libraries(test_mesh_vtk1d gtest_main)
add_test(NAME TestMeshVtk1d COMMAND vtk1d)

add_executable(test_mesh_vtk2d vtk2d.cpp)
set_target_properties(test_mesh_vtk2d PROPERTIES OUTPUT_NAME vtk2d)
target_link_libraries(test_mesh_vtk2d gtest_main)
add_test(NAME TestMeshVtk2d COMMAND vtk2d)
