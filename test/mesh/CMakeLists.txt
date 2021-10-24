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

add_executable(test_mesh_cgns cgns.cpp)
target_include_directories(test_mesh_cgns PRIVATE ${CGNS_INC})
target_link_libraries(test_mesh_cgns cgns_shared)
set_target_properties(test_mesh_cgns PROPERTIES OUTPUT_NAME cgns)
add_test(NAME TestMeshCgns COMMAND cgns)

add_executable(test_mesh_metis metis.cpp)
target_include_directories(test_mesh_metis PRIVATE ${METIS_INC})
target_link_libraries(test_mesh_metis metis)
set_target_properties(test_mesh_metis PROPERTIES OUTPUT_NAME metis)
add_test(NAME TestMeshMetis COMMAND metis)

add_executable(test_mesh_mapper mapper.cpp)
target_include_directories(test_mesh_mapper PRIVATE ${CGNS_INC} ${METIS_INC})
target_link_libraries(test_mesh_mapper cgns_shared metis)
set_target_properties(test_mesh_mapper PROPERTIES OUTPUT_NAME mapper)
add_test(NAME TestMeshMapper COMMAND mapper)

add_executable(test_mesh_shuffler shuffler.cpp)
target_include_directories(test_mesh_shuffler PRIVATE ${CGNS_INC} ${METIS_INC})
target_link_libraries(test_mesh_shuffler cgns_shared metis)
set_target_properties(test_mesh_shuffler PROPERTIES OUTPUT_NAME shuffler)
add_test(NAME TestMeshShuffler COMMAND shuffler)

add_executable(test_mesh_part part.cpp)
target_include_directories(test_mesh_part PRIVATE ${CGNS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(test_mesh_part cgns_shared ${MPI_mpi_LIBRARY})
set_target_properties(test_mesh_part PROPERTIES OUTPUT_NAME part)
