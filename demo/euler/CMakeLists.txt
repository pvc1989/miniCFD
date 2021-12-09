if (${PROJECT_NAME}_ENABLE_VTK)

  add_executable(demo_euler_tube tube.cpp)
  set_target_properties(demo_euler_tube PROPERTIES OUTPUT_NAME tube)
  target_link_libraries(demo_euler_tube ${VTK_LIBRARIES})

  add_executable(demo_euler_box box.cpp)
  set_target_properties(demo_euler_box PROPERTIES OUTPUT_NAME box)
  target_link_libraries(demo_euler_box ${VTK_LIBRARIES})

  endif (${PROJECT_NAME}_ENABLE_VTK)

add_executable(demo_euler_double_mach double_mach.cpp)
set_target_properties(demo_euler_double_mach PROPERTIES OUTPUT_NAME double_mach)
target_include_directories(demo_euler_double_mach PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_euler_double_mach cgns_shared metis ${MPI_mpi_LIBRARY})

add_executable(demo_euler_yf17 yf17.cpp)
set_target_properties(demo_euler_yf17 PROPERTIES OUTPUT_NAME yf17)
target_include_directories(demo_euler_yf17 PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_euler_yf17 cgns_shared metis ${MPI_mpi_LIBRARY})
