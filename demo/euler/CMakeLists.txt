if (${PROJECT_NAME}_ENABLE_VTK)

  add_executable(demo_euler_tube tube.cpp)
  set_target_properties(demo_euler_tube PROPERTIES OUTPUT_NAME tube)
  target_link_libraries(demo_euler_tube ${VTK_LIBRARIES})

  add_executable(demo_euler_box box.cpp)
  set_target_properties(demo_euler_box PROPERTIES OUTPUT_NAME box)
  target_link_libraries(demo_euler_box ${VTK_LIBRARIES})

endif (${PROJECT_NAME}_ENABLE_VTK)

add_library(demo_euler_main main.cpp)
target_include_directories(demo_euler_main PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_euler_main ${CGNS_LIB} metis ${MPI_LIBRARY})

set (cases
  # shock tube problems:
  sod
  lax
  almost_vacuumed
  # two-dimensional problems:
  double_mach
  forward_step
  # three-dimensional problems:
  yf17
)
foreach (case ${cases})
  add_executable(demo_euler_${case} ${case}.cpp)
  set_target_properties(demo_euler_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_euler_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_euler_${case} demo_euler_main)
endforeach (case ${cases})
