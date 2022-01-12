set (cases
  linear
  double
)
foreach (case ${cases})
  add_executable(demo_single_${case} ${case}.cpp)
  set_target_properties(demo_single_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_single_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_single_${case})
endforeach (case ${cases})
