set (cases
  single
  double
  multiple
)
foreach (case ${cases})
  add_executable(demo_linear_${case} ${case}.cpp)
  set_target_properties(demo_linear_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_linear_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_linear_${case} ${CGNS_LIB} metis ${MPI_LIBRARY})
endforeach (case ${cases})
