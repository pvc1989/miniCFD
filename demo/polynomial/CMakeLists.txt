set (cases
  single
)
foreach (case ${cases})
  add_executable(demo_polynomial_${case} ${case}.cpp)
  set_target_properties(demo_polynomial_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_polynomial_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_polynomial_${case} ${CGNS_LIB} metis ${MPI_LIBRARY})
endforeach (case ${cases})
