set (cases
  dg
  fr
)
foreach (case ${cases})
  add_executable(test_spatial_${case} ${case}.cpp)
  target_include_directories(test_spatial_${case} PRIVATE ${CGNS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(test_spatial_${case} ${CGNS_LIB} ${MPI_LIBRARIES})
  set_target_properties(test_spatial_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_spatial_${case} COMMAND mpirun -n 4 ${case})
endforeach (case ${cases})
