set (cases
  standing
)
foreach (case ${cases})
  add_executable(demo_burgers_${case} ${case}.cpp)
  set_target_properties(demo_burgers_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_burgers_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
  target_link_libraries(demo_burgers_${case} ${CGNS_LIB} metis ${MPI_LIBRARIES})
endforeach (case ${cases})
