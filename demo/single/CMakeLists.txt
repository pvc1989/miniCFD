add_library(demo_single_rkdg rkdg.cpp)
target_include_directories(demo_single_rkdg PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_single_rkdg cgns_shared metis ${MPI_mpi_LIBRARY})

set (cases
  linear
  burgers
)
foreach (case ${cases})
  add_executable(demo_single_${case} ${case}.cpp)
  set_target_properties(demo_single_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_single_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_single_${case} demo_single_rkdg)
endforeach (case ${cases})
