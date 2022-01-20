add_library(demo_burgers_rkdg rkdg.cpp)
target_include_directories(demo_burgers_rkdg PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_burgers_rkdg ${CGNS_LIB} metis ${MPI_LIBRARY})

set (cases
  standing
)
foreach (case ${cases})
  add_executable(demo_burgers_${case} ${case}.cpp)
  set_target_properties(demo_burgers_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_burgers_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_burgers_${case} demo_burgers_rkdg)
endforeach (case ${cases})
