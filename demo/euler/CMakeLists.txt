add_library(demo_euler_sourceless sourceless.cpp)
set_target_properties(demo_euler_sourceless PROPERTIES OUTPUT_NAME sourceless)
target_include_directories(demo_euler_sourceless PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_euler_sourceless ${CGNS_LIB} metis ${MPI_LIBRARY})

set (cases
  # shock tube problems:
  sod
  lax
  vacuum
  # two-dimensional problems:
  double_mach
  forward_step
  # three-dimensional problems:
  yf17
  cylinder
)
foreach (case ${cases})
  add_executable(demo_euler_${case} ${case}.cpp)
  set_target_properties(demo_euler_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_euler_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_euler_${case} demo_euler_sourceless)
endforeach (case ${cases})

add_executable(demo_euler_rotor rotor.cpp)
set_target_properties(demo_euler_rotor PROPERTIES OUTPUT_NAME rotor)
target_include_directories(demo_euler_rotor PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_euler_rotor ${CGNS_LIB} metis ${MPI_LIBRARY})
