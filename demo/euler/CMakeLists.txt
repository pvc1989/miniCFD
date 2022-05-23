set (libs
  sourceless
  rotor_source
)
foreach(lib ${libs})
  add_library(demo_euler_${lib} ${lib}.cpp)
  set_target_properties(demo_euler_${lib} PROPERTIES OUTPUT_NAME ${lib})
  target_include_directories(demo_euler_${lib} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_euler_${lib} ${CGNS_LIB} metis ${MPI_LIBRARY})
endforeach(lib ${libs})

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
  mach_disk
  cylinder
)
foreach (case ${cases})
  add_executable(demo_euler_${case} ${case}.cpp)
  set_target_properties(demo_euler_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_euler_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_euler_${case} demo_euler_sourceless)
endforeach (case ${cases})

set (cases
  wing
  rotor
)
foreach (case ${cases})
  add_executable(demo_euler_${case} ${case}.cpp)
  set_target_properties(demo_euler_${case} PROPERTIES OUTPUT_NAME ${case})
  target_include_directories(demo_euler_${case} PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
  target_link_libraries(demo_euler_${case} demo_euler_rotor_source)
endforeach (case ${cases})
