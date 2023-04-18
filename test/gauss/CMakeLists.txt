add_executable(test_integrator_gauss gauss.cpp)
set_target_properties(test_integrator_gauss PROPERTIES OUTPUT_NAME gauss)
add_test(NAME test_integrator_gauss COMMAND gauss)

set (cases
  triangle
  quadrangle
  tetrahedron
  hexahedron
)
foreach (case ${cases})
  add_executable(test_integrator_${case} ${case}.cpp)
  target_include_directories(test_integrator_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_integrator_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_integrator_${case} COMMAND ${case})
endforeach (case ${cases})
