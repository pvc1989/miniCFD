add_executable(test_gauss_legendre legendre.cpp)
set_target_properties(test_gauss_legendre PROPERTIES OUTPUT_NAME legendre)
add_test(NAME test_gauss_legendre COMMAND legendre)

set (cases
  triangle
  quadrangle
  tetrahedron
  hexahedron
  wedge
)
foreach (case ${cases})
  add_executable(test_gauss_${case} ${case}.cpp)
  target_include_directories(test_gauss_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_gauss_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_gauss_${case} COMMAND ${case})
endforeach (case ${cases})
