set (cases
  tetrahedron
  hexahedron
)
foreach (case ${cases})
  add_executable(test_lagrange_${case} ${case}.cpp)
  target_include_directories(test_lagrange_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_lagrange_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_lagrange_${case} COMMAND ${case})
endforeach (case ${cases})
