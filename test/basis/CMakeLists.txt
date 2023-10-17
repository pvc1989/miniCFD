set (cases
  taylor
  lagrange
  linear
)
foreach(case ${cases})
  add_executable(test_basis_${case} ${case}.cpp)
  target_include_directories(test_basis_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_basis_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_basis_${case} COMMAND ${case})
endforeach(case ${cases})
