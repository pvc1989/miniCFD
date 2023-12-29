set (cases
  index
)
foreach (case ${cases})
  add_executable(test_constant_${case} ${case}.cpp)
  target_include_directories(test_constant_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_constant_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_constant_${case} COMMAND ${case})
endforeach (case ${cases})
