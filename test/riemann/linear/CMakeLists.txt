set (cases
  single
  double
  multiple
)
foreach (case ${cases})
add_executable(test_riemann_linear_${case} ${case}.cpp)
set_target_properties(test_riemann_linear_${case} PROPERTIES OUTPUT_NAME ${case})
add_test(NAME test_riemann_linear_${case} COMMAND ${case})
endforeach (case ${cases})
