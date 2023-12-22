set (cases
  single
  double
  multiple
  burgers
)
foreach (case ${cases})
add_executable(test_riemann_simple_${case} ${case}.cpp)
set_target_properties(test_riemann_simple_${case} PROPERTIES OUTPUT_NAME ${case})
add_test(NAME test_riemann_simple_${case} COMMAND ${case})
endforeach (case ${cases})
