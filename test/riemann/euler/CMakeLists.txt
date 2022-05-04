set (cases
  types
  exact
  hllc
  ausm
  performance
)
foreach (case ${cases})
  add_executable(test_riemann_euler_${case} ${case}.cpp)
  set_target_properties(test_riemann_euler_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_riemann_euler_${case} COMMAND ${case})
endforeach (case ${cases})

add_executable(test_riemann_euler_eigen eigen.cpp)
target_include_directories(test_riemann_euler_eigen PRIVATE ${EIGEN_INC})
set_target_properties(test_riemann_euler_eigen PROPERTIES OUTPUT_NAME eigen)
add_test(NAME test_riemann_euler_eigen COMMAND eigen)
