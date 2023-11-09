set (cases
  rk
)
foreach (case ${cases})
  add_executable(test_temporal_${case} ${case}.cpp)
  target_include_directories(test_temporal_${case} PRIVATE ${EIGEN_INC})
  set_target_properties(test_temporal_${case} PROPERTIES OUTPUT_NAME ${case})
  add_test(NAME test_temporal_${case} COMMAND ${case})
endforeach (case ${cases})
