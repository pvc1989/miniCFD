add_executable(test_algebra_column column.cpp)
set_target_properties(test_algebra_column PROPERTIES OUTPUT_NAME column)
add_test(NAME TestAlgebraColumn COMMAND column)

add_executable(test_algebra_matrix matrix.cpp)
set_target_properties(test_algebra_matrix PROPERTIES OUTPUT_NAME matrix)
add_test(NAME TestAlgebraMatrix COMMAND matrix)
