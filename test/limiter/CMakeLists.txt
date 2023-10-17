
add_executable(test_limiter_weno weno.cpp)
target_include_directories(test_limiter_weno PRIVATE ${EIGEN_INC} ${CGNS_INC} ${METIS_INC} ${MPI_INCLUDE_PATH})
target_link_libraries(test_limiter_weno ${CGNS_LIB} ${MPI_LIBRARIES} metis)
set_target_properties(test_limiter_weno PROPERTIES OUTPUT_NAME weno)
add_test(NAME test_limiter_weno COMMAND weno)
