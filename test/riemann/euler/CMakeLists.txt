add_executable(types types.cpp)
target_link_libraries(types gtest_main)

add_executable(exact exact.cpp)
target_link_libraries(exact gtest_main)
