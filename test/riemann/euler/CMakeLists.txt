add_executable(types types.cpp)
target_link_libraries(types gtest_main)

add_executable(exact exact.cpp)
target_link_libraries(exact gtest_main)

add_executable(hllc hllc.cpp)
target_link_libraries(hllc gtest_main)

add_executable(ausm ausm.cpp)
target_link_libraries(ausm gtest_main)
