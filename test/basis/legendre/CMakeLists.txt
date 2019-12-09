add_executable(legendre legendre.cpp)
target_link_libraries(legendre gtest_main)
add_test(NAME Legendre COMMAND legendre)
