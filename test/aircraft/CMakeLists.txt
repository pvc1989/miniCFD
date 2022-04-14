add_executable(test_wing_rotary rotary.cpp)
set_target_properties(test_wing_rotary PROPERTIES OUTPUT_NAME rotary)
add_test(NAME TestWingRotary COMMAND rotary)
