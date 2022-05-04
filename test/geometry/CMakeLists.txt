add_executable(test_geometry_frame frame.cpp)
set_target_properties(test_geometry_frame PROPERTIES OUTPUT_NAME frame)
add_test(NAME test_geometry_frame COMMAND frame)
