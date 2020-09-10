add_executable(test_projector_line line.cpp)
set_target_properties(test_projector_line PROPERTIES OUTPUT_NAME line)
add_test(NAME TestProjectorLine COMMAND line)
