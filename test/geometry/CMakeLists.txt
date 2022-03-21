add_executable(test_geometry_point point.cpp)
set_target_properties(test_geometry_point PROPERTIES OUTPUT_NAME point)
add_test(NAME TestGeometryPoint COMMAND point)

add_executable(test_geometry_vector vector.cpp)
set_target_properties(test_geometry_vector PROPERTIES OUTPUT_NAME vector)
add_test(NAME TestGeometryVector COMMAND vector)

add_executable(test_geometry_line line.cpp)
set_target_properties(test_geometry_line PROPERTIES OUTPUT_NAME line)
add_test(NAME TestGeometryLine COMMAND Line)

add_executable(test_geometry_surface surface.cpp)
set_target_properties(test_geometry_surface PROPERTIES OUTPUT_NAME surface)
add_test(NAME TestGeometrySurface COMMAND surface)

add_executable(test_geometry_triangle triangle.cpp)
set_target_properties(test_geometry_triangle PROPERTIES OUTPUT_NAME triangle)
add_test(NAME TestGeometryTriangle COMMAND triangle)

add_executable(test_geometry_rectangle rectangle.cpp)
set_target_properties(test_geometry_rectangle PROPERTIES OUTPUT_NAME rectangle)
add_test(NAME TestGeometryRectangle COMMAND rectangle)

add_executable(test_geometry_frame frame.cpp)
set_target_properties(test_geometry_frame PROPERTIES OUTPUT_NAME frame)
add_test(NAME TestGeometryFrame COMMAND frame)
