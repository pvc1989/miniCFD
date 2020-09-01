add_executable(demo_euler_tube tube.cpp)
set_target_properties(demo_euler_tube PROPERTIES OUTPUT_NAME tube)
target_link_libraries(demo_euler_tube ${VTK_LIBRARIES})

add_executable(demo_euler_box box.cpp)
set_target_properties(demo_euler_box PROPERTIES OUTPUT_NAME box)
target_link_libraries(demo_euler_box ${VTK_LIBRARIES})
