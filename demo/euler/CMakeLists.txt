add_executable(tube tube.cpp)
target_link_libraries(tube ${VTK_LIBRARIES})

add_executable(box box.cpp)
target_link_libraries(box ${VTK_LIBRARIES})
