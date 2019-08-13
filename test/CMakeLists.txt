add_executable(mesh_test mesh_test.cpp)
target_link_libraries(mesh_test gtest_main)
add_test(NAME MeshTest COMMAND mesh_test)

add_executable(geometry geometry.cpp)
target_link_libraries(geometry gtest_main)
add_test(NAME Geometry COMMAND geometry)

add_executable(element element.cpp)
target_link_libraries(element gtest_main)
add_test(NAME Element COMMAND element)
