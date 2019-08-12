# add_executable(mesh_test mesh_test.cpp)
# target_link_libraries(mesh_test gtest_main)
# add_test(NAME MeshTest COMMAND mesh_test)

add_executable(node_test node_test.cpp)
target_link_libraries(node_test gtest_main)
add_test(NAME NodeTest COMMAND node_test)

add_executable(edge_test edge_test.cpp)
target_link_libraries(edge_test gtest_main)
add_test(NAME EdgeTest COMMAND edge_test)

add_executable(face_test face_test.cpp)
target_link_libraries(face_test gtest_main)
add_test(NAME FaceTest COMMAND face_test)
