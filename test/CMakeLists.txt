add_executable(algebra algebra.cpp)
target_link_libraries(algebra gtest_main)
add_test(NAME Algebra COMMAND algebra)

add_executable(geometry geometry.cpp)
target_link_libraries(geometry gtest_main)
add_test(NAME Geometry COMMAND geometry)

add_executable(element element.cpp)
target_link_libraries(element gtest_main)
add_test(NAME Element COMMAND element)

add_executable(mesh mesh.cpp)
target_link_libraries(mesh gtest_main)
add_test(NAME Mesh COMMAND mesh)

add_executable(vtk vtk.cpp)
target_link_libraries(vtk ${VTK_LIBRARIES} gtest_main)
add_test(NAME VTK COMMAND vtk)

add_subdirectory(riemann)
