add_subdirectory(algebra)
add_subdirectory(geometry)
add_subdirectory(element)


add_executable(mesh mesh.cpp)
target_link_libraries(mesh gtest_main)
add_test(NAME Mesh COMMAND mesh)

add_executable(mesh1d mesh1d.cpp)
target_link_libraries(mesh1d gtest_main)
add_test(NAME Mesh1D COMMAND mesh1d)

add_executable(vtk vtk.cpp)
target_link_libraries(vtk ${VTK_LIBRARIES} gtest_main)
add_test(NAME VTK COMMAND vtk)

add_executable(vtk1d vtk1d.cpp)
target_link_libraries(vtk1d ${VTK_LIBRARIES} gtest_main)
add_test(NAME VTK1D COMMAND vtk1d)

add_subdirectory(integrator)
add_subdirectory(polynomial)
add_subdirectory(projector)
add_subdirectory(riemann)
