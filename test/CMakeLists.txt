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

find_package(VTK REQUIRED)
if (vtkIOExport_LOADED AND vtkIOImport_LOADED)
vtk_module_config(VTK
  vtkCommonCore
  vtkIOLegacy
  vtkIOXML
  vtkIOGeometry
  vtkIOImport
  vtkIOExport
  vtksys
)
else()
vtk_module_config(VTK
  vtkCommonCore
  vtkIOLegacy
  vtkIOXML
  vtkIOGeometry
  vtkIOImport
  vtksys
)
endif()
include(${VTK_USE_FILE})

configure_file (
  "${PROJECT_SOURCE_DIR}/test/data.hpp.in"
  "${PROJECT_BINARY_DIR}/test/data.hpp"
)
include_directories("${PROJECT_BINARY_DIR}/test")

add_executable(vtk vtk.cpp)
target_link_libraries(vtk ${VTK_LIBRARIES} gtest_main)
add_test(NAME VTK COMMAND vtk)

add_subdirectory(riemann)

add_executable(model model.cpp)
target_link_libraries(model ${VTK_LIBRARIES} gtest_main)
add_test(NAME Model COMMAND model)
