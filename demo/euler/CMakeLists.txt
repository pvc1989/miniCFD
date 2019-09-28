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

include_directories("${PROJECT_BINARY_DIR}")

add_executable(tube tube.cpp)
target_link_libraries(tube ${VTK_LIBRARIES})

add_executable(box box.cpp)
target_link_libraries(box ${VTK_LIBRARIES})
