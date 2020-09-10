cmake_minimum_required(VERSION 3.0)
project(miniCFD)
set(CMAKE_CXX_STANDARD 17)

# GoogleTest related settings
option(${PROJECT_NAME}_SUBMODULE_GOOGLETEST "Add GoogleTest as a git submodule." "ON")
if (${PROJECT_NAME}_SUBMODULE_GOOGLETEST)
  add_subdirectory(ThirdParty/GoogleTest)
else (${PROJECT_NAME}_SUBMODULE_GOOGLETEST)
  # Download and unpack `googletest` at configure time
  configure_file(CMakeLists.txt.in googletest/download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest/download )
  if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif ()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest/download )
  if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif ()
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Add `googletest` directly to our build.
  # This defines the `gtest` and `gtest_main`s targets.
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest/src
                   ${CMAKE_CURRENT_BINARY_DIR}/googletest/build
                   EXCLUDE_FROM_ALL)
  # The `gtest`/`gtest_main` targets carry header search path
  # dependencies automatically when using CMake 2.8.11 or later.
  # Otherwise we have to add them here ourselves.
  if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
  endif ()
  # Now simply link against gtest or gtest_main as needed.
endif (${PROJECT_NAME}_SUBMODULE_GOOGLETEST)
# End of GoogleTest related settings

option(${PROJECT_NAME}_ENABLE_MPI "Enable MPI-based parallel computating." "ON")
if (${PROJECT_NAME}_ENABLE_MPI)
  find_path(MPI_INC NAMES "mpi.h")
  find_library(MPI_LIB NAMES mpi)
endif (${PROJECT_NAME}_ENABLE_MPI)

option(${PROJECT_NAME}_SUBMODULE_CGNS "Add CGNS as a git submodule." "ON")
if (${PROJECT_NAME}_SUBMODULE_CGNS)
  set(CGNS_ENABLE_HDF5 ON CACHE BOOL "Build CGNS on top of HDF5." FORCE)
  if (${PROJECT_NAME}_ENABLE_MPI)
    set(CGNS_ENABLE_PARALLEL ON CACHE BOOL "Enable parallel CGNS interface." FORCE)
    set(HDF5_NEED_MPI ON CACHE BOOL "Link HDF5 to MPI." FORCE)
  endif (${PROJECT_NAME}_ENABLE_MPI)
  add_subdirectory(ThirdParty/CGNS)
  set(CGNS_INC "${cgns_SOURCE_DIR}/src;${cgns_BINARY_DIR}/src")
else (${PROJECT_NAME}_SUBMODULE_CGNS) # Find installed VTK.
  if (${PROJECT_NAME}_ENABLE_MPI)
    find_path(CGNS_INC "pcgnslib.h" HINTS ENV CGNS_ROOT PATH_SUFFIXES include)
  else (${PROJECT_NAME}_ENABLE_MPI)
    find_path(CGNS_INC "cgnslib.h" HINTS ENV CGNS_ROOT PATH_SUFFIXES include)
  endif (${PROJECT_NAME}_ENABLE_MPI)
  find_library(CGNS_LIB cgns HINTS ENV CGNS_ROOT PATH_SUFFIXES lib)
endif (${PROJECT_NAME}_SUBMODULE_CGNS)

option(${PROJECT_NAME}_ENABLE_VTK "Enable VTK-based IO." "OFF")
if (${PROJECT_NAME}_ENABLE_VTK)
  find_package(VTK COMPONENTS
    vtkCommonCore
    vtkCommonDataModel
    vtkIOLegacy
    vtkIOXML
    vtkIOGeometry
    vtkIOImport
    vtkIOExport
    vtksys
  QUIET)
  if (VTK_VERSION VERSION_LESS "8.90.0")
    include(${VTK_USE_FILE})
  endif ()
endif (${PROJECT_NAME}_ENABLE_VTK)

# Additional headers that depends on ${PROJECT_SOURCE_DIR}
include_directories("${PROJECT_SOURCE_DIR}/include")
# Additional headers that depends on ${PROJECT_BINARY_DIR}
configure_file(
  "${PROJECT_SOURCE_DIR}/data/path.hpp.in"
  "${PROJECT_BINARY_DIR}/include/mini/data/path.hpp"
)
include_directories("${PROJECT_BINARY_DIR}/include")

option(${PROJECT_NAME}_BUILD_TESTS "Build tests for this project." "ON")
if (${PROJECT_NAME}_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif ()

option(${PROJECT_NAME}_BUILD_DEMOS "Build demos for this project." "ON")
if (${PROJECT_NAME}_BUILD_DEMOS)
  add_subdirectory(demo)
endif (${PROJECT_NAME}_BUILD_DEMOS)
