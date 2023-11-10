add_executable(test_spatial_dg dg.cpp)
target_include_directories(test_spatial_dg PRIVATE ${CGNS_INC} ${EIGEN_INC} ${MPI_INCLUDE_PATH})
target_link_libraries(test_spatial_dg ${CGNS_LIB} ${MPI_LIBRARIES})
set_target_properties(test_spatial_dg PROPERTIES OUTPUT_NAME dg)
