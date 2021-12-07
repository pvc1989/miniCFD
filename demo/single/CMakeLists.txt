add_executable(demo_single_linear linear.cpp)
set_target_properties(demo_single_linear PROPERTIES OUTPUT_NAME linear)
target_include_directories(demo_single_linear PRIVATE ${CGNS_INC} ${METIS_INC} ${EIGEN_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(demo_single_linear cgns_shared metis ${MPI_mpi_LIBRARY})
