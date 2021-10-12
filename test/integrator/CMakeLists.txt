add_executable(test_integrator_gauss gauss.cpp)
set_target_properties(test_integrator_gauss PROPERTIES OUTPUT_NAME gauss)
add_test(NAME TestIntegratorGauss COMMAND gauss)

add_executable(test_integrator_line line.cpp)
target_include_directories(test_integrator_line PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_line PROPERTIES OUTPUT_NAME line)
add_test(NAME TestIntegratorLine COMMAND line)

add_executable(test_integrator_quad quad.cpp)
target_include_directories(test_integrator_quad PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_quad PROPERTIES OUTPUT_NAME quad)
add_test(NAME TestIntegratorQuad COMMAND quad)

add_executable(test_integrator_hexa hexa.cpp)
target_include_directories(test_integrator_hexa PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_hexa PROPERTIES OUTPUT_NAME hexa)
add_test(NAME TestIntegratorHexa COMMAND hexa)

add_executable(test_integrator_basis basis.cpp)
target_include_directories(test_integrator_basis PRIVATE ${EIGEN_INC} ${CGNS_INC} ${METIS_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(test_integrator_basis cgns_shared metis)
set_target_properties(test_integrator_basis PROPERTIES OUTPUT_NAME basis)
add_test(NAME TestIntegratorBasis COMMAND basis)
