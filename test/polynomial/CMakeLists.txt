add_executable(test_polynomial_legendre legendre.cpp)
set_target_properties(test_polynomial_legendre PROPERTIES OUTPUT_NAME legendre)
add_test(NAME TestPolynomialLegendre COMMAND legendre)

add_executable(test_polynomial_quad quad.cpp)
target_include_directories(test_polynomial_quad PRIVATE ${EIGEN_INC})
set_target_properties(test_polynomial_quad PROPERTIES OUTPUT_NAME quad)
add_test(NAME TestIntegratorQuad COMMAND quad)

add_executable(test_polynomial_hexa hexa.cpp)
target_include_directories(test_polynomial_hexa PRIVATE ${EIGEN_INC})
set_target_properties(test_polynomial_hexa PROPERTIES OUTPUT_NAME hexa)
add_test(NAME TestIntegratorHexa COMMAND hexa)

add_executable(test_polynomial_basis basis.cpp)
target_include_directories(test_polynomial_basis PRIVATE ${EIGEN_INC})
set_target_properties(test_polynomial_basis PROPERTIES OUTPUT_NAME basis)
add_test(NAME TestIntegratorBasis COMMAND basis)

add_executable(test_polynomial_projection projection.cpp)
target_include_directories(test_polynomial_projection PRIVATE ${EIGEN_INC} ${CGNS_INC} ${METIS_INC} ${MPI_CXX_HEADER_DIR})
target_link_libraries(test_polynomial_projection cgns_shared metis)
set_target_properties(test_polynomial_projection PROPERTIES OUTPUT_NAME projection)
add_test(NAME TestIntegratorProjection COMMAND projection)
