add_executable(test_integrator_gauss gauss.cpp)
set_target_properties(test_integrator_gauss PROPERTIES OUTPUT_NAME gauss)
add_test(NAME TestIntegratorGauss COMMAND gauss)

add_executable(test_integrator_triangle triangle.cpp)
target_include_directories(test_integrator_triangle PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_triangle PROPERTIES OUTPUT_NAME triangle)
add_test(NAME TestIntegratorTriangle COMMAND triangle)

add_executable(test_integrator_quadrangle quadrangle.cpp)
target_include_directories(test_integrator_quadrangle PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_quadrangle PROPERTIES OUTPUT_NAME quadrangle)
add_test(NAME TestIntegratorQuadrangle COMMAND quadrangle)

add_executable(test_integrator_hexa hexa.cpp)
target_include_directories(test_integrator_hexa PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_hexa PROPERTIES OUTPUT_NAME hexa)
add_test(NAME TestIntegratorHexa COMMAND hexa)

add_executable(test_integrator_tetra tetra.cpp)
target_include_directories(test_integrator_tetra PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_tetra PROPERTIES OUTPUT_NAME tetra)
add_test(NAME TestIntegratorTetra COMMAND tetra)
