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
