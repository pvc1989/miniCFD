add_executable(test_integrator_gauss gauss.cpp)
set_target_properties(test_integrator_gauss PROPERTIES OUTPUT_NAME gauss)
add_test(NAME TestIntegratorGauss COMMAND gauss)

add_executable(test_integrator_tri tri.cpp)
target_include_directories(test_integrator_tri PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_tri PROPERTIES OUTPUT_NAME tri)
add_test(NAME TestIntegratorTri COMMAND tri)

add_executable(test_integrator_quad quad.cpp)
target_include_directories(test_integrator_quad PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_quad PROPERTIES OUTPUT_NAME quad)
add_test(NAME TestIntegratorQuad COMMAND quad)

add_executable(test_integrator_hexa hexa.cpp)
target_include_directories(test_integrator_hexa PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_hexa PROPERTIES OUTPUT_NAME hexa)
add_test(NAME TestIntegratorHexa COMMAND hexa)

add_executable(test_integrator_tetra tetra.cpp)
target_include_directories(test_integrator_tetra PRIVATE ${EIGEN_INC})
set_target_properties(test_integrator_tetra PROPERTIES OUTPUT_NAME tetra)
add_test(NAME TestIntegratorTetra COMMAND tetra)
