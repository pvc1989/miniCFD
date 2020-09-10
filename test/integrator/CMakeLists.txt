add_executable(test_integrator_gauss gauss.cpp)
set_target_properties(test_integrator_gauss PROPERTIES OUTPUT_NAME gauss)
add_test(NAME TestIntegratorGauss COMMAND gauss)

