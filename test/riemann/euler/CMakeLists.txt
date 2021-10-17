add_executable(test_riemann_euler_types types.cpp)
set_target_properties(test_riemann_euler_types PROPERTIES OUTPUT_NAME types)
add_test(NAME TestRiemannEulerTypes COMMAND types)

add_executable(test_riemann_euler_exact exact.cpp)
set_target_properties(test_riemann_euler_exact PROPERTIES OUTPUT_NAME exact)
add_test(NAME TestRiemannEulerExact COMMAND exact)

add_executable(test_riemann_euler_hllc hllc.cpp)
set_target_properties(test_riemann_euler_hllc PROPERTIES OUTPUT_NAME hllc)
add_test(NAME TestRiemannEulerHllc COMMAND hllc)

add_executable(test_riemann_euler_ausm ausm.cpp)
set_target_properties(test_riemann_euler_ausm PROPERTIES OUTPUT_NAME ausm)
add_test(NAME TestRiemannEulerAusm COMMAND ausm)

add_executable(test_riemann_euler_eigen eigen.cpp)
target_include_directories(test_riemann_euler_eigen PRIVATE ${EIGEN_INC})
set_target_properties(test_riemann_euler_eigen PROPERTIES OUTPUT_NAME eigen)
add_test(NAME TestRiemannEulerEigen COMMAND eigen)
