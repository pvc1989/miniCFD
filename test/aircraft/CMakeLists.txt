add_executable(test_aircraft_rotor rotor.cpp)
set_target_properties(test_aircraft_rotor PROPERTIES OUTPUT_NAME rotor)
add_test(NAME TestAircraftRotor COMMAND rotor)
