add_executable(test_aircraft_rotor rotor.cpp)
set_target_properties(test_aircraft_rotor PROPERTIES OUTPUT_NAME rotor)
add_test(NAME test_aircraft_rotor COMMAND rotor)

add_executable(test_aircraft_section section.cpp)
set_target_properties(test_aircraft_section PROPERTIES OUTPUT_NAME section)
add_test(NAME test_aircraft_section COMMAND section)
