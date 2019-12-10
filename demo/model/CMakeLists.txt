add_executable(model model.cpp)
target_link_libraries(model ${VTK_LIBRARIES})

add_executable(model1d model1d.cpp)
target_link_libraries(model1d ${VTK_LIBRARIES})

