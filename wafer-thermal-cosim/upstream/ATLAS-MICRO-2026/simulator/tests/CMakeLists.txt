add_executable(test_config_parser test_config_parser.cpp)
target_link_libraries(
    test_config_parser PRIVATE 
    atlasim-lib
    yaml-cpp
)


add_executable(test_dram test_dram.cpp)
target_link_libraries(
    test_dram PRIVATE 
    atlasim-lib
    yaml-cpp
    atlasim-dram
    atlasim-noc
)


add_executable(test_component_input test_component_input.cpp)
target_link_libraries(
    test_component_input PRIVATE
    atlasim-lib
    yaml-cpp
)


add_executable(test_core test_core.cpp)
target_link_libraries(
    test_core PRIVATE 
    atlasim-lib
    yaml-cpp
    atlasim-dram
    atlasim-noc
)
