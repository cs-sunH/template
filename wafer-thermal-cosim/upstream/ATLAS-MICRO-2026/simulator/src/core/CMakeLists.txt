add_library(atlasim-core OBJECT)
target_sources(
    atlasim-core PRIVATE
    component.cpp
    buffer.cpp
    matrix_unit.cpp
    vector_unit.cpp
    noc_interface.cpp
    core.cpp
    core_array.cpp
)
target_link_libraries(
    atlasim-core PRIVATE
    yaml-cpp
    atlasim-dram
)
