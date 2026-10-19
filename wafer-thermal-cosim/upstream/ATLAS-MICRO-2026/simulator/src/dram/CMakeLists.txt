add_subdirectory(ramulator2)

add_library(
    atlasim-dram OBJECT
    dram_wrapper.cpp
)
target_link_libraries(
    atlasim-dram PUBLIC
    ramulator
)
