add_library(atlasim-chip OBJECT)
target_sources(
    atlasim-chip PRIVATE
    chip.cpp
)
target_link_libraries(
    atlasim-chip PRIVATE
    yaml-cpp
    atlasim-core
    atlasim-noc
)
