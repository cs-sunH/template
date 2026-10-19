add_subdirectory(dram)
add_subdirectory(core)
add_subdirectory(chip)
add_subdirectory(noc)

# build lib
add_library(
    atlasim-lib SHARED
)
target_link_libraries(
    atlasim-lib PRIVATE
    yaml-cpp
    atlasim-dram
    atlasim-core
    atlasim-noc
    atlasim-chip
)

# build executable
add_executable(
    atlasim-exe
    simulate.cpp
)
target_link_libraries(
    atlasim-exe PUBLIC
    atlasim-lib
)

# python bindings
pybind11_add_module(atlasim python_bindings.cpp)
target_link_libraries(atlasim PRIVATE atlasim-lib atlasim-noc)

# Ensure RPATH is set correctly for Linux so the module finds the shared lib
if(UNIX AND NOT APPLE)
    target_link_options(atlasim PRIVATE "-Wl,-rpath,$ORIGIN")
endif()
