add_subdirectory(booksim2/src)

add_library(
    atlasim-noc OBJECT
    noc_wrapper.cpp
)
target_include_directories(atlasim-noc PUBLIC booksim2/src)
target_include_directories(atlasim-noc PUBLIC booksim2/src/allocators)
target_include_directories(atlasim-noc PUBLIC booksim2/src/arbiters)
target_include_directories(atlasim-noc PUBLIC booksim2/src/include)
target_include_directories(atlasim-noc PUBLIC booksim2/src/networks)
target_include_directories(atlasim-noc PUBLIC booksim2/src/power)
target_include_directories(atlasim-noc PUBLIC booksim2/src/routers)
target_link_libraries(
  atlasim-noc PRIVATE
  noc
)
