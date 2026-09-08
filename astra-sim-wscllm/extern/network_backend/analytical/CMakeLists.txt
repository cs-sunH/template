# CMake Requirement
cmake_minimum_required(VERSION 3.15)

# C++ requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Setup project
project(Analytical)

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_aware)")

# Can be compiled into either library or executable
option(NETWORK_BACKEND_BUILD_AS_LIBRARY "Build as a library" OFF)

# Focused regression fixture for the bounded, streaming link observer. Kept
# opt-in so ordinary analytical builds retain their existing target graph.
option(ANALYTICAL_BUILD_FLUID_SCHEDULER_LINK_OBSERVER_TEST
       "Build the FluidScheduler link-observer regression test" OFF)

# Focused regression fixture for cancellation of superseded EventQueue and
# FluidScheduler wakeups. Kept opt-in like the observer fixture.
option(ANALYTICAL_BUILD_STALE_EVENT_CANCELLATION_TEST
       "Build the stale-event cancellation regression test" OFF)

# Reuse the repository-vendored yaml-cpp target. The fallback keeps this
# backend buildable as a standalone CMake project without downloading sources.
if(NOT TARGET yaml-cpp)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../../helper/yaml-cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp"
    )
endif()

# Include src files to compile
file(GLOB srcs_common
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/event-queue/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/network-parser/*.cpp
)

file(GLOB srcs_congestion_aware
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/network/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/fluid/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/topology/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/basic-topology/*.cpp
)

# Compile Congestion Aware Backend
if (BUILDTARGET STREQUAL "all" OR BUILDTARGET STREQUAL "congestion_aware")
    if (NETWORK_BACKEND_BUILD_AS_LIBRARY)
        add_library(Analytical_Congestion_Aware STATIC ${srcs_congestion_aware} ${srcs_common})

        # Properties
        set_target_properties(Analytical_Congestion_Aware
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
        )
    else ()
        add_executable(Analytical_Congestion_Aware ${srcs_congestion_aware} ${srcs_common})

        # Properties
        set_target_properties(Analytical_Congestion_Aware
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
        )
    endif ()

    # Common properties
    set_target_properties(Analytical_Congestion_Aware PROPERTIES COMPILE_WARNING_AS_ERROR ON)

    # Link libraries
    target_link_libraries(Analytical_Congestion_Aware PUBLIC yaml-cpp)

    # Include directories
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/)
    target_include_directories(Analytical_Congestion_Aware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/)
endif ()

if (ANALYTICAL_BUILD_FLUID_SCHEDULER_LINK_OBSERVER_TEST)
    if (NOT NETWORK_BACKEND_BUILD_AS_LIBRARY)
        message(FATAL_ERROR
                "ANALYTICAL_BUILD_FLUID_SCHEDULER_LINK_OBSERVER_TEST requires NETWORK_BACKEND_BUILD_AS_LIBRARY=ON")
    endif ()
    if (NOT TARGET Analytical_Congestion_Aware)
        message(FATAL_ERROR
                "ANALYTICAL_BUILD_FLUID_SCHEDULER_LINK_OBSERVER_TEST requires BUILDTARGET=congestion_aware or all")
    endif ()

    enable_testing()
    add_executable(Analytical_FluidSchedulerLinkObserverTest
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/fluid/tests/FluidSchedulerLinkObserverTest.cpp)
    target_link_libraries(Analytical_FluidSchedulerLinkObserverTest PRIVATE Analytical_Congestion_Aware)
    set_target_properties(Analytical_FluidSchedulerLinkObserverTest
            PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    add_test(NAME Analytical_FluidSchedulerLinkObserverTest
             COMMAND Analytical_FluidSchedulerLinkObserverTest)
endif ()

if (ANALYTICAL_BUILD_STALE_EVENT_CANCELLATION_TEST)
    if (NOT NETWORK_BACKEND_BUILD_AS_LIBRARY)
        message(FATAL_ERROR
                "ANALYTICAL_BUILD_STALE_EVENT_CANCELLATION_TEST requires NETWORK_BACKEND_BUILD_AS_LIBRARY=ON")
    endif ()
    if (NOT TARGET Analytical_Congestion_Aware)
        message(FATAL_ERROR
                "ANALYTICAL_BUILD_STALE_EVENT_CANCELLATION_TEST requires BUILDTARGET=congestion_aware or all")
    endif ()

    enable_testing()
    add_executable(Analytical_StaleEventCancellationTest
            ${CMAKE_CURRENT_SOURCE_DIR}/congestion_aware/fluid/tests/StaleEventCancellationTest.cpp)
    target_link_libraries(Analytical_StaleEventCancellationTest
            PRIVATE Analytical_Congestion_Aware)
    set_target_properties(Analytical_StaleEventCancellationTest
            PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    add_test(NAME Analytical_StaleEventCancellationTest
             COMMAND Analytical_StaleEventCancellationTest)
endif ()
