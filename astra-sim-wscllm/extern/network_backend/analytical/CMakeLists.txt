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
# P3-3.9 E2 (2026-09-28): whitelist validation, fail-closed. A typo like
# -DBUILDTARGET=Congestion_Aware (wrong case) used to configure successfully
# and silently produce no backend target at all.
set(_BUILDTARGET_VALUES "all" "congestion_aware")
if(NOT BUILDTARGET IN_LIST _BUILDTARGET_VALUES)
    message(FATAL_ERROR
            "BUILDTARGET='${BUILDTARGET}' is invalid; legal values: all, congestion_aware")
endif()

# Can be compiled into either library or executable. Library mode is the
# default contract: the source globs below carry no main(), so the executable
# branch cannot link standalone (historical congestion_aware/main.cc was
# removed with the offline static pipeline). Consumers that need a binary
# (the aggregate AstraSim build) add their own entry-point targets and link
# the Analytical_Congestion_Aware library instead.
option(NETWORK_BACKEND_BUILD_AS_LIBRARY "Build as a library" ON)

# Focused regression fixture for the bounded, streaming link observer. Kept
# opt-in so ordinary analytical builds retain their existing target graph.
# Test-only and manual in this repo for now (2026-09-26): no in-repo runner
# sets this flag yet; configure and ctest it by hand when touching the
# observer path.
option(ANALYTICAL_BUILD_FLUID_SCHEDULER_LINK_OBSERVER_TEST
       "Build the FluidScheduler link-observer regression test" OFF)

# Focused regression fixture for cancellation of superseded EventQueue and
# FluidScheduler wakeups. Kept opt-in like the observer fixture.
# Test-only and manual in this repo for now (2026-09-26): no in-repo runner
# sets this flag yet; configure and ctest it by hand when touching the
# wakeup/alarm cancellation path.
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
# (CONFIGURE_DEPENDS, P3-3.9 E2 2026-09-28: new sources are picked up by an
# existing build tree instead of being silently dropped until reconfigure.)
file(GLOB srcs_common CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/event-queue/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/network-parser/*.cpp
)

file(GLOB srcs_congestion_aware CONFIGURE_DEPENDS
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
        message(FATAL_ERROR
                "NETWORK_BACKEND_BUILD_AS_LIBRARY=OFF (standalone executable) is not supported: the backend sources define no main(). Configure with -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON.")
    endif ()

    # Common properties
    set_target_properties(Analytical_Congestion_Aware PROPERTIES COMPILE_WARNING_AS_ERROR ON)

    # Link libraries
    target_link_libraries(Analytical_Congestion_Aware PUBLIC yaml-cpp)

    # Include directories
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/)
    # P3-3.9 E2 (2026-09-28): the former `${CMAKE_CURRENT_SOURCE_DIR}/extern/`
    # include pointed at extern/network_backend/analytical/extern/, which does
    # not exist; no backend source includes through such a path (all backend
    # includes resolve via include/ above), so the dead line was removed.
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
