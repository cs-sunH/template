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


# Joint v7 event attribution: consume the cache option declared by the repo
# root CMakeLists (option JOINT_EVENT_PROFILE). This assembly-added
# directory does not inherit the root's directory scope, so the definition
# must be applied here too. Diagnostic-only; never changes behavior.
if(JOINT_EVENT_PROFILE)
    add_compile_definitions(JOINT_V7_EVENT_PROFILE)
endif()

# Joint wall-clock bracket kill-switch (batch 1): consume the cache option
# JOINT_WALL_PROFILE declared by the repo root CMakeLists (same mirroring
# reason as the event-profile block above). Must carry the same value as
# the root and the frontend mirror so the WallBracket inline stays
# ODR-uniform. Measurement only; never changes behavior.
if(JOINT_WALL_PROFILE)
    add_compile_definitions(JOINT_V7_WALL_PROFILE)
endif()

# Compilation target
set(BUILDTARGET "all" CACHE STRING "Compilation target ([all]/congestion_aware)")
# Fail closed on an illegal value: a bare if() over the legal names used to
# configure successfully for any typo and produce zero targets -- the
# breakage only surfaced as a missing binary at build time with no
# diagnosis.
if(NOT BUILDTARGET STREQUAL "all" AND NOT BUILDTARGET STREQUAL "congestion_aware")
    message(FATAL_ERROR
            "BUILDTARGET='${BUILDTARGET}' is not a legal value; supported: all, congestion_aware")
endif()

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
# L24(2026-09-29 审查修复):两处 GLOB 加 CONFIGURE_DEPENDS——增删源文件后
# 不再依赖手工重跑 cmake 才能被发现。
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
        # Joint perf R3 M3 experiment: IPO on the backend archive so
        # CommonNetworkApi (the sim_get_time callee side) joins the LTO
        # partition for speculative devirtualization (option declared by
        # the repo root CMakeLists; default OFF leaves the production path
        # untouched).
        if(JOINT_LTO)
            set_property(TARGET Analytical_Congestion_Aware
                         PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()

        # Properties
        set_target_properties(Analytical_Congestion_Aware
                PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/
        )
    else ()
        # L24(2026-09-29 审查修复):fail-closed——本目录源码无 main()(全目录
        # 仅两个 opt-in 测试可执行体自带 main,见文件末尾两个 if 分支),此前
        # OFF(默认)分支的 add_executable 必然在链接期死于 undefined
        # reference to `main`。仓内唯一受支持路径即
        # -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON(后端作为库供前端链接,见
        # 仓 README §4 ①);显式声明 unsupported,configure 期即报错,不等
        # 链接期才暴露。
        message(FATAL_ERROR
                "NETWORK_BACKEND_BUILD_AS_LIBRARY=OFF (default) is unsupported: "
                "these sources provide no main(); the standalone executable "
                "target lives in the frontend build tree. Configure with "
                "-DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON (repo README section 4, step 1).")
    endif ()

    # Common properties
    set_target_properties(Analytical_Congestion_Aware PROPERTIES COMPILE_WARNING_AS_ERROR ON)

    # Link libraries
    target_link_libraries(Analytical_Congestion_Aware PUBLIC yaml-cpp)

    # Include directories
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    target_include_directories(Analytical_Congestion_Aware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/astra-network-analytical/)
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
