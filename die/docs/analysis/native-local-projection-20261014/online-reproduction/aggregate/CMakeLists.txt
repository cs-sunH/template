cmake_minimum_required(VERSION 3.22)
project(JointIntegration LANGUAGES C CXX)

enable_testing()

if(NOT DEFINED REPOSITORY_ROOT)
    execute_process(
        COMMAND git -C "${CMAKE_CURRENT_LIST_DIR}" rev-parse --show-toplevel
        RESULT_VARIABLE GIT_ROOT_STATUS
        OUTPUT_VARIABLE REPOSITORY_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT GIT_ROOT_STATUS EQUAL 0)
        message(FATAL_ERROR "Set REPOSITORY_ROOT or run inside the source git tree")
    endif()
endif()

if(NOT DEFINED ASTRA_SIM_ROOT)
    set(ASTRA_SIM_ROOT "${REPOSITORY_ROOT}/astra-sim-joint")
endif()

add_subdirectory("${ASTRA_SIM_ROOT}" AstraSim)
add_subdirectory("${ASTRA_SIM_ROOT}/extern/network_backend/analytical" backend)
add_subdirectory("${ASTRA_SIM_ROOT}/astra-sim/network_frontend/analytical" frontend)
