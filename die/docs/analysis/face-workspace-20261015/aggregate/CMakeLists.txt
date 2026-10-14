cmake_minimum_required(VERSION 3.22)
project(FaceWorkspaceIntegration LANGUAGES C CXX)

enable_testing()

if(NOT DEFINED ASTRA_SIM_ROOT)
    set(ASTRA_SIM_ROOT "/workspace/template/astra-sim-face")
endif()

add_subdirectory("${ASTRA_SIM_ROOT}" AstraSim)
add_subdirectory("${ASTRA_SIM_ROOT}/extern/network_backend/analytical" backend)
add_subdirectory("${ASTRA_SIM_ROOT}/astra-sim/network_frontend/analytical" frontend)
