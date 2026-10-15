cmake_minimum_required(VERSION 3.22)
project(ProtobufDependencySmoke LANGUAGES CXX)
find_package(Protobuf REQUIRED)
set(CMAKE_CXX_STANDARD 17)
get_target_property(PROTOC_PATH protobuf::protoc IMPORTED_LOCATION)
get_target_property(PROTOBUF_PATH protobuf::libprotobuf IMPORTED_LOCATION_RELEASE)
file(WRITE "${CMAKE_BINARY_DIR}/resolved.txt" "Protobuf_VERSION=${Protobuf_VERSION}\nProtobuf_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}\nProtobuf_LIBRARIES=${Protobuf_LIBRARIES}\nPROTOC_PATH=${PROTOC_PATH}\nPROTOBUF_PATH=${PROTOBUF_PATH}\n")
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/et_def.pb.cc" "${CMAKE_BINARY_DIR}/et_def.pb.h"
 COMMAND protobuf::protoc "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}" "--cpp_out=${CMAKE_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/et_def.proto"
 DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/et_def.proto" protobuf::protoc VERBATIM)
add_executable(protobuf_smoke smoke.cc "${CMAKE_BINARY_DIR}/et_def.pb.cc")
target_include_directories(protobuf_smoke PRIVATE "${CMAKE_BINARY_DIR}" "${Protobuf_INCLUDE_DIR}")
target_link_libraries(protobuf_smoke PRIVATE "${Protobuf_LIBRARIES}")
