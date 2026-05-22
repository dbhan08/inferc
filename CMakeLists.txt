cmake_minimum_required(VERSION 3.20)
project(inferc LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force Homebrew prefix to win over anaconda for protobuf etc.
list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew")
set(Protobuf_PROTOC_EXECUTABLE "/opt/homebrew/bin/protoc" CACHE FILEPATH "" FORCE)

# Sensible warnings on Clang/GCC
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Dependencies ---
# Use protobuf's own *Config.cmake (ships with brew protobuf), not CMake's bundled FindProtobuf.
# This pulls in abseil transitively, which protobuf 34.x requires.
find_package(absl CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Protobuf: ${Protobuf_VERSION}")

find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
message(STATUS "Accelerate: ${ACCELERATE_FRAMEWORK}")

# nlohmann/json (vendored single header)
add_library(json INTERFACE)
target_include_directories(json INTERFACE ${CMAKE_SOURCE_DIR}/third_party/json)

# --- ONNX protobuf codegen ---
set(ONNX_PROTO ${CMAKE_SOURCE_DIR}/third_party/onnx_pb/onnx.proto)
set(ONNX_PB_SRC ${CMAKE_BINARY_DIR}/onnx_pb/onnx.pb.cc)
set(ONNX_PB_HDR ${CMAKE_BINARY_DIR}/onnx_pb/onnx.pb.h)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/onnx_pb)

add_custom_command(
  OUTPUT ${ONNX_PB_SRC} ${ONNX_PB_HDR}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
          --cpp_out=${CMAKE_BINARY_DIR}/onnx_pb
          --proto_path=${CMAKE_SOURCE_DIR}/third_party/onnx_pb
          ${ONNX_PROTO}
  DEPENDS ${ONNX_PROTO}
  COMMENT "Generating onnx.pb.{h,cc} via protoc ${Protobuf_VERSION}"
  VERBATIM
)

add_library(onnx_pb STATIC ${ONNX_PB_SRC})
target_include_directories(onnx_pb PUBLIC ${CMAKE_BINARY_DIR}/onnx_pb)
target_link_libraries(onnx_pb PUBLIC protobuf::libprotobuf)

# --- inferc_core: shared internals (grows as we add IR, runtime, kernels) ---
add_library(inferc_core STATIC
  src/util/version.cpp
  src/frontend/onnx_loader.cc
  src/frontend/onnx_to_ir.cc
  src/ir/graph.cc
  src/ir/shape_inference.cc
  src/ir/passes/pass_utils.cc
  src/ir/passes/recognize_gelu.cc
  src/ir/passes/fuse_matmul_add_gelu.cc
  src/runtime/tensor.cc
  src/runtime/executor.cc
  src/kernels/matmul.cc
  src/kernels/elementwise.cc
  src/kernels/activation.cc
  src/kernels/embedding.cc
  src/kernels/movement.cc
  src/kernels/fused_matmul_add_gelu.cc
  src/profiler/profiler.cc
)
target_include_directories(inferc_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(inferc_core
  PUBLIC onnx_pb json
  PRIVATE ${ACCELERATE_FRAMEWORK}
)

# --- CLI ---
add_executable(inferc src/cli/main.cpp)
target_link_libraries(inferc PRIVATE inferc_core)

# --- Tests ---
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
