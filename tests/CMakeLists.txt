include(GoogleTest)

add_executable(smoke_test smoke_test.cc)
target_link_libraries(smoke_test PRIVATE inferc_core GTest::gtest_main)
gtest_discover_tests(smoke_test)

add_executable(loader_test loader_test.cc)
target_link_libraries(loader_test PRIVATE inferc_core GTest::gtest_main)
gtest_discover_tests(loader_test)

add_executable(shape_inference_test shape_inference_test.cc)
target_link_libraries(shape_inference_test PRIVATE inferc_core GTest::gtest_main)
target_compile_definitions(shape_inference_test PRIVATE
  INFERC_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(shape_inference_test)

add_executable(kernels_test kernels_test.cc)
target_link_libraries(kernels_test PRIVATE inferc_core GTest::gtest_main)
gtest_discover_tests(kernels_test)

add_executable(correctness_test correctness_test.cc)
target_link_libraries(correctness_test PRIVATE inferc_core GTest::gtest_main)
target_compile_definitions(correctness_test PRIVATE
  INFERC_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(correctness_test)

add_executable(profiler_test profiler_test.cc)
target_link_libraries(profiler_test PRIVATE inferc_core GTest::gtest_main)
gtest_discover_tests(profiler_test)

add_executable(passes_test passes_test.cc)
target_link_libraries(passes_test PRIVATE inferc_core GTest::gtest_main)
gtest_discover_tests(passes_test)

add_executable(amx_probe_test amx_probe_test.cc)
target_link_libraries(amx_probe_test PRIVATE inferc_core GTest::gtest_main)
gtest_discover_tests(amx_probe_test)
