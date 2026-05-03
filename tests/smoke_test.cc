#include <gtest/gtest.h>

#include "util/version.h"
#include "onnx.pb.h"

TEST(Smoke, VersionStringIsNonEmpty) {
  EXPECT_FALSE(inferc::kVersion.empty());
}

TEST(Smoke, OnnxProtoLinks) {
  // Construct a default ONNX message — proves protobuf codegen + link works.
  onnx::ModelProto model;
  model.set_ir_version(0);
  EXPECT_EQ(model.ir_version(), 0);
}
