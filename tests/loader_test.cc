#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "frontend/onnx_loader.h"
#include "onnx.pb.h"

namespace {

// Build a minimal ONNX model in memory: one Relu node, one input, one output.
// Serializes it to bytes, parses back, summarizes — same flow LoadOnnx uses
// from disk, just bypassing the file I/O.
onnx::ModelProto MakeTinyModel() {
  onnx::ModelProto m;
  m.set_ir_version(7);
  m.set_producer_name("inferc-test");
  m.set_producer_version("0");

  auto* op = m.add_opset_import();
  op->set_version(14);
  op->set_domain("");

  auto* g = m.mutable_graph();
  g->set_name("tiny");

  auto* in = g->add_input();
  in->set_name("x");
  auto* x_type = in->mutable_type()->mutable_tensor_type();
  x_type->set_elem_type(onnx::TensorProto::FLOAT);
  x_type->mutable_shape()->add_dim()->set_dim_value(2);
  x_type->mutable_shape()->add_dim()->set_dim_value(3);

  auto* out = g->add_output();
  out->set_name("y");
  auto* y_type = out->mutable_type()->mutable_tensor_type();
  y_type->set_elem_type(onnx::TensorProto::FLOAT);
  y_type->mutable_shape()->add_dim()->set_dim_value(2);
  y_type->mutable_shape()->add_dim()->set_dim_value(3);

  auto* node = g->add_node();
  node->set_op_type("Relu");
  node->add_input("x");
  node->add_output("y");

  return m;
}

}  // namespace

TEST(Loader, SerializeRoundtripPreservesSummaryFields) {
  onnx::ModelProto m = MakeTinyModel();
  std::string buf;
  ASSERT_TRUE(m.SerializeToString(&buf));

  onnx::ModelProto m2;
  ASSERT_TRUE(m2.ParseFromString(buf));

  auto s = inferc::SummarizeModel(m2);
  EXPECT_EQ(s.ir_version, 7);
  EXPECT_EQ(s.opset_version, 14);
  EXPECT_EQ(s.opset_domain, "ai.onnx");
  EXPECT_EQ(s.graph_name, "tiny");
  EXPECT_EQ(s.node_count, 1);
  EXPECT_EQ(s.op_type_counts["Relu"], 1);

  ASSERT_EQ(s.inputs.size(), 1u);
  EXPECT_EQ(s.inputs[0].name, "x");
  EXPECT_EQ(s.inputs[0].dtype, onnx::TensorProto::FLOAT);
  ASSERT_EQ(s.inputs[0].shape.size(), 2u);
  EXPECT_EQ(s.inputs[0].shape[0], 2);
  EXPECT_EQ(s.inputs[0].shape[1], 3);

  ASSERT_EQ(s.outputs.size(), 1u);
  EXPECT_EQ(s.outputs[0].name, "y");
}

TEST(Loader, PrintSummaryDoesNotCrashOnEmptyModel) {
  onnx::ModelProto empty;
  auto s = inferc::SummarizeModel(empty);
  std::ostringstream oss;
  inferc::PrintSummary(s, oss);
  EXPECT_FALSE(oss.str().empty());
}

TEST(Loader, ElementBytesKnownTypes) {
  EXPECT_EQ(inferc::OnnxElementBytes(onnx::TensorProto::FLOAT), 4);
  EXPECT_EQ(inferc::OnnxElementBytes(onnx::TensorProto::INT64), 8);
  EXPECT_EQ(inferc::OnnxElementBytes(onnx::TensorProto::INT8), 1);
  EXPECT_EQ(inferc::OnnxElementBytes(onnx::TensorProto::FLOAT16), 2);
  EXPECT_EQ(inferc::OnnxElementBytes(onnx::TensorProto::STRING), 0);  // variable
}
