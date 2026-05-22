// End-to-end numerical correctness gate for inferc v1.
//
// Loads the real DistilBERT ONNX, the prepared input fixtures, runs inferc's
// executor, and asserts the logits match ORT's golden_logits.bin within the
// project-wide tolerance of 1e-3 max-abs-diff.
//
// This is the load-bearing test for v1: every kernel, every shape, every
// attention mask must be correct for this to pass.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
#include "ir/passes/fuse_matmul_add_gelu.h"
#include "ir/passes/recognize_gelu.h"
#include "runtime/executor.h"
#include "runtime/tensor.h"

namespace {

const std::string kModelPath =
    std::string(INFERC_SOURCE_DIR) + "/models/distilbert.onnx";
const std::string kIdsPath =
    std::string(INFERC_SOURCE_DIR) + "/models/input_ids.bin";
const std::string kMaskPath =
    std::string(INFERC_SOURCE_DIR) + "/models/attention_mask.bin";
const std::string kGoldenPath =
    std::string(INFERC_SOURCE_DIR) + "/models/golden_logits.bin";

bool FileExists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

std::vector<uint8_t> ReadAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(n);
  f.read(reinterpret_cast<char*>(bytes.data()), n);
  return bytes;
}

}  // namespace

TEST(EndToEnd, DistilBERTMatchesORTWithin1eMinus3) {
  if (!FileExists(kModelPath) || !FileExists(kIdsPath) ||
      !FileExists(kMaskPath) || !FileExists(kGoldenPath)) {
    GTEST_SKIP() << "Run scripts/fetch_distilbert.py and scripts/make_inputs.py first";
  }

  // Load the model and convert to IR.
  onnx::ModelProto model;
  ASSERT_TRUE(inferc::LoadOnnx(kModelPath, &model));
  inferc::Graph graph;
  std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(model, &graph, &err)) << err;

  // Prepare inputs: token IDs + attention mask, both int64 [1, 128].
  auto ids_bytes = ReadAll(kIdsPath);
  auto mask_bytes = ReadAll(kMaskPath);
  ASSERT_EQ(ids_bytes.size(), 1u * 128u * sizeof(int64_t));
  ASSERT_EQ(mask_bytes.size(), 1u * 128u * sizeof(int64_t));

  inferc::Shape in_shape = {1, 128};
  inferc::rt::Tensor input_ids =
      inferc::rt::Tensor::FromHostBytes(inferc::DType::kInt64, in_shape,
                                        ids_bytes.data());
  inferc::rt::Tensor attention_mask =
      inferc::rt::Tensor::FromHostBytes(inferc::DType::kInt64, in_shape,
                                        mask_bytes.data());

  // Build executor and run.
  inferc::rt::Executor exec(graph);
  std::map<std::string, inferc::rt::Tensor> inputs = {
      {"input_ids", input_ids},
      {"attention_mask", attention_mask},
  };
  std::map<std::string, inferc::rt::Tensor> outputs;
  ASSERT_NO_THROW(outputs = exec.Run(inputs));

  // Compare to ORT's golden.
  ASSERT_FALSE(outputs.empty());
  const auto& logits = outputs.begin()->second;
  ASSERT_EQ(logits.dtype(), inferc::DType::kFloat32);
  ASSERT_EQ(logits.shape(), (inferc::Shape{1, 2}));

  auto golden_bytes = ReadAll(kGoldenPath);
  ASSERT_EQ(golden_bytes.size(), 2u * sizeof(float));
  const float* golden = reinterpret_cast<const float*>(golden_bytes.data());
  const float* got = logits.data<float>();

  float max_diff = 0.0f;
  for (int64_t i = 0; i < logits.numel(); ++i) {
    float d = std::fabs(got[i] - golden[i]);
    if (d > max_diff) max_diff = d;
  }
  std::cout << "inferc=[" << got[0] << ", " << got[1] << "] "
            << "ort=[" << golden[0] << ", " << golden[1] << "] "
            << "max_abs_diff=" << max_diff << "\n";

  EXPECT_LE(max_diff, 1e-3f) << "numerical correctness gate failed";

  // Sanity: argmax should still be the same class.
  int inferc_pred = got[1] > got[0] ? 1 : 0;
  int golden_pred = golden[1] > golden[0] ? 1 : 0;
  EXPECT_EQ(inferc_pred, golden_pred);
}

TEST(EndToEnd, OptimizedDistilBERTMatchesORTWithin1eMinus3) {
  if (!FileExists(kModelPath) || !FileExists(kIdsPath) ||
      !FileExists(kMaskPath) || !FileExists(kGoldenPath)) {
    GTEST_SKIP() << "Run scripts/fetch_distilbert.py and scripts/make_inputs.py first";
  }

  onnx::ModelProto model;
  ASSERT_TRUE(inferc::LoadOnnx(kModelPath, &model));
  inferc::Graph graph;
  std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(model, &graph, &err)) << err;

  const int before = static_cast<int>(graph.nodes.size());
  const int gelus = inferc::passes::RecognizeGelu(&graph);
  const int fused = inferc::passes::FuseMatMulAddGelu(&graph);
  const int after = static_cast<int>(graph.nodes.size());
  // DistilBERT has 6 transformer blocks, so we expect exactly 6 of each.
  EXPECT_EQ(gelus, 6);
  EXPECT_EQ(fused, 6);
  EXPECT_LT(after, before);

  auto ids_bytes = ReadAll(kIdsPath);
  auto mask_bytes = ReadAll(kMaskPath);
  inferc::Shape in_shape = {1, 128};
  inferc::rt::Tensor input_ids = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, in_shape, ids_bytes.data());
  inferc::rt::Tensor attention_mask = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, in_shape, mask_bytes.data());

  inferc::rt::Executor exec(graph);
  std::map<std::string, inferc::rt::Tensor> inputs = {
      {"input_ids", input_ids},
      {"attention_mask", attention_mask},
  };
  std::map<std::string, inferc::rt::Tensor> outputs;
  ASSERT_NO_THROW(outputs = exec.Run(inputs));

  const auto& logits = outputs.begin()->second;
  auto golden_bytes = ReadAll(kGoldenPath);
  const float* golden = reinterpret_cast<const float*>(golden_bytes.data());
  const float* got = logits.data<float>();
  float max_diff = 0.0f;
  for (int64_t i = 0; i < logits.numel(); ++i) {
    float d = std::fabs(got[i] - golden[i]);
    if (d > max_diff) max_diff = d;
  }
  std::cout << "optimized: inferc=[" << got[0] << ", " << got[1] << "] "
            << "ort=[" << golden[0] << ", " << golden[1] << "] "
            << "max_abs_diff=" << max_diff
            << " (nodes " << before << "->" << after << ")\n";
  EXPECT_LE(max_diff, 1e-3f) << "optimized model fails correctness gate";
}
