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
#include "ir/passes/constant_fold.h"
#include "ir/passes/fuse_matmul_add_gelu.h"
#include "ir/passes/recognize_gelu.h"
#include "ir/passes/recognize_layernorm.h"
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
  const int layernorms = inferc::passes::RecognizeLayerNorm(&graph);
  const int gelus = inferc::passes::RecognizeGelu(&graph);
  const int fused = inferc::passes::FuseMatMulAddGelu(&graph);
  const int after = static_cast<int>(graph.nodes.size());
  // DistilBERT has 6 transformer blocks: 6 GELU + 6 fused FFN, and 13 LayerNorms
  // (2 per block + 1 post-embedding).
  EXPECT_EQ(layernorms, 13);
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

// ---- GPT-2 forward-pass correctness gate (v2 / Session 10) ----

namespace {
const std::string kGpt2ModelPath = std::string(INFERC_SOURCE_DIR) + "/models/gpt2.onnx";
const std::string kGpt2IdsPath   = std::string(INFERC_SOURCE_DIR) + "/models/gpt2_input_ids.bin";
const std::string kGpt2MaskPath  = std::string(INFERC_SOURCE_DIR) + "/models/gpt2_attention_mask.bin";
const std::string kGpt2GoldenPath = std::string(INFERC_SOURCE_DIR) + "/models/gpt2_golden_logits.bin";
constexpr int64_t kGpt2SeqLen = 8;
constexpr int64_t kGpt2VocabSize = 50257;
}

TEST(EndToEnd, GPT2ForwardPassMatchesORT) {
  if (!FileExists(kGpt2ModelPath) || !FileExists(kGpt2IdsPath) ||
      !FileExists(kGpt2MaskPath) || !FileExists(kGpt2GoldenPath)) {
    GTEST_SKIP() << "Run scripts/fetch_gpt2.py and scripts/make_gpt2_inputs.py first";
  }

  onnx::ModelProto model;
  ASSERT_TRUE(inferc::LoadOnnx(kGpt2ModelPath, &model));
  inferc::Graph graph;
  std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(model, &graph, &err)) << err;

  auto ids_bytes = ReadAll(kGpt2IdsPath);
  auto mask_bytes = ReadAll(kGpt2MaskPath);
  ASSERT_EQ(ids_bytes.size(),  static_cast<size_t>(kGpt2SeqLen) * sizeof(int64_t));
  ASSERT_EQ(mask_bytes.size(), static_cast<size_t>(kGpt2SeqLen) * sizeof(int64_t));

  inferc::Shape in_shape = {1, kGpt2SeqLen};
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

  ASSERT_TRUE(outputs.count("logits")) << "GPT-2 graph did not produce 'logits' output";
  const auto& logits = outputs.at("logits");
  ASSERT_EQ(logits.dtype(), inferc::DType::kFloat32);
  ASSERT_EQ(logits.shape(), (inferc::Shape{1, kGpt2SeqLen, kGpt2VocabSize}));

  auto golden_bytes = ReadAll(kGpt2GoldenPath);
  ASSERT_EQ(golden_bytes.size(),
            static_cast<size_t>(kGpt2SeqLen * kGpt2VocabSize) * sizeof(float));
  const float* golden = reinterpret_cast<const float*>(golden_bytes.data());
  const float* got = logits.data<float>();

  float max_diff = 0.0f;
  for (int64_t i = 0; i < logits.numel(); ++i) {
    float d = std::fabs(got[i] - golden[i]);
    if (d > max_diff) max_diff = d;
  }

  // Argmax of the last-position logits — should match ORT's "next-token prediction."
  const int64_t last_pos = kGpt2SeqLen - 1;
  int inferc_argmax = 0, golden_argmax = 0;
  float inferc_max = got[last_pos * kGpt2VocabSize];
  float golden_max = golden[last_pos * kGpt2VocabSize];
  for (int64_t i = 1; i < kGpt2VocabSize; ++i) {
    float gv = got[last_pos * kGpt2VocabSize + i];
    float ov = golden[last_pos * kGpt2VocabSize + i];
    if (gv > inferc_max)  { inferc_max  = gv; inferc_argmax = static_cast<int>(i); }
    if (ov > golden_max)  { golden_max  = ov; golden_argmax = static_cast<int>(i); }
  }

  std::cout << "GPT-2 forward: max_abs_diff=" << max_diff
            << " inferc_argmax=" << inferc_argmax
            << " golden_argmax=" << golden_argmax << "\n";

  EXPECT_LE(max_diff, 1e-3f) << "GPT-2 forward pass fails correctness gate";
  EXPECT_EQ(inferc_argmax, golden_argmax);
}

// ---- GPT-2 greedy decode (v2 / Session 11) ----

namespace {
const std::string kGpt2PastModelPath =
    std::string(INFERC_SOURCE_DIR) + "/models/gpt2_with_past.onnx";
const std::string kGpt2GoldenTokensPath =
    std::string(INFERC_SOURCE_DIR) + "/models/gpt2_golden_tokens.bin";
constexpr int kGpt2GoldenTokenCount = 32;
}

TEST(EndToEnd, GPT2GreedyDecodeMatchesORT) {
  if (!FileExists(kGpt2ModelPath) || !FileExists(kGpt2PastModelPath) ||
      !FileExists(kGpt2IdsPath) || !FileExists(kGpt2GoldenTokensPath)) {
    GTEST_SKIP() << "Run scripts/fetch_gpt2.py and scripts/make_gpt2_inputs.py first";
  }

  // Load both models and build executors.
  onnx::ModelProto model_a, model_b;
  ASSERT_TRUE(inferc::LoadOnnx(kGpt2ModelPath, &model_a));
  ASSERT_TRUE(inferc::LoadOnnx(kGpt2PastModelPath, &model_b));
  inferc::Graph graph_a, graph_b;
  std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(model_a, &graph_a, &err)) << err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(model_b, &graph_b, &err)) << err;
  // Session 13: exercise the full optimized decode path. Constant-folding the
  // LM-head Transpose (and the gated GEMV dispatch, default-on in the kernel)
  // must not change the decoded tokens — and makes this gate ~50x faster.
  inferc::passes::FoldConstantTranspose(&graph_a);
  inferc::passes::FoldConstantTranspose(&graph_b);
  inferc::passes::RecognizeLayerNorm(&graph_a);
  inferc::passes::RecognizeLayerNorm(&graph_b);
  inferc::rt::Executor exec_prefill(graph_a);
  inferc::rt::Executor exec_step(graph_b);

  // Load prompt token IDs.
  auto ids_bytes = ReadAll(kGpt2IdsPath);
  const int64_t N = static_cast<int64_t>(ids_bytes.size() / sizeof(int64_t));

  auto make_mask = [](int64_t len) {
    inferc::rt::Tensor t(inferc::DType::kInt64, inferc::Shape{1, len});
    int64_t* p = t.data<int64_t>();
    for (int64_t i = 0; i < len; ++i) p[i] = 1;
    return t;
  };

  // Prefill.
  inferc::rt::Tensor input_ids = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, {1, N}, ids_bytes.data());
  std::map<std::string, inferc::rt::Tensor> in_pref = {
      {"input_ids", input_ids},
      {"attention_mask", make_mask(N)},
  };
  auto out_pref = exec_prefill.Run(in_pref);
  ASSERT_TRUE(out_pref.count("logits"));
  const auto& prefill_logits = out_pref.at("logits");
  const int64_t vocab = prefill_logits.shape().back();
  const float* lp = prefill_logits.data<float>();
  int64_t next_token = 0;
  float best = lp[(N - 1) * vocab];
  for (int64_t v = 1; v < vocab; ++v) {
    float x = lp[(N - 1) * vocab + v];
    if (x > best) { best = x; next_token = v; }
  }

  // Initial cache.
  std::map<std::string, inferc::rt::Tensor> cache;
  for (auto& [name, t] : out_pref) {
    if (name.rfind("present.", 0) == 0) {
      cache["past_key_values." + name.substr(std::string("present.").size())] = t;
    }
  }

  // Decode loop.
  std::vector<int64_t> generated;
  generated.push_back(next_token);
  int64_t cur_seq_len = N;
  for (int step = 1; step < kGpt2GoldenTokenCount; ++step) {
    cur_seq_len += 1;
    std::map<std::string, inferc::rt::Tensor> in_step = cache;
    int64_t tok_buf = next_token;
    in_step["input_ids"] = inferc::rt::Tensor::FromHostBytes(
        inferc::DType::kInt64, {1, 1}, &tok_buf);
    in_step["attention_mask"] = make_mask(cur_seq_len);
    auto out_step = exec_step.Run(in_step);
    const auto& slogits = out_step.at("logits");
    const float* sp = slogits.data<float>();
    next_token = 0;
    best = sp[0];
    for (int64_t v = 1; v < vocab; ++v) {
      if (sp[v] > best) { best = sp[v]; next_token = v; }
    }
    generated.push_back(next_token);
    cache.clear();
    for (auto& [name, t] : out_step) {
      if (name.rfind("present.", 0) == 0) {
        cache["past_key_values." + name.substr(std::string("present.").size())] = t;
      }
    }
  }

  // Compare to golden.
  auto golden_bytes = ReadAll(kGpt2GoldenTokensPath);
  ASSERT_EQ(golden_bytes.size(),
            static_cast<size_t>(kGpt2GoldenTokenCount) * sizeof(int64_t));
  const int64_t* golden = reinterpret_cast<const int64_t*>(golden_bytes.data());

  int mismatches = 0;
  int first_mismatch = -1;
  for (int i = 0; i < kGpt2GoldenTokenCount; ++i) {
    if (generated[i] != golden[i]) {
      ++mismatches;
      if (first_mismatch < 0) first_mismatch = i;
    }
  }
  if (mismatches > 0) {
    std::cout << "GPT-2 decode mismatch at position " << first_mismatch
              << " inferc=" << generated[first_mismatch]
              << " golden=" << golden[first_mismatch] << "\n";
  } else {
    std::cout << "GPT-2 decode: 32/32 tokens match ORT golden\n";
  }
  EXPECT_EQ(mismatches, 0);
}
