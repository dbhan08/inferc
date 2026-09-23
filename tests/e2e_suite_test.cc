// Broad end-to-end correctness suite (not hand-picked inputs).
//
// Walks models/e2e_suite/{distilbert,gpt2}/manifest.json, produced by
// scripts/make_e2e_suite.py: curated sentences of varying length and content,
// seeded random-token sequences, and several padded lengths. Every case is
// checked against ONNX Runtime golden output on both the raw graph and the
// fully fused graph.
//
// Gates:
//   DistilBERT  max-abs-diff <= 1e-5 on logits, same argmax.
//   GPT-2       max-abs-diff <= max(1e-3, 1e-5 * max|logit|), last-position
//               argmax equal, >= 95% per-position argmax agreement, and the
//               16-token KV-cached greedy continuation equal to ORT's.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
#include "ir/passes/constant_fold.h"
#include "ir/passes/fuse_matmul_add_gelu.h"
#include "ir/passes/recognize_attention.h"
#include "ir/passes/recognize_gelu.h"
#include "ir/passes/recognize_layernorm.h"
#include "json.hpp"
#include "runtime/executor.h"
#include "runtime/tensor.h"

namespace {

using json = nlohmann::json;
const std::string kRoot = std::string(INFERC_SOURCE_DIR);

bool Exists(const std::string& p) { std::ifstream f(p); return f.good(); }

template <typename T>
std::vector<T> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.good()) << path;
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<T> v(static_cast<size_t>(n) / sizeof(T));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

bool LoadGraph(const std::string& path, inferc::Graph* g) {
  onnx::ModelProto model;
  if (!inferc::LoadOnnx(path, &model)) return false;
  std::string err;
  return inferc::ConvertOnnxToIR(model, g, &err);
}

void FuseAll(inferc::Graph* g) {
  inferc::passes::FoldConstantTranspose(g);
  inferc::passes::RecognizeAttention(g);
  inferc::passes::FuseQKVProjection(g);
  inferc::passes::RecognizeLayerNorm(g);
  inferc::passes::RecognizeGelu(g);
  inferc::passes::RecognizeGeluTanh(g);
  inferc::passes::FuseMatMulAddGelu(g);
}

inferc::rt::Tensor I64(const std::vector<int64_t>& v, int64_t len) {
  return inferc::rt::Tensor::FromHostBytes(inferc::DType::kInt64, {1, len}, v.data());
}

int64_t Argmax(const float* p, int64_t n) {
  int64_t best = 0;
  for (int64_t i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
  return best;
}

struct DistilCase { std::string name; int64_t seq_len; std::string note; };

void RunDistilBERT(bool fused) {
  const std::string dir = kRoot + "/models/e2e_suite/distilbert/";
  const std::string model = kRoot + "/models/distilbert.onnx";
  if (!Exists(dir + "manifest.json") || !Exists(model))
    GTEST_SKIP() << "Run scripts/make_e2e_suite.py first";
  json manifest; std::ifstream(dir + "manifest.json") >> manifest;

  inferc::Graph g;
  ASSERT_TRUE(LoadGraph(model, &g));
  if (fused) FuseAll(&g);
  inferc::rt::Executor exec(g);

  int n_cases = 0; float worst = 0.f;
  for (const auto& c : manifest["cases"]) {
    const std::string name = c["name"];
    const int64_t L = c["seq_len"];
    auto ids = ReadBin<int64_t>(dir + name + "_ids.bin");
    auto mask = ReadBin<int64_t>(dir + name + "_mask.bin");
    auto golden = ReadBin<float>(dir + name + "_logits.bin");
    ASSERT_EQ(ids.size(), size_t(L));
    std::map<std::string, inferc::rt::Tensor> in = {
        {"input_ids", I64(ids, L)}, {"attention_mask", I64(mask, L)}};
    std::map<std::string, inferc::rt::Tensor> out;
    ASSERT_NO_THROW(out = exec.Run(in)) << name;
    const auto& logits = out.begin()->second;
    ASSERT_EQ(logits.numel(), int64_t(golden.size())) << name;
    const float* got = logits.data<float>();
    float md = 0.f;
    for (size_t i = 0; i < golden.size(); ++i) md = std::max(md, std::fabs(got[i] - golden[i]));
    worst = std::max(worst, md);
    EXPECT_LE(md, 1e-5f) << name << " (" << c["note"] << ")";
    EXPECT_EQ(Argmax(got, 2), Argmax(golden.data(), 2)) << name;
    ++n_cases;
  }
  std::cout << "DistilBERT e2e suite (" << (fused ? "fused" : "raw") << "): " << n_cases
            << " cases, worst max_abs_diff=" << worst << "\n";
  EXPECT_GE(n_cases, 12);
}

}  // namespace

TEST(E2ESuite, DistilBERTRawGraph) { RunDistilBERT(false); }
TEST(E2ESuite, DistilBERTFusedGraph) { RunDistilBERT(true); }

TEST(E2ESuite, GPT2ForwardAndGreedyDecode) {
  const std::string dir = kRoot + "/models/e2e_suite/gpt2/";
  const std::string model = kRoot + "/models/gpt2.onnx";
  const std::string model_past = kRoot + "/models/gpt2_with_past.onnx";
  if (!Exists(dir + "manifest.json") || !Exists(model) || !Exists(model_past))
    GTEST_SKIP() << "Run scripts/make_e2e_suite.py first";
  json manifest; std::ifstream(dir + "manifest.json") >> manifest;

  inferc::Graph ga, gb;
  ASSERT_TRUE(LoadGraph(model, &ga));
  ASSERT_TRUE(LoadGraph(model_past, &gb));
  // Same optimized pipeline as `inferc decode`.
  for (auto* g : {&ga, &gb}) {
    inferc::passes::FoldConstantTranspose(g);
    inferc::passes::RecognizeLayerNorm(g);
    inferc::passes::RecognizeGeluTanh(g);
  }
  inferc::rt::Executor prefill(ga), step(gb);

  auto ones = [](int64_t len) {
    inferc::rt::Tensor t(inferc::DType::kInt64, inferc::Shape{1, len});
    for (int64_t i = 0; i < len; ++i) t.data<int64_t>()[i] = 1;
    return t;
  };

  int n_cases = 0; float worst = 0.f;
  for (const auto& c : manifest["cases"]) {
    const std::string name = c["name"];
    const int64_t N = c["seq_len"];
    const int64_t V = c["vocab"];
    const int n_greedy = c["n_greedy"];
    auto ids = ReadBin<int64_t>(dir + name + "_ids.bin");
    auto golden = ReadBin<float>(dir + name + "_logits.bin");
    auto golden_gen = ReadBin<int64_t>(dir + name + "_greedy.bin");
    ASSERT_EQ(golden.size(), size_t(N * V)) << name;

    std::map<std::string, inferc::rt::Tensor> in = {
        {"input_ids", I64(ids, N)}, {"attention_mask", ones(N)}};
    std::map<std::string, inferc::rt::Tensor> out;
    ASSERT_NO_THROW(out = prefill.Run(in)) << name;
    ASSERT_TRUE(out.count("logits")) << name;
    const auto& logits = out.at("logits");
    ASSERT_EQ(logits.shape(), (inferc::Shape{1, N, V})) << name;
    const float* got = logits.data<float>();

    float md = 0.f, mag = 0.f; int agree = 0;
    for (int64_t p = 0; p < N; ++p) {
      for (int64_t v = 0; v < V; ++v) {
        md = std::max(md, std::fabs(got[p * V + v] - golden[p * V + v]));
        mag = std::max(mag, std::fabs(golden[p * V + v]));
      }
      if (Argmax(got + p * V, V) == Argmax(golden.data() + p * V, V)) ++agree;
    }
    worst = std::max(worst, md);
    // GPT-2 logits are O(100), so gate at 1e-5 relative to the largest logit
    // (fp32 rounding), floored at 1e-3 absolute.
    EXPECT_LE(md, std::max(1e-3f, 1e-5f * mag)) << name << " (" << c["note"] << ") mag=" << mag;
    EXPECT_EQ(Argmax(got + (N - 1) * V, V), Argmax(golden.data() + (N - 1) * V, V)) << name;
    EXPECT_GE(double(agree) / double(N), 0.95) << name << ": per-position argmax agreement "
                                                << agree << "/" << N;

    // KV-cached greedy continuation vs ORT's full-recompute greedy continuation.
    std::map<std::string, inferc::rt::Tensor> cache;
    for (auto& [k, t] : out)
      if (k.rfind("present.", 0) == 0) cache["past_key_values." + k.substr(8)] = t;
    int64_t next = Argmax(got + (N - 1) * V, V);
    std::vector<int64_t> gen{next};
    int64_t cur = N;
    for (int s = 1; s < n_greedy; ++s) {
      ++cur;
      std::map<std::string, inferc::rt::Tensor> in_step = cache;
      int64_t tok = next;
      in_step["input_ids"] = inferc::rt::Tensor::FromHostBytes(inferc::DType::kInt64, {1, 1}, &tok);
      in_step["attention_mask"] = ones(cur);
      auto o = step.Run(in_step);
      next = Argmax(o.at("logits").data<float>(), V);
      gen.push_back(next);
      cache.clear();
      for (auto& [k, t] : o)
        if (k.rfind("present.", 0) == 0) cache["past_key_values." + k.substr(8)] = t;
    }
    int mism = 0;
    for (int i = 0; i < n_greedy; ++i) if (gen[i] != golden_gen[i]) ++mism;
    EXPECT_EQ(mism, 0) << name << ": greedy tokens differ from ORT (" << c["greedy_text"] << ")";
    std::cout << "GPT-2 " << name << ": N=" << N << " max_abs_diff=" << md << " argmax agree "
              << agree << "/" << N << " greedy " << (n_greedy - mism) << "/" << n_greedy << "\n";
    ++n_cases;
  }
  std::cout << "GPT-2 e2e suite: " << n_cases << " cases, worst max_abs_diff=" << worst << "\n";
  EXPECT_GE(n_cases, 6);
}
