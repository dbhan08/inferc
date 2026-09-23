// Llama-family end-to-end gate: RMSNorm + RoPE + GQA + SwiGLU models run in
// inferc and match ONNX Runtime.
//
//   llama15M   (Xenova/llama2.c-stories15M): exercises If subgraphs, Less,
//              Sigmoid. Prefill logits at every position + 12-token greedy
//              continuation by full recompute.
//   tinyllama  (TinyLlama-1.1B-Chat fp32, 4.4 GB external weights): exercises
//              Greater, Trilu, ScatterND, Sin/Cos, external-data loading and
//              zero-copy weights. Prefill logits + 6-token KV-cached greedy
//              decode through the same merged graph (present.* -> past_key_values.*).
//
// Goldens come from scripts/make_llama_suite.py. Both tests skip when the
// fixtures are absent.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
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

std::string DirOf(const std::string& p) {
  auto i = p.find_last_of('/');
  return i == std::string::npos ? std::string(".") : p.substr(0, i);
}

int64_t Argmax(const float* p, int64_t n) {
  int64_t best = 0;
  for (int64_t i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
  return best;
}

inferc::rt::Tensor I64(const std::vector<int64_t>& v) {
  return inferc::rt::Tensor::FromHostBytes(inferc::DType::kInt64,
                                           {1, static_cast<int64_t>(v.size())}, v.data());
}

inferc::rt::Tensor Ones(int64_t n) {
  std::vector<int64_t> v(static_cast<size_t>(n), 1);
  return I64(v);
}

inferc::rt::Tensor Arange(int64_t start, int64_t n) {
  std::vector<int64_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) v[i] = start + i;
  return I64(v);
}

// Compare inferc prefill logits with ORT's; returns max-abs-diff and the
// fraction of positions whose argmax agrees.
void CheckLogits(const float* got, const std::vector<float>& golden, int64_t N, int64_t V,
                 float* max_diff, float* mag, int* agree) {
  *max_diff = 0.f; *mag = 0.f; *agree = 0;
  for (int64_t p = 0; p < N; ++p) {
    for (int64_t v = 0; v < V; ++v) {
      *max_diff = std::max(*max_diff, std::fabs(got[p * V + v] - golden[p * V + v]));
      *mag = std::max(*mag, std::fabs(golden[p * V + v]));
    }
    if (Argmax(got + p * V, V) == Argmax(golden.data() + p * V, V)) ++*agree;
  }
}

}  // namespace

TEST(Llama, Stories15MForwardAndGreedy) {
  const std::string dir = kRoot + "/models/e2e_suite/llama15m/";
  const std::string model = kRoot + "/models/llama15M.onnx";
  if (!Exists(dir + "manifest.json") || !Exists(model))
    GTEST_SKIP() << "Run scripts/make_llama_suite.py first";
  json manifest; std::ifstream(dir + "manifest.json") >> manifest;

  onnx::ModelProto proto;
  ASSERT_TRUE(inferc::LoadOnnx(model, &proto));
  inferc::Graph g; std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(proto, &g, &err)) << err;
  inferc::rt::Executor exec(g);

  for (const auto& c : manifest["cases"]) {
    const std::string name = c["name"];
    const int64_t N = c["seq_len"], V = c["vocab"];
    const int n_greedy = c["n_greedy"];
    auto ids = ReadBin<int64_t>(dir + name + "_ids.bin");
    auto golden = ReadBin<float>(dir + name + "_logits.bin");
    auto golden_gen = ReadBin<int64_t>(dir + name + "_greedy.bin");

    std::map<std::string, inferc::rt::Tensor> out;
    ASSERT_NO_THROW(out = exec.Run({{"input_ids", I64(ids)}, {"attention_mask", Ones(N)}})) << name;
    const auto& logits = out.at("logits");
    ASSERT_EQ(logits.shape(), (inferc::Shape{1, N, V})) << name;
    float md, mag; int agree;
    CheckLogits(logits.data<float>(), golden, N, V, &md, &mag, &agree);
    EXPECT_LE(md, std::max(1e-3f, 1e-5f * mag)) << name;
    EXPECT_EQ(agree, N) << name << ": per-position argmax agreement";

    // Greedy continuation by full recompute (this export has no KV-cache inputs).
    std::vector<int64_t> cur = ids;
    int mism = 0;
    for (int s = 0; s < n_greedy; ++s) {
      auto o = exec.Run({{"input_ids", I64(cur)}, {"attention_mask", Ones((int64_t)cur.size())}});
      const auto& l = o.at("logits");
      int64_t nxt = Argmax(l.data<float>() + (cur.size() - 1) * V, V);
      if (nxt != golden_gen[s]) ++mism;
      cur.push_back(nxt);
    }
    EXPECT_EQ(mism, 0) << name << ": greedy tokens differ (" << c["greedy_text"] << ")";
    std::cout << "llama15M " << name << ": N=" << N << " max_abs_diff=" << md
              << " argmax " << agree << "/" << N << " greedy " << (n_greedy - mism) << "/"
              << n_greedy << "\n";
  }
}

TEST(Llama, TinyLlama1BForwardAndKVDecode) {
  const std::string dir = kRoot + "/models/e2e_suite/tinyllama/";
  const std::string model = kRoot + "/models/tinyllama/onnx/model.onnx";
  if (!Exists(dir + "manifest.json") || !Exists(model) || !Exists(model + "_data"))
    GTEST_SKIP() << "Run scripts/make_llama_suite.py first (needs models/tinyllama/onnx/model.onnx_data)";
  json manifest; std::ifstream(dir + "manifest.json") >> manifest;

  onnx::ModelProto proto;
  ASSERT_TRUE(inferc::LoadOnnx(model, &proto));
  inferc::Graph g; std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(proto, &g, &err, DirOf(model))) << err;
  proto.Clear();  // weights now live in the IR only (4.4 GB once)
  inferc::rt::Executor exec(g);

  // Discover KV layout from the graph inputs: past_key_values.<i>.{key,value}.
  std::vector<std::string> past_names;
  for (const auto& in : g.inputs)
    if (in.rfind("past_key_values.", 0) == 0) past_names.push_back(in);
  ASSERT_FALSE(past_names.empty());
  const inferc::Tensor* pk = g.GetTensor(past_names[0]);
  ASSERT_TRUE(pk && pk->shape.size() == 4);
  const int64_t kv_heads = pk->shape[1], head_dim = pk->shape[3];

  auto empty_past = [&]() {
    std::map<std::string, inferc::rt::Tensor> m;
    for (const auto& n : past_names)
      m[n] = inferc::rt::Tensor::Zeros(inferc::DType::kFloat32, {1, kv_heads, 0, head_dim});
    return m;
  };

  for (const auto& c : manifest["cases"]) {
    const std::string name = c["name"];
    const int64_t N = c["seq_len"], V = c["vocab"];
    const int n_greedy = c["n_greedy"];
    auto ids = ReadBin<int64_t>(dir + name + "_ids.bin");
    auto golden = ReadBin<float>(dir + name + "_logits.bin");
    auto golden_gen = ReadBin<int64_t>(dir + name + "_greedy.bin");

    auto in = empty_past();
    in["input_ids"] = I64(ids);
    in["attention_mask"] = Ones(N);
    in["position_ids"] = Arange(0, N);
    std::map<std::string, inferc::rt::Tensor> out;
    auto t0 = std::chrono::steady_clock::now();
    ASSERT_NO_THROW(out = exec.Run(in)) << name;
    const double prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const auto& logits = out.at("logits");
    ASSERT_EQ(logits.shape(), (inferc::Shape{1, N, V})) << name;
    float md, mag; int agree;
    CheckLogits(logits.data<float>(), golden, N, V, &md, &mag, &agree);
    EXPECT_LE(md, std::max(1e-3f, 1e-5f * mag)) << name;
    EXPECT_EQ(agree, N) << name << ": per-position argmax agreement";

    // KV-cached greedy decode vs ORT's full-recompute greedy continuation.
    double decode_ms = 0.0;
    std::map<std::string, inferc::rt::Tensor> cache;
    for (auto& [k, t] : out)
      if (k.rfind("present.", 0) == 0) cache["past_key_values." + k.substr(8)] = t;
    int64_t next = Argmax(logits.data<float>() + (N - 1) * V, V);
    int mism = (next != golden_gen[0]) ? 1 : 0;
    int64_t cur = N;
    for (int s = 1; s < n_greedy; ++s) {
      std::map<std::string, inferc::rt::Tensor> st = cache;
      st["input_ids"] = I64({next});
      st["attention_mask"] = Ones(cur + 1);
      st["position_ids"] = Arange(cur, 1);
      auto t1 = std::chrono::steady_clock::now();
      auto o = exec.Run(st);
      decode_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t1).count();
      next = Argmax(o.at("logits").data<float>(), V);
      if (next != golden_gen[s]) ++mism;
      ++cur;
      cache.clear();
      for (auto& [k, t] : o)
        if (k.rfind("present.", 0) == 0) cache["past_key_values." + k.substr(8)] = t;
    }
    EXPECT_EQ(mism, 0) << name << ": greedy tokens differ (" << c["greedy_text"] << ")";
    std::cout << "TinyLlama " << name << ": N=" << N << " max_abs_diff=" << md << " argmax "
              << agree << "/" << N << " greedy " << (n_greedy - mism) << "/" << n_greedy
              << "  prefill " << prefill_ms << " ms, decode " << decode_ms / (n_greedy - 1)
              << " ms/token (fp32, unfused)\n";
  }
}
