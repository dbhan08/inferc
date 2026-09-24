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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
#include "json.hpp"
#include "profiler/profiler.h"
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

// Speed: 128-token TinyLlama prefill, Accelerate MatMul path vs the Paper-1
// AMX pre-packed path (INFERC_AMX=1), same graph, same inputs. Runs only when
// INFERC_BENCH=1 (loads the 4.4 GB model twice).
TEST(Llama, TinyLlamaPrefill128AmxVsAccelerate) {
  const char* bench = std::getenv("INFERC_BENCH");
  if (!bench || bench[0] != '1') GTEST_SKIP() << "set INFERC_BENCH=1";
  const std::string model = kRoot + "/models/tinyllama/onnx/model.onnx";
  if (!Exists(model) || !Exists(model + "_data")) GTEST_SKIP() << "no TinyLlama weights";

  onnx::ModelProto proto;
  ASSERT_TRUE(inferc::LoadOnnx(model, &proto));
  inferc::Graph g; std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(proto, &g, &err, DirOf(model))) << err;
  proto.Clear();

  std::vector<std::string> past_names;
  for (const auto& in : g.inputs)
    if (in.rfind("past_key_values.", 0) == 0) past_names.push_back(in);
  const inferc::Tensor* pk = g.GetTensor(past_names[0]);
  const int64_t kv_heads = pk->shape[1], head_dim = pk->shape[3];
  const int64_t N = 128;
  std::vector<int64_t> ids(N);
  for (int64_t i = 0; i < N; ++i) ids[i] = 3 + (i * 7919) % 31000;
  auto make_inputs = [&]() {
    std::map<std::string, inferc::rt::Tensor> m;
    for (const auto& n : past_names)
      m[n] = inferc::rt::Tensor::Zeros(inferc::DType::kFloat32, {1, kv_heads, 0, head_dim});
    m["input_ids"] = I64(ids); m["attention_mask"] = Ones(N); m["position_ids"] = Arange(0, N);
    return m;
  };
  // Weight shape per MatMul node (captured before the AMX arm may release
  // unpacked initializer bytes). "dynamic-B" = attention QK / PV matmuls.
  std::map<std::string, std::string> shape_of_node;
  for (const auto& n : g.nodes) {
    if (n.op_type != "MatMul") continue;
    const auto* w = g.GetTensor(n.inputs[1]);
    shape_of_node[n.name] = (w && w->IsInitializer()) ? "W" + inferc::ShapeToString(w->shape)
                                                       : std::string("dynamic-B");
  }
  // One arm = one executor: warm-up, a profiled pass (per-op and per-weight-
  // shape MatMul breakdown), then the median of 7 timed passes. The AMX arm
  // runs LAST because INFERC_AMX=1 releases each packed weight's unpacked
  // bytes from the shared graph (one resident copy; see Executor ctor).
  auto run_arm = [&](const char* amx, std::vector<float>* logits_out) {
    setenv("INFERC_AMX", amx, 1);
    inferc::rt::Executor exec(g);
    auto in = make_inputs();
    for (int w = 0; w < 2; ++w) exec.Run(in);
    inferc::prof::Profiler prof;
    exec.Run(in, &prof);
    std::map<std::string, std::pair<double, int>> by_op, by_shape;
    double total = 0;
    for (const auto& r : prof.iterations().back().ops) {
      by_op[r.op_type].first += r.ms; by_op[r.op_type].second += 1; total += r.ms;
      if (r.op_type == "MatMul") {
        auto& e = by_shape[shape_of_node[r.node_name]]; e.first += r.ms; e.second += 1;
      }
    }
    std::vector<std::pair<std::string, std::pair<double, int>>> v(by_op.begin(), by_op.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
    std::cout << "INFERC_AMX=" << amx << " per-op profile (total " << total << " ms):\n";
    for (size_t i = 0; i < v.size() && i < 8; ++i)
      std::cout << "  " << v[i].first << "  " << v[i].second.first << " ms  x" << v[i].second.second << "\n";
    std::cout << "INFERC_AMX=" << amx << " MatMul by weight shape:\n";
    for (const auto& [k, e] : by_shape)
      std::cout << "  " << k << "  " << e.first << " ms  x" << e.second << "\n";
    std::vector<double> ts;
    std::map<std::string, inferc::rt::Tensor> out;
    for (int r = 0; r < 7; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      out = exec.Run(in);
      ts.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    const auto& l = out.at("logits");
    logits_out->assign(l.data<float>(), l.data<float>() + l.numel());
    return ts[ts.size() / 2];
  };
  std::vector<float> l_acc, l_amx;
  // INFERC_BENCH_ARMS=amx runs only the AMX arm (diagnostics: no Accelerate
  // pass in the same process beforehand).
  const char* arms = std::getenv("INFERC_BENCH_ARMS");
  const bool amx_only = arms && std::string(arms) == "amx";
  const double t_acc = amx_only ? 0.0 : run_arm("0", &l_acc);
  const double t_amx = run_arm("1", &l_amx);
  float md = 0.f;
  for (size_t i = 0; i < l_acc.size(); ++i) md = std::max(md, std::fabs(l_acc[i] - l_amx[i]));
  std::cout << "TinyLlama prefill N=128 (median of 7): Accelerate " << t_acc << " ms, AMX pre-pack "
            << t_amx << " ms, speedup " << t_acc / t_amx << "x, logits max_abs_diff " << md << "\n";
  EXPECT_LE(md, 1e-3f);
}
