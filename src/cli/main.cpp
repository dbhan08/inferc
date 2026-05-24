#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
#include "ir/passes/fuse_matmul_add_gelu.h"
#include "ir/passes/recognize_gelu.h"
#include "ir/shape_inference.h"
#include "json.hpp"
#include "profiler/profiler.h"
#include "runtime/executor.h"
#include "runtime/tensor.h"
#include "util/version.h"

namespace {

void PrintUsage() {
  std::printf(
      "inferc %.*s\n"
      "Usage: inferc <command> [args]\n"
      "\n"
      "Commands:\n"
      "  inspect <model.onnx> [--ir]\n"
      "      Print model summary; --ir prints the internal IR with shapes.\n"
      "\n"
      "  run <model.onnx> --input-ids <bin> --attention-mask <bin>\n"
      "                   [--output <bin>] [--shape B,S]\n"
      "                   [--profile <out.json>] [-n <iters>] [--warmup <n>]\n"
      "      Execute the model end-to-end. Reads token IDs and attention\n"
      "      mask from binary files (int64 little-endian). With --output,\n"
      "      writes float32 logits. With --profile, runs -n iterations\n"
      "      (default 1) after --warmup warmups (default 0) and writes a\n"
      "      structured JSON report. Default shape is 1,128.\n"
      "\n"
      "  compare <a.json> <b.json>\n"
      "      Print a side-by-side table from two profile JSONs.\n"
      "\n"
      "  bench [--model <onnx>] [--ort-model <onnx>] [--input-ids <bin>]\n"
      "        [--attention-mask <bin>] [-n <iters>] [--warmup <n>] [--outdir <dir>]\n"
      "      Profile inferc + ORT on the same inputs and print the table.\n"
      "      --ort-model defaults to --model; pass the unoptimized .onnx when\n"
      "      --model is an inferc-optimized plan (ORT can't load custom ops).\n"
      "      Defaults assume models/distilbert.onnx + the make_inputs.py outputs.\n"
      "\n"
      "  optimize <model.onnx> --out <model.opt.onnx>\n"
      "      Apply IR passes (recognize-GELU + MatMul+Add+GELU fusion) and\n"
      "      write the optimized graph as ONNX. inferc run can consume it.\n"
      "\n"
      "  decode --model <gpt2.onnx> --past-model <gpt2_with_past.onnx>\n"
      "         --prompt-ids <bin> --max-tokens N --output <bin>\n"
      "      Autoregressive greedy decode for decoder transformers. Prefill\n"
      "      via --model, then run --past-model in a KV-cached loop. Writes\n"
      "      the generated int64 token IDs to --output.\n"
      "\n"
      "Options:\n"
      "  --version, -v        Print version.\n"
      "  --help, -h           This message.\n",
      static_cast<int>(inferc::kVersion.size()), inferc::kVersion.data());
}

void PrintIR(const inferc::Graph& g) {
  std::cout << "IR Graph: " << g.name << "\n";
  std::cout << "  inputs:  ";
  for (size_t i = 0; i < g.inputs.size(); ++i) {
    if (i) std::cout << ", ";
    const auto* t = g.GetTensor(g.inputs[i]);
    std::cout << g.inputs[i];
    if (t) std::cout << ":" << inferc::DTypeName(t->dtype)
                     << inferc::ShapeToString(t->shape);
  }
  std::cout << "\n  outputs: ";
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    if (i) std::cout << ", ";
    const auto* t = g.GetTensor(g.outputs[i]);
    std::cout << g.outputs[i];
    if (t) std::cout << ":" << inferc::DTypeName(t->dtype)
                     << inferc::ShapeToString(t->shape);
  }
  std::cout << "\n\nNodes (" << g.nodes.size() << "):\n";
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    const auto& n = g.nodes[i];
    std::cout << "  [" << std::setw(4) << i << "] "
              << std::left << std::setw(16) << n.op_type;
    if (!n.outputs.empty()) {
      const auto* t = g.GetTensor(n.outputs[0]);
      if (t) std::cout << " -> " << inferc::DTypeName(t->dtype)
                       << inferc::ShapeToString(t->shape);
    }
    std::cout << "\n";
  }
}

int CmdInspect(int argc, char** argv) {
  if (argc < 1) {
    std::fprintf(stderr, "usage: inferc inspect <model.onnx> [--ir]\n");
    return 2;
  }
  const std::string path = argv[0];
  bool show_ir = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--ir") == 0) show_ir = true;
  }

  onnx::ModelProto model;
  if (!inferc::LoadOnnx(path, &model)) {
    std::fprintf(stderr, "inferc: failed to parse '%s'\n", path.c_str());
    return 1;
  }
  if (!show_ir) {
    inferc::ModelSummary summary = inferc::SummarizeModel(model);
    inferc::PrintSummary(summary, std::cout);
    return 0;
  }

  inferc::Graph graph;
  std::string err;
  if (!inferc::ConvertOnnxToIR(model, &graph, &err)) {
    std::fprintf(stderr, "inferc: ONNX->IR failed: %s\n", err.c_str());
    return 1;
  }
  std::vector<std::string> unsupported;
  if (!inferc::InferShapes(&graph, &err, &unsupported)) {
    std::fprintf(stderr, "inferc: shape inference failed: %s\n", err.c_str());
    return 1;
  }
  PrintIR(graph);
  if (!unsupported.empty()) {
    std::cout << "\nWarning: " << unsupported.size()
              << " op type(s) not supported by shape inference:";
    for (const auto& op : unsupported) std::cout << " " << op;
    std::cout << "\n";
  }
  return 0;
}

bool ReadBinaryFile(const std::string& path, std::vector<uint8_t>* bytes) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) return false;
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  bytes->resize(static_cast<size_t>(n));
  return static_cast<bool>(f.read(reinterpret_cast<char*>(bytes->data()), n));
}

bool WriteBinaryFile(const std::string& path, const void* data, size_t n) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) return false;
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
  return f.good();
}

bool ParseShape(const std::string& s, inferc::Shape* out) {
  out->clear();
  size_t i = 0;
  while (i < s.size()) {
    int64_t v = 0;
    bool any = false;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      v = v * 10 + (s[i] - '0');
      ++i;
      any = true;
    }
    if (!any) return false;
    out->push_back(v);
    if (i < s.size() && s[i] == ',') ++i;
  }
  return true;
}

// Rewrite the ONNX ModelProto's graph.node list from the (possibly-passed)
// IR. Initializers, inputs, outputs, value_info are left untouched — passes
// only mutate the node list. Also ensures the "inferc" custom opset is
// registered so downstream tools accept the fused op.
void RewriteOnnxFromIr(const inferc::Graph& g, onnx::ModelProto* model) {
  auto* og = model->mutable_graph();
  og->clear_node();
  bool need_inferc_opset = false;
  for (const auto& n : g.nodes) {
    auto* dst = og->add_node();
    dst->set_op_type(n.op_type);
    if (!n.domain.empty()) {
      dst->set_domain(n.domain);
      if (n.domain == "inferc") need_inferc_opset = true;
    }
    if (!n.name.empty()) dst->set_name(n.name);
    for (const auto& in : n.inputs)  *dst->add_input() = in;
    for (const auto& out : n.outputs) *dst->add_output() = out;
    for (const auto& attr : n.attributes) *dst->add_attribute() = attr;
  }
  if (need_inferc_opset) {
    bool present = false;
    for (const auto& op : model->opset_import()) {
      if (op.domain() == "inferc") { present = true; break; }
    }
    if (!present) {
      auto* op = model->add_opset_import();
      op->set_domain("inferc");
      op->set_version(1);
    }
  }
}

int CmdOptimize(int argc, char** argv) {
  if (argc < 1) {
    std::fprintf(stderr, "usage: inferc optimize <model.onnx> --out <out.onnx>\n");
    return 2;
  }
  const std::string in_path = argv[0];
  std::string out_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else {
      std::fprintf(stderr, "inferc optimize: unknown arg '%s'\n", a.c_str());
      return 2;
    }
  }
  if (out_path.empty()) {
    std::fprintf(stderr, "inferc optimize: --out required\n");
    return 2;
  }

  onnx::ModelProto model;
  if (!inferc::LoadOnnx(in_path, &model)) {
    std::fprintf(stderr, "inferc optimize: failed to parse %s\n", in_path.c_str());
    return 1;
  }
  inferc::Graph graph;
  std::string err;
  if (!inferc::ConvertOnnxToIR(model, &graph, &err)) {
    std::fprintf(stderr, "inferc optimize: ONNX->IR failed: %s\n", err.c_str());
    return 1;
  }
  // Shape inference is not required for the current passes (they match by
  // structure + constant value), but running it provides better diagnostics
  // if a future pass needs shapes.
  if (!inferc::InferShapes(&graph, &err, nullptr)) {
    std::fprintf(stderr,
                 "inferc optimize: warning: pre-pass shape inference failed: %s\n",
                 err.c_str());
  }

  const int n_before = static_cast<int>(graph.nodes.size());
  const int gelus = inferc::passes::RecognizeGelu(&graph);
  const int fused = inferc::passes::FuseMatMulAddGelu(&graph);
  const int n_after = static_cast<int>(graph.nodes.size());

  RewriteOnnxFromIr(graph, &model);

  std::ofstream out(out_path, std::ios::binary);
  if (!out.is_open() || !model.SerializeToOstream(&out)) {
    std::fprintf(stderr, "inferc optimize: failed to write %s\n", out_path.c_str());
    return 1;
  }

  std::cout << "inferc optimize:\n"
            << "  recognize-GELU folded: " << gelus << " patterns\n"
            << "  MatMul+Add+GELU fused: " << fused << " patterns\n"
            << "  nodes: " << n_before << " -> " << n_after
            << " (" << (n_before - n_after) << " removed)\n"
            << "  wrote " << out_path << "\n";
  return 0;
}

struct RunOptions {
  std::string model_path;
  std::string ids_path;
  std::string mask_path;
  std::string out_path;          // optional — write logits here
  std::string profile_out_path;  // optional — write profile JSON here
  std::string shape_str = "1,128";
  std::string name_override;     // optional — overrides backend label in profile
  int iters = 1;
  int warmup = 0;
};

// Derive a backend name: explicit override wins, else "inferc-optimized" if
// the IR contains any fused op, else "inferc-baseline".
std::string PickBackendName(const std::string& override_name,
                            const inferc::Graph& graph) {
  if (!override_name.empty()) return override_name;
  for (const auto& n : graph.nodes) {
    if (n.op_type == "FusedMatMulAddGELU") return "inferc-optimized";
  }
  return "inferc-baseline";
}

int DoRun(const RunOptions& o, const std::string& default_name) {
  inferc::Shape input_shape;
  if (!ParseShape(o.shape_str, &input_shape) || input_shape.size() != 2) {
    std::fprintf(stderr, "inferc run: --shape must be 'B,S' (e.g. 1,128)\n");
    return 2;
  }
  onnx::ModelProto model;
  if (!inferc::LoadOnnx(o.model_path, &model)) {
    std::fprintf(stderr, "inferc run: failed to parse '%s'\n", o.model_path.c_str());
    return 1;
  }
  inferc::Graph graph;
  std::string err;
  if (!inferc::ConvertOnnxToIR(model, &graph, &err)) {
    std::fprintf(stderr, "inferc run: ONNX->IR failed: %s\n", err.c_str());
    return 1;
  }
  std::vector<uint8_t> ids_bytes, mask_bytes;
  if (!ReadBinaryFile(o.ids_path, &ids_bytes)) {
    std::fprintf(stderr, "inferc run: failed to read %s\n", o.ids_path.c_str());
    return 1;
  }
  if (!ReadBinaryFile(o.mask_path, &mask_bytes)) {
    std::fprintf(stderr, "inferc run: failed to read %s\n", o.mask_path.c_str());
    return 1;
  }
  inferc::rt::Tensor input_ids = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, input_shape, ids_bytes.data());
  inferc::rt::Tensor attention_mask = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, input_shape, mask_bytes.data());

  inferc::rt::Executor exec(graph);
  std::map<std::string, inferc::rt::Tensor> in;
  in["input_ids"] = input_ids;
  in["attention_mask"] = attention_mask;

  // Warmup (no profiling).
  for (int i = 0; i < o.warmup; ++i) {
    try { (void)exec.Run(in); }
    catch (const std::exception& e) {
      std::fprintf(stderr, "inferc run: warmup failed: %s\n", e.what());
      return 1;
    }
  }

  inferc::prof::Profiler profiler;
  inferc::prof::Profiler* p = o.profile_out_path.empty() ? nullptr : &profiler;
  std::map<std::string, inferc::rt::Tensor> out_tensors;
  for (int i = 0; i < o.iters; ++i) {
    try { out_tensors = exec.Run(in, p); }
    catch (const std::exception& e) {
      std::fprintf(stderr, "inferc run: executor failed (iter %d): %s\n", i, e.what());
      return 1;
    }
  }
  if (p) p->SnapshotPeakRss();

  if (graph.outputs.empty()) {
    std::fprintf(stderr, "inferc run: graph has no outputs\n");
    return 1;
  }
  const auto& logits = out_tensors.at(graph.outputs[0]);

  if (!o.out_path.empty()) {
    if (!WriteBinaryFile(o.out_path, logits.bytes(),
                         static_cast<size_t>(logits.byte_size()))) {
      std::fprintf(stderr, "inferc run: failed to write %s\n", o.out_path.c_str());
      return 1;
    }
    std::cout << "inferc run: " << graph.outputs[0] << " "
              << inferc::DTypeName(logits.dtype())
              << inferc::ShapeToString(logits.shape()) << " -> " << o.out_path << "\n";
    if (logits.dtype() == inferc::DType::kFloat32 && logits.numel() <= 16) {
      std::cout << "  values:";
      const float* fp = logits.data<float>();
      for (int64_t i = 0; i < logits.numel(); ++i) std::cout << " " << fp[i];
      std::cout << "\n";
    }
  }

  if (p) {
    std::ofstream f(o.profile_out_path);
    if (!f.is_open()) {
      std::fprintf(stderr, "inferc run: failed to open %s for write\n",
                   o.profile_out_path.c_str());
      return 1;
    }
    const std::string backend =
        o.name_override.empty() ? PickBackendName("", graph) : o.name_override;
    (void)default_name;  // legacy parameter, supplanted by auto-detect
    f << p->ToJson(backend, o.model_path);
    f.close();
    auto s = p->TotalStats();
    std::cout << "inferc run: profile -> " << o.profile_out_path
              << " (" << o.iters << " iters, mean=" << std::fixed
              << std::setprecision(2) << s.mean << "ms"
              << " p50=" << s.p50 << "ms p95=" << s.p95 << "ms)\n";
  }
  return 0;
}

int CmdRun(int argc, char** argv) {
  if (argc < 1) {
    std::fprintf(stderr,
        "usage: inferc run <model.onnx> --input-ids <bin> "
        "--attention-mask <bin> [--output <bin>] [--shape B,S] "
        "[--profile <out.json>] [-n <iters>] [--warmup <n>]\n");
    return 2;
  }
  RunOptions o;
  o.model_path = argv[0];
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--input-ids" && i + 1 < argc)            o.ids_path = argv[++i];
    else if (a == "--attention-mask" && i + 1 < argc)  o.mask_path = argv[++i];
    else if (a == "--output" && i + 1 < argc)          o.out_path = argv[++i];
    else if (a == "--profile" && i + 1 < argc)         o.profile_out_path = argv[++i];
    else if (a == "--shape" && i + 1 < argc)           o.shape_str = argv[++i];
    else if (a == "-n" && i + 1 < argc)                o.iters = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc)          o.warmup = std::atoi(argv[++i]);
    else if (a == "--name" && i + 1 < argc)            o.name_override = argv[++i];
    else {
      std::fprintf(stderr, "inferc: unknown run arg '%s'\n", a.c_str());
      return 2;
    }
  }
  if (o.ids_path.empty() || o.mask_path.empty()) {
    std::fprintf(stderr, "inferc run: --input-ids and --attention-mask required\n");
    return 2;
  }
  if (o.out_path.empty() && o.profile_out_path.empty()) {
    std::fprintf(stderr, "inferc run: must provide --output and/or --profile\n");
    return 2;
  }
  if (o.iters < 1) o.iters = 1;
  if (o.warmup < 0) o.warmup = 0;
  return DoRun(o, "inferc-baseline");
}

// ---- compare / bench ----

bool LoadJson(const std::string& path, nlohmann::json* out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  try { f >> *out; } catch (...) { return false; }
  return true;
}

std::string FmtBytesMB(int64_t b) {
  std::ostringstream s;
  s << std::fixed << std::setprecision(1) << (b / (1024.0 * 1024.0));
  return s.str();
}

void PrintCompareTable(const nlohmann::json& a, const nlohmann::json& b) {
  auto bn = [](const nlohmann::json& j) {
    return j.value("backend", std::string("?"));
  };
  auto it = [](const nlohmann::json& j) {
    return j.value("iterations", int64_t{0});
  };
  auto get_total = [](const nlohmann::json& j, const std::string& k) -> double {
    if (!j.contains("total")) return 0.0;
    return j["total"].value(k, 0.0);
  };

  std::cout << "\n=== inferc compare ===\n";
  std::cout << "  model: " << a.value("model", std::string("?")) << "\n\n";

  // Totals
  std::cout << std::left
            << std::setw(20) << "backend"
            << std::right << std::setw(8) << "iters"
            << std::setw(11) << "mean(ms)"
            << std::setw(11) << "p50(ms)"
            << std::setw(11) << "p95(ms)"
            << std::setw(11) << "min(ms)"
            << std::setw(11) << "max(ms)"
            << std::setw(10) << "RSS(MB)"
            << "\n";
  std::cout << std::string(93, '-') << "\n";
  auto row = [&](const nlohmann::json& j) {
    std::cout << std::left << std::setw(20) << bn(j)
              << std::right << std::setw(8) << it(j)
              << std::fixed << std::setprecision(2)
              << std::setw(11) << get_total(j, "mean_ms")
              << std::setw(11) << get_total(j, "p50_ms")
              << std::setw(11) << get_total(j, "p95_ms")
              << std::setw(11) << get_total(j, "min_ms")
              << std::setw(11) << get_total(j, "max_ms")
              << std::setw(10) << FmtBytesMB(j.value("peak_rss_bytes", int64_t{0}))
              << "\n";
  };
  row(a);
  row(b);

  double a_mean = get_total(a, "mean_ms");
  double b_mean = get_total(b, "mean_ms");
  if (a_mean > 0 && b_mean > 0) {
    const bool a_faster = a_mean < b_mean;
    const double factor = a_faster ? (b_mean / a_mean) : (a_mean / b_mean);
    std::cout << "\n  " << bn(a) << " is " << std::fixed << std::setprecision(2)
              << factor << "x " << (a_faster ? "faster" : "slower")
              << " than " << bn(b) << " (mean total)\n";
  }

  // Top ops by total time, joined across both reports.
  if (a.contains("per_op_type") || b.contains("per_op_type")) {
    std::cout << "\nTop ops by mean total time per iter (ratio = " << bn(b)
              << "/" << bn(a) << "):\n";
    std::cout << std::left
              << std::setw(20) << "op_type"
              << std::right << std::setw(11) << "calls/iter"
              << std::setw(16) << (bn(a) + "(ms)").substr(0, 15)
              << std::setw(16) << (bn(b) + "(ms)").substr(0, 15)
              << std::setw(10) << "ratio"
              << "\n";
    std::cout << std::string(73, '-') << "\n";

    // Collect union of op_types, sorted by max mean_ms across the two.
    // calls_per_iter prefers the first (inferc) report's count; ORT's count
    // diverges where ORT has fused ops together.
    std::map<std::string, std::tuple<int64_t, double, double>> rows;
    auto add = [&](const nlohmann::json& j, int slot) {
      if (!j.contains("per_op_type")) return;
      for (auto& [op, val] : j["per_op_type"].items()) {
        auto& row = rows[op];
        if (slot == 0) std::get<1>(row) = val["total_ms"].value("mean_ms", 0.0);
        else           std::get<2>(row) = val["total_ms"].value("mean_ms", 0.0);
        int64_t calls = val.value("calls_per_iter", int64_t{0});
        if (slot == 0 || std::get<0>(row) == 0) std::get<0>(row) = calls;
      }
    };
    add(a, 0);
    add(b, 1);
    std::vector<std::pair<std::string, std::tuple<int64_t, double, double>>> sorted(
        rows.begin(), rows.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& x, const auto& y) {
                double mx = std::max(std::get<1>(x.second), std::get<2>(x.second));
                double my = std::max(std::get<1>(y.second), std::get<2>(y.second));
                return mx > my;
              });
    int shown = 0;
    for (const auto& [op, v] : sorted) {
      if (shown++ >= 12) break;
      int64_t calls = std::get<0>(v);
      double am = std::get<1>(v);
      double bm = std::get<2>(v);
      std::cout << std::left << std::setw(20) << op
                << std::right << std::setw(11) << calls
                << std::fixed << std::setprecision(3)
                << std::setw(16) << am
                << std::setw(16) << bm;
      if (am > 0 && bm > 0) {
        double r = bm / am;
        // 3 decimal places when r < 1 so sub-1 ratios are legible.
        int prec = r < 1.0 ? 3 : 2;
        std::cout << std::setw(10) << std::setprecision(prec) << r;
      } else {
        std::cout << std::setw(10) << "—";
      }
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

int CmdCompare(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: inferc compare <a.json> <b.json>\n");
    return 2;
  }
  nlohmann::json a, b;
  if (!LoadJson(argv[0], &a)) {
    std::fprintf(stderr, "inferc compare: failed to read %s\n", argv[0]);
    return 1;
  }
  if (!LoadJson(argv[1], &b)) {
    std::fprintf(stderr, "inferc compare: failed to read %s\n", argv[1]);
    return 1;
  }
  PrintCompareTable(a, b);
  return 0;
}

int CmdBench(int argc, char** argv) {
  RunOptions o;
  o.model_path = "models/distilbert.onnx";
  o.ids_path = "models/input_ids.bin";
  o.mask_path = "models/attention_mask.bin";
  o.iters = 100;
  o.warmup = 10;
  std::string outdir = "bench_out";
  std::string ort_model_path;  // defaults to o.model_path
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc)              o.model_path = argv[++i];
    else if (a == "--ort-model" && i + 1 < argc)     ort_model_path = argv[++i];
    else if (a == "--input-ids" && i + 1 < argc)     o.ids_path = argv[++i];
    else if (a == "--attention-mask" && i + 1 < argc) o.mask_path = argv[++i];
    else if (a == "-n" && i + 1 < argc)              o.iters = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc)        o.warmup = std::atoi(argv[++i]);
    else if (a == "--outdir" && i + 1 < argc)        outdir = argv[++i];
    else {
      std::fprintf(stderr, "inferc bench: unknown arg '%s'\n", a.c_str());
      return 2;
    }
  }
  if (ort_model_path.empty()) ort_model_path = o.model_path;

  std::string mkdir_cmd = "mkdir -p '" + outdir + "'";
  if (std::system(mkdir_cmd.c_str()) != 0) {
    std::fprintf(stderr, "inferc bench: failed to mkdir %s\n", outdir.c_str());
    return 1;
  }
  std::string inferc_json = outdir + "/inferc_baseline.json";
  std::string ort_json = outdir + "/baseline_ort.json";

  std::cout << "[1/3] Profiling inferc (" << o.iters << " iters + "
            << o.warmup << " warmup)...\n";
  o.profile_out_path = inferc_json;
  if (int rc = DoRun(o, "inferc"); rc != 0) return rc;

  std::cout << "\n[2/3] Profiling ort-cpu (" << o.iters << " iters + "
            << o.warmup << " warmup)...\n";
  std::ostringstream py;
  py << "poetry run python bench/bench_ort.py"
     << " --model " << ort_model_path
     << " --input-ids " << o.ids_path
     << " --attention-mask " << o.mask_path
     << " -n " << o.iters
     << " --warmup " << o.warmup
     << " --out " << ort_json;
  int rc = std::system(py.str().c_str());
  if (rc != 0) {
    std::fprintf(stderr, "inferc bench: bench_ort.py failed (rc=%d)\n", rc);
    return 1;
  }

  std::cout << "\n[3/3] Comparing...\n";
  nlohmann::json a, b;
  if (!LoadJson(inferc_json, &a) || !LoadJson(ort_json, &b)) {
    std::fprintf(stderr, "inferc bench: failed to load output JSONs\n");
    return 1;
  }
  PrintCompareTable(a, b);
  return 0;
}

// ============================================================================
// CmdDecode: autoregressive greedy decode for GPT-2-style decoders.
//
// Uses TWO ONNX models:
//   --model            (no-past)  : full forward over the prompt; the
//                                   "prefill" stage that initializes the cache.
//   --past-model       (with-past): single-token forward that consumes the
//                                   past_key_values cache and emits an
//                                   updated present_key_values cache.
//
// The flow:
//   1. Load prompt token IDs (int64 binary).
//   2. Prefill via --model: input_ids=[1, N], attention_mask=[1, N] all 1s.
//      Read logits[0, N-1] → argmax = first generated token.
//      Read all present.*.{key,value} outputs → initial cache.
//   3. Repeat for --max-tokens steps using --past-model:
//        feed input_ids=[1,1]=[next_token],
//             attention_mask=[1, current_seq_len] all 1s,
//             past_key_values.* = (renamed) present.* from previous step.
//        argmax logits[0,0] → next token; cache = present.*; append token.
//   4. Write generated token IDs (int64) to --output.
// ============================================================================

int CmdDecode(int argc, char** argv) {
  if (argc < 1) {
    std::fprintf(stderr,
        "usage: inferc decode --model <gpt2.onnx> --past-model <gpt2_with_past.onnx>\n"
        "                     --prompt-ids <bin> --max-tokens N --output <bin>\n");
    return 2;
  }
  std::string model_path, past_model_path, ids_path, out_path;
  int max_tokens = 32;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc)            model_path = argv[++i];
    else if (a == "--past-model" && i + 1 < argc)  past_model_path = argv[++i];
    else if (a == "--prompt-ids" && i + 1 < argc)  ids_path = argv[++i];
    else if (a == "--output" && i + 1 < argc)      out_path = argv[++i];
    else if (a == "--max-tokens" && i + 1 < argc)  max_tokens = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "inferc decode: unknown arg '%s'\n", a.c_str());
      return 2;
    }
  }
  if (model_path.empty() || past_model_path.empty() || ids_path.empty() ||
      out_path.empty() || max_tokens <= 0) {
    std::fprintf(stderr, "inferc decode: missing required args\n");
    return 2;
  }

  // ---- Load both models, build executors, infer shapes. ----
  onnx::ModelProto model_a, model_b;
  if (!inferc::LoadOnnx(model_path, &model_a)) {
    std::fprintf(stderr, "inferc decode: failed to parse %s\n", model_path.c_str());
    return 1;
  }
  if (!inferc::LoadOnnx(past_model_path, &model_b)) {
    std::fprintf(stderr, "inferc decode: failed to parse %s\n", past_model_path.c_str());
    return 1;
  }
  inferc::Graph graph_a, graph_b;
  std::string err;
  if (!inferc::ConvertOnnxToIR(model_a, &graph_a, &err)) {
    std::fprintf(stderr, "inferc decode: ONNX->IR (prefill) failed: %s\n", err.c_str());
    return 1;
  }
  if (!inferc::ConvertOnnxToIR(model_b, &graph_b, &err)) {
    std::fprintf(stderr, "inferc decode: ONNX->IR (with-past) failed: %s\n", err.c_str());
    return 1;
  }
  inferc::rt::Executor exec_prefill(graph_a);
  inferc::rt::Executor exec_step(graph_b);

  // ---- Read prompt token IDs. ----
  std::vector<uint8_t> ids_bytes;
  if (!ReadBinaryFile(ids_path, &ids_bytes)) {
    std::fprintf(stderr, "inferc decode: failed to read %s\n", ids_path.c_str());
    return 1;
  }
  const int64_t N = static_cast<int64_t>(ids_bytes.size() / sizeof(int64_t));
  if (N <= 0) {
    std::fprintf(stderr, "inferc decode: empty prompt\n");
    return 1;
  }

  // ---- Helper: build the all-ones int64 attention mask of given length. ----
  auto make_attn_mask = [](int64_t len) {
    inferc::rt::Tensor t(inferc::DType::kInt64, inferc::Shape{1, len});
    int64_t* p = t.data<int64_t>();
    for (int64_t i = 0; i < len; ++i) p[i] = 1;
    return t;
  };

  // ---- Prefill: full forward over the prompt. ----
  inferc::rt::Tensor input_ids = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, {1, N}, ids_bytes.data());
  std::map<std::string, inferc::rt::Tensor> in_prefill;
  in_prefill["input_ids"] = input_ids;
  in_prefill["attention_mask"] = make_attn_mask(N);

  std::map<std::string, inferc::rt::Tensor> out_prefill;
  try {
    out_prefill = exec_prefill.Run(in_prefill);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inferc decode: prefill failed: %s\n", e.what());
    return 1;
  }

  // Read logits[0, N-1] from prefill, argmax → first generated token.
  if (!out_prefill.count("logits")) {
    std::fprintf(stderr, "inferc decode: prefill missing 'logits' output\n");
    return 1;
  }
  const auto& prefill_logits = out_prefill.at("logits");
  const int64_t vocab = prefill_logits.shape().back();
  const float* lp = prefill_logits.data<float>();
  int64_t next_token = 0;
  float best = lp[(N - 1) * vocab];
  for (int64_t v = 1; v < vocab; ++v) {
    float x = lp[(N - 1) * vocab + v];
    if (x > best) { best = x; next_token = v; }
  }

  // Build initial cache: rename `present.*` outputs → `past_key_values.*` inputs.
  std::map<std::string, inferc::rt::Tensor> cache;
  for (auto& [name, t] : out_prefill) {
    if (name.rfind("present.", 0) == 0) {
      cache["past_key_values." + name.substr(std::string("present.").size())] = t;
    }
  }

  // ---- Decode loop. ----
  std::vector<int64_t> generated;
  generated.reserve(max_tokens);
  generated.push_back(next_token);

  int64_t cur_seq_len = N;
  for (int step = 1; step < max_tokens; ++step) {
    cur_seq_len += 1;
    std::map<std::string, inferc::rt::Tensor> in_step = cache;
    int64_t tok_buf = next_token;
    in_step["input_ids"] = inferc::rt::Tensor::FromHostBytes(
        inferc::DType::kInt64, {1, 1}, &tok_buf);
    in_step["attention_mask"] = make_attn_mask(cur_seq_len);

    std::map<std::string, inferc::rt::Tensor> out_step;
    try {
      out_step = exec_step.Run(in_step);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "inferc decode: step %d failed: %s\n", step, e.what());
      return 1;
    }
    const auto& step_logits = out_step.at("logits");
    const float* slp = step_logits.data<float>();
    next_token = 0;
    best = slp[0];
    for (int64_t v = 1; v < vocab; ++v) {
      if (slp[v] > best) { best = slp[v]; next_token = v; }
    }
    generated.push_back(next_token);

    // Refresh cache from present.* outputs.
    cache.clear();
    for (auto& [name, t] : out_step) {
      if (name.rfind("present.", 0) == 0) {
        cache["past_key_values." + name.substr(std::string("present.").size())] = t;
      }
    }
  }

  // ---- Write generated token IDs to --output. ----
  if (!WriteBinaryFile(out_path, generated.data(),
                       generated.size() * sizeof(int64_t))) {
    std::fprintf(stderr, "inferc decode: failed to write %s\n", out_path.c_str());
    return 1;
  }

  std::cout << "inferc decode: generated " << generated.size() << " tokens -> "
            << out_path << "\n  tokens:";
  for (auto t : generated) std::cout << " " << t;
  std::cout << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { PrintUsage(); return 0; }
  if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0) {
    std::printf("%.*s\n", static_cast<int>(inferc::kVersion.size()),
                inferc::kVersion.data());
    return 0;
  }
  if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    PrintUsage();
    return 0;
  }
  const std::string cmd = argv[1];
  if (cmd == "inspect")  return CmdInspect(argc - 2, argv + 2);
  if (cmd == "run")      return CmdRun(argc - 2, argv + 2);
  if (cmd == "optimize") return CmdOptimize(argc - 2, argv + 2);
  if (cmd == "decode")   return CmdDecode(argc - 2, argv + 2);
  if (cmd == "compare")  return CmdCompare(argc - 2, argv + 2);
  if (cmd == "bench")    return CmdBench(argc - 2, argv + 2);

  std::fprintf(stderr, "inferc: unknown command '%s'\n", argv[1]);
  PrintUsage();
  return 1;
}
