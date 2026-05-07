#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
#include "ir/shape_inference.h"
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
      "                   --output <bin> [--shape B,S]\n"
      "      Execute the model end-to-end. Reads token IDs and attention\n"
      "      mask from binary files (int64 little-endian), writes float32\n"
      "      logits to <bin>. Default shape is 1,128.\n"
      "\n"
      "  optimize <model.onnx> --out ...      (Session 7)\n"
      "  compare <a.json> <b.json>            (Session 6)\n"
      "  bench                                (Session 6)\n"
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

int CmdRun(int argc, char** argv) {
  if (argc < 1) {
    std::fprintf(stderr,
        "usage: inferc run <model.onnx> --input-ids <bin> "
        "--attention-mask <bin> --output <bin> [--shape B,S]\n");
    return 2;
  }
  const std::string model_path = argv[0];
  std::string ids_path, mask_path, out_path, shape_str = "1,128";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--input-ids" && i + 1 < argc)        ids_path = argv[++i];
    else if (a == "--attention-mask" && i + 1 < argc) mask_path = argv[++i];
    else if (a == "--output" && i + 1 < argc)      out_path = argv[++i];
    else if (a == "--shape" && i + 1 < argc)       shape_str = argv[++i];
    else {
      std::fprintf(stderr, "inferc: unknown run arg '%s'\n", a.c_str());
      return 2;
    }
  }
  if (ids_path.empty() || mask_path.empty() || out_path.empty()) {
    std::fprintf(stderr, "inferc run: --input-ids, --attention-mask, --output required\n");
    return 2;
  }
  inferc::Shape input_shape;
  if (!ParseShape(shape_str, &input_shape) || input_shape.size() != 2) {
    std::fprintf(stderr, "inferc run: --shape must be 'B,S' (e.g. 1,128)\n");
    return 2;
  }

  // 1) Load model + IR.
  onnx::ModelProto model;
  if (!inferc::LoadOnnx(model_path, &model)) {
    std::fprintf(stderr, "inferc run: failed to parse '%s'\n", model_path.c_str());
    return 1;
  }
  inferc::Graph graph;
  std::string err;
  if (!inferc::ConvertOnnxToIR(model, &graph, &err)) {
    std::fprintf(stderr, "inferc run: ONNX->IR failed: %s\n", err.c_str());
    return 1;
  }

  // 2) Read inputs.
  std::vector<uint8_t> ids_bytes, mask_bytes;
  if (!ReadBinaryFile(ids_path, &ids_bytes)) {
    std::fprintf(stderr, "inferc run: failed to read %s\n", ids_path.c_str());
    return 1;
  }
  if (!ReadBinaryFile(mask_path, &mask_bytes)) {
    std::fprintf(stderr, "inferc run: failed to read %s\n", mask_path.c_str());
    return 1;
  }
  inferc::rt::Tensor input_ids = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, input_shape, ids_bytes.data());
  inferc::rt::Tensor attention_mask = inferc::rt::Tensor::FromHostBytes(
      inferc::DType::kInt64, input_shape, mask_bytes.data());

  // 3) Build executor and run.
  inferc::rt::Executor exec(graph);
  std::map<std::string, inferc::rt::Tensor> in;
  in["input_ids"] = input_ids;
  in["attention_mask"] = attention_mask;
  std::map<std::string, inferc::rt::Tensor> out;
  try {
    out = exec.Run(in);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inferc run: executor failed: %s\n", e.what());
    return 1;
  }

  // 4) Write logits.
  if (graph.outputs.empty()) {
    std::fprintf(stderr, "inferc run: graph has no outputs\n");
    return 1;
  }
  const auto& logits = out.at(graph.outputs[0]);
  if (!WriteBinaryFile(out_path, logits.bytes(),
                       static_cast<size_t>(logits.byte_size()))) {
    std::fprintf(stderr, "inferc run: failed to write %s\n", out_path.c_str());
    return 1;
  }

  std::cout << "inferc run: " << graph.outputs[0] << " "
            << inferc::DTypeName(logits.dtype())
            << inferc::ShapeToString(logits.shape()) << " -> " << out_path << "\n";
  if (logits.dtype() == inferc::DType::kFloat32 && logits.numel() <= 16) {
    std::cout << "  values:";
    const float* p = logits.data<float>();
    for (int64_t i = 0; i < logits.numel(); ++i) std::cout << " " << p[i];
    std::cout << "\n";
  }
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
  if (cmd == "inspect") return CmdInspect(argc - 2, argv + 2);
  if (cmd == "run")     return CmdRun(argc - 2, argv + 2);

  std::fprintf(stderr, "inferc: unknown command '%s'\n", argv[1]);
  PrintUsage();
  return 1;
}
