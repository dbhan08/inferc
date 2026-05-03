#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
#include "ir/shape_inference.h"
#include "util/version.h"

namespace {

void PrintUsage() {
  std::printf(
      "inferc %.*s\n"
      "Usage: inferc <command> [args]\n"
      "\n"
      "Commands:\n"
      "  inspect <model.onnx> [--ir]      — model summary; --ir prints the IR\n"
      "                                     with inferred shapes per node\n"
      "  optimize <model.onnx> --out ...  — Session 7\n"
      "  run <model|plan> --input ...     — Session 5\n"
      "  compare <a.json> <b.json>        — Session 6\n"
      "  bench                            — Session 6\n"
      "\n"
      "Options:\n"
      "  --version, -v                    — print version and exit\n"
      "  --help, -h                       — this message\n",
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
  std::cout << "\n\n";
  std::cout << "Nodes (" << g.nodes.size() << "):\n";
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    const auto& n = g.nodes[i];
    std::cout << "  [" << std::setw(4) << i << "] "
              << std::left << std::setw(16) << n.op_type;
    // Print first output's resolved shape.
    if (!n.outputs.empty()) {
      const auto* t = g.GetTensor(n.outputs[0]);
      if (t) {
        std::cout << " -> " << inferc::DTypeName(t->dtype)
                  << inferc::ShapeToString(t->shape);
      }
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage();
    return 0;
  }
  if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0) {
    std::printf("%.*s\n", static_cast<int>(inferc::kVersion.size()), inferc::kVersion.data());
    return 0;
  }
  if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
    PrintUsage();
    return 0;
  }

  const std::string cmd = argv[1];
  if (cmd == "inspect") return CmdInspect(argc - 2, argv + 2);

  std::fprintf(stderr, "inferc: unknown command '%s'\n", argv[1]);
  PrintUsage();
  return 1;
}
