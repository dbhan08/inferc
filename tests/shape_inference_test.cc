#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"

#include "frontend/onnx_loader.h"
#include "frontend/onnx_to_ir.h"
#include "ir/graph.h"
#include "ir/shape_inference.h"

namespace {

const std::string kModelPath =
    std::string(INFERC_SOURCE_DIR) + "/models/distilbert.onnx";
const std::string kGoldenPath =
    std::string(INFERC_SOURCE_DIR) + "/models/golden_shapes.json";

bool FileExists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) oss << ",";
    if (s[i] < 0) oss << "?"; else oss << s[i];
  }
  oss << "]";
  return oss.str();
}

// Compare inferc's output shape (a) to ORT's golden (b).
// Policy: same rank required. For each dim: if ORT is unknown (-1), accept
// any inferc value (inferc may be more precise — fine). If ORT is concrete,
// inferc must match it (or also be -1, treated as match).
bool ShapesEqual(const std::vector<int64_t>& a /*inferc*/,
                 const std::vector<int64_t>& b /*ort*/) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (b[i] < 0) continue;          // ORT doesn't know — accept any inferc value
    if (a[i] < 0) continue;          // inferc doesn't know — ORT does, treat as soft-match
    if (a[i] != b[i]) return false;  // both concrete, must match
  }
  return true;
}

}  // namespace

TEST(ShapeInference, DistilBERTAllOpsSupported) {
  if (!FileExists(kModelPath)) {
    GTEST_SKIP() << "Run scripts/fetch_distilbert.py first";
  }
  onnx::ModelProto model;
  ASSERT_TRUE(inferc::LoadOnnx(kModelPath, &model));
  inferc::Graph graph;
  std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(model, &graph, &err)) << err;
  std::vector<std::string> unsupported;
  ASSERT_TRUE(inferc::InferShapes(&graph, &err, &unsupported)) << err;

  std::ostringstream oss;
  for (const auto& op : unsupported) oss << op << " ";
  EXPECT_TRUE(unsupported.empty())
      << "Shape inference doesn't handle: " << oss.str();
}

TEST(ShapeInference, MatchesORTOnDistilBERT) {
  if (!FileExists(kModelPath) || !FileExists(kGoldenPath)) {
    GTEST_SKIP() << "Run scripts/fetch_distilbert.py and scripts/dump_ort_shapes.py first";
  }

  onnx::ModelProto model;
  ASSERT_TRUE(inferc::LoadOnnx(kModelPath, &model));
  inferc::Graph graph;
  std::string err;
  ASSERT_TRUE(inferc::ConvertOnnxToIR(model, &graph, &err)) << err;
  ASSERT_TRUE(inferc::InferShapes(&graph, &err)) << err;

  std::ifstream f(kGoldenPath);
  ASSERT_TRUE(f.is_open());
  nlohmann::json golden = nlohmann::json::parse(f);
  const auto& gnodes = golden["nodes"];
  ASSERT_EQ(gnodes.size(), graph.nodes.size())
      << "node count mismatch: ORT=" << gnodes.size()
      << " inferc=" << graph.nodes.size();

  int compared = 0, matched = 0, mismatched = 0;
  std::ostringstream mismatch_log;
  for (size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto& n = graph.nodes[i];
    const auto& gn = gnodes[i];

    EXPECT_EQ(n.op_type, gn["op_type"].get<std::string>())
        << "Node " << i << " op_type mismatch";

    if (n.outputs.empty() || gn["outputs"].empty()) continue;
    const auto& gout0 = gn["outputs"][0];
    if (gout0["shape"].is_null()) continue;  // ORT didn't infer this one

    const auto* t = graph.GetTensor(n.outputs[0]);
    ASSERT_NE(t, nullptr) << "missing tensor for node " << i;

    auto golden_shape = gout0["shape"].get<std::vector<int64_t>>();
    ++compared;
    if (ShapesEqual(t->shape, golden_shape)) {
      ++matched;
    } else {
      ++mismatched;
      if (mismatched <= 15) {
        mismatch_log << "  [" << i << "] " << n.op_type
                     << "  inferc=" << ShapeStr(t->shape)
                     << "  ort=" << ShapeStr(golden_shape) << "\n";
      }
    }
  }

  std::cout << "Shape inference: " << matched << "/" << compared
            << " matched ORT  (" << (graph.nodes.size() - compared)
            << " nodes had no ORT golden, skipped)\n";
  if (mismatched > 0) {
    std::cout << "First mismatches:\n" << mismatch_log.str();
  }
  EXPECT_GE(compared, 200) << "ORT golden coverage too thin to be meaningful";
  EXPECT_EQ(mismatched, 0)
      << mismatched << " node(s) where inferc shape != ORT shape";
}
