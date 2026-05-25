#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "ir/graph.h"
#include "ir/passes/constant_fold.h"
#include "ir/passes/fuse_matmul_add_gelu.h"
#include "ir/passes/recognize_gelu.h"
#include "onnx.pb.h"

namespace {

using inferc::Graph;
using inferc::Node;
using inferc::passes::FoldConstantTranspose;
using inferc::passes::FuseMatMulAddGelu;
using inferc::passes::RecognizeGelu;

// Build a Constant node whose value attribute is a single fp32 scalar.
Node MakeConstantScalarF32(const std::string& out_name, float v) {
  Node n;
  n.op_type = "Constant";
  n.outputs = {out_name};
  onnx::AttributeProto attr;
  attr.set_name("value");
  attr.set_type(onnx::AttributeProto::TENSOR);
  auto* t = attr.mutable_t();
  t->set_data_type(onnx::TensorProto::FLOAT);
  std::string raw(sizeof(float), '\0');
  std::memcpy(raw.data(), &v, sizeof(float));
  t->set_raw_data(std::move(raw));
  n.attributes.push_back(std::move(attr));
  return n;
}

Node MakeOp(const std::string& op, std::vector<std::string> ins,
            std::vector<std::string> outs) {
  Node n;
  n.op_type = op;
  n.inputs = std::move(ins);
  n.outputs = std::move(outs);
  return n;
}

int CountOp(const Graph& g, const std::string& op) {
  int c = 0;
  for (const auto& n : g.nodes) if (n.op_type == op) ++c;
  return c;
}

}  // namespace

// ---- RecognizeGelu ----

TEST(RecognizeGelu, FoldsExactErfChain) {
  Graph g;
  g.inputs = {"X"};
  g.outputs = {"Y"};
  // Constants: sqrt(2), 1.0, 0.5.
  g.nodes.push_back(MakeConstantScalarF32("c_sqrt2", 1.41421356f));
  g.nodes.push_back(MakeConstantScalarF32("c_one", 1.0f));
  g.nodes.push_back(MakeConstantScalarF32("c_half", 0.5f));
  // GELU chain.
  g.nodes.push_back(MakeOp("Div",  {"X", "c_sqrt2"}, {"div_out"}));
  g.nodes.push_back(MakeOp("Erf",  {"div_out"},      {"erf_out"}));
  g.nodes.push_back(MakeOp("Add",  {"erf_out", "c_one"}, {"plus_one"}));
  g.nodes.push_back(MakeOp("Mul",  {"X", "plus_one"},    {"mul1_out"}));
  g.nodes.push_back(MakeOp("Mul",  {"mul1_out", "c_half"}, {"Y"}));

  const int folded = RecognizeGelu(&g);
  EXPECT_EQ(folded, 1);
  EXPECT_EQ(g.nodes.size(), 1u);
  ASSERT_EQ(g.nodes[0].op_type, "Gelu");
  EXPECT_EQ(g.nodes[0].inputs, std::vector<std::string>{"X"});
  EXPECT_EQ(g.nodes[0].outputs, std::vector<std::string>{"Y"});
}

TEST(RecognizeGelu, AddOperandOrderTolerant) {
  // Verify the matcher accepts both Add(erf,1) and Add(1,erf), and the same
  // for the 0.5-Mul; Div is not commutative and must stay (X, sqrt2).
  Graph g;
  g.inputs = {"X"};
  g.outputs = {"Y"};
  g.nodes.push_back(MakeConstantScalarF32("c_sqrt2", 1.41421356f));
  g.nodes.push_back(MakeConstantScalarF32("c_one", 1.0f));
  g.nodes.push_back(MakeConstantScalarF32("c_half", 0.5f));
  g.nodes.push_back(MakeOp("Div",  {"X", "c_sqrt2"}, {"div_out"}));
  g.nodes.push_back(MakeOp("Erf",  {"div_out"},      {"erf_out"}));
  // Add with const first.
  g.nodes.push_back(MakeOp("Add",  {"c_one", "erf_out"}, {"plus_one"}));
  // Mul with X first.
  g.nodes.push_back(MakeOp("Mul",  {"X", "plus_one"},    {"mul1_out"}));
  // 0.5-Mul with const first.
  g.nodes.push_back(MakeOp("Mul",  {"c_half", "mul1_out"}, {"Y"}));
  EXPECT_EQ(RecognizeGelu(&g), 1);
  EXPECT_EQ(g.nodes.size(), 1u);
}

TEST(RecognizeGelu, BailsOnWrongConstant) {
  // Replace sqrt(2) with some other value — should NOT fold.
  Graph g;
  g.inputs = {"X"};
  g.outputs = {"Y"};
  g.nodes.push_back(MakeConstantScalarF32("c_wrong", 1.5f));
  g.nodes.push_back(MakeConstantScalarF32("c_one", 1.0f));
  g.nodes.push_back(MakeConstantScalarF32("c_half", 0.5f));
  g.nodes.push_back(MakeOp("Div",  {"X", "c_wrong"}, {"div_out"}));
  g.nodes.push_back(MakeOp("Erf",  {"div_out"},      {"erf_out"}));
  g.nodes.push_back(MakeOp("Add",  {"erf_out", "c_one"}, {"plus_one"}));
  g.nodes.push_back(MakeOp("Mul",  {"X", "plus_one"},    {"mul1_out"}));
  g.nodes.push_back(MakeOp("Mul",  {"mul1_out", "c_half"}, {"Y"}));
  EXPECT_EQ(RecognizeGelu(&g), 0);
  EXPECT_EQ(g.nodes.size(), 8u);  // nothing changed
}

TEST(RecognizeGelu, BailsOnExtraConsumer) {
  // If div_out is consumed by more than just Erf, we must NOT fold.
  Graph g;
  g.inputs = {"X"};
  g.outputs = {"Y", "div_out"};  // div_out as a graph output → second use
  g.nodes.push_back(MakeConstantScalarF32("c_sqrt2", 1.41421356f));
  g.nodes.push_back(MakeConstantScalarF32("c_one", 1.0f));
  g.nodes.push_back(MakeConstantScalarF32("c_half", 0.5f));
  g.nodes.push_back(MakeOp("Div",  {"X", "c_sqrt2"}, {"div_out"}));
  g.nodes.push_back(MakeOp("Erf",  {"div_out"},      {"erf_out"}));
  g.nodes.push_back(MakeOp("Add",  {"erf_out", "c_one"}, {"plus_one"}));
  g.nodes.push_back(MakeOp("Mul",  {"X", "plus_one"},    {"mul1_out"}));
  g.nodes.push_back(MakeOp("Mul",  {"mul1_out", "c_half"}, {"Y"}));
  // Note: matcher uses UniqueConsumer (node count == 1), graph outputs do
  // not increment the producer-node count — but the test gives div_out two
  // consumers via an extra Identity-like sink to be safe.
  g.nodes.push_back(MakeOp("Identity", {"div_out"}, {"sink"}));
  EXPECT_EQ(RecognizeGelu(&g), 0);
}

// ---- FuseMatMulAddGelu ----

TEST(FuseMatMulAddGelu, FusesSimpleTriplet) {
  Graph g;
  g.inputs = {"X", "W", "B"};
  g.outputs = {"Y"};
  g.nodes.push_back(MakeOp("MatMul", {"X", "W"}, {"mm_out"}));
  g.nodes.push_back(MakeOp("Add",    {"mm_out", "B"}, {"add_out"}));
  g.nodes.push_back(MakeOp("Gelu",   {"add_out"}, {"Y"}));
  EXPECT_EQ(FuseMatMulAddGelu(&g), 1);
  ASSERT_EQ(g.nodes.size(), 1u);
  EXPECT_EQ(g.nodes[0].op_type, "FusedMatMulAddGELU");
  EXPECT_EQ(g.nodes[0].domain, "inferc");
  ASSERT_EQ(g.nodes[0].inputs.size(), 3u);
  EXPECT_EQ(g.nodes[0].inputs[0], "X");
  EXPECT_EQ(g.nodes[0].inputs[1], "W");
  EXPECT_EQ(g.nodes[0].inputs[2], "B");
  EXPECT_EQ(g.nodes[0].outputs, std::vector<std::string>{"Y"});
}

TEST(FuseMatMulAddGelu, BiasOperandOrder) {
  // Add takes (B, mm_out) instead of (mm_out, B) — must still fuse with B as bias.
  Graph g;
  g.inputs = {"X", "W", "B"};
  g.outputs = {"Y"};
  g.nodes.push_back(MakeOp("MatMul", {"X", "W"}, {"mm_out"}));
  g.nodes.push_back(MakeOp("Add",    {"B", "mm_out"}, {"add_out"}));
  g.nodes.push_back(MakeOp("Gelu",   {"add_out"}, {"Y"}));
  EXPECT_EQ(FuseMatMulAddGelu(&g), 1);
  ASSERT_EQ(g.nodes.size(), 1u);
  EXPECT_EQ(g.nodes[0].inputs[2], "B");
}

TEST(FuseMatMulAddGelu, BailsWhenMatMulOutputHasExtraConsumer) {
  // If something else reads mm_out, we cannot remove the MatMul.
  Graph g;
  g.inputs = {"X", "W", "B"};
  g.outputs = {"Y", "mm_out"};
  g.nodes.push_back(MakeOp("MatMul", {"X", "W"}, {"mm_out"}));
  g.nodes.push_back(MakeOp("Add",    {"mm_out", "B"}, {"add_out"}));
  g.nodes.push_back(MakeOp("Gelu",   {"add_out"}, {"Y"}));
  EXPECT_EQ(FuseMatMulAddGelu(&g), 0);
  // Plus an extra direct consumer.
  g.nodes.push_back(MakeOp("Identity", {"mm_out"}, {"sink"}));
  EXPECT_EQ(FuseMatMulAddGelu(&g), 0);
}

TEST(FuseMatMulAddGelu, NoMatchIfNoGelu) {
  // MatMul → Add but no Gelu downstream. Should not fuse.
  Graph g;
  g.inputs = {"X", "W", "B"};
  g.outputs = {"add_out"};
  g.nodes.push_back(MakeOp("MatMul", {"X", "W"}, {"mm_out"}));
  g.nodes.push_back(MakeOp("Add",    {"mm_out", "B"}, {"add_out"}));
  EXPECT_EQ(FuseMatMulAddGelu(&g), 0);
  EXPECT_EQ(CountOp(g, "FusedMatMulAddGELU"), 0);
}

// ---- Constant folding: Transpose-of-initializer ----

// Make a float32 initializer tensor and register it in the graph.
void AddF32Initializer(Graph* g, const std::string& name, inferc::Shape shape,
                       const std::vector<float>& data) {
  inferc::Tensor t;
  t.name = name;
  t.dtype = inferc::DType::kFloat32;
  t.shape = std::move(shape);
  t.raw_data.resize(data.size() * sizeof(float));
  std::memcpy(t.raw_data.data(), data.data(), t.raw_data.size());
  g->tensors[name] = std::move(t);
}

Node MakeTranspose(const std::string& in, const std::string& out,
                   std::vector<int64_t> perm) {
  Node n = MakeOp("Transpose", {in}, {out});
  onnx::AttributeProto attr;
  attr.set_name("perm");
  attr.set_type(onnx::AttributeProto::INTS);
  for (int64_t p : perm) attr.add_ints(p);
  n.attributes.push_back(std::move(attr));
  return n;
}

TEST(FoldConstantTranspose, FoldsInitializerTranspose) {
  // W = [[1,2,3],[4,5,6]]  (shape [2,3]); Transpose perm [1,0] -> [3,2].
  Graph g;
  AddF32Initializer(&g, "W", {2, 3}, {1, 2, 3, 4, 5, 6});
  g.nodes.push_back(MakeTranspose("W", "Wt", {1, 0}));
  g.outputs = {"Wt_consumer"};
  // A consumer so Wt isn't a graph output (folds only non-outputs).
  g.nodes.push_back(MakeOp("Identity", {"Wt"}, {"Wt_consumer"}));

  EXPECT_EQ(FoldConstantTranspose(&g), 1);
  // Transpose node removed; Identity remains.
  EXPECT_EQ(CountOp(g, "Transpose"), 0);
  EXPECT_EQ(CountOp(g, "Identity"), 1);

  // Wt is now an initializer holding the transposed data.
  const inferc::Tensor* wt = g.GetTensor("Wt");
  ASSERT_NE(wt, nullptr);
  EXPECT_TRUE(wt->IsInitializer());
  EXPECT_EQ(wt->shape, (inferc::Shape{3, 2}));
  const float* d = reinterpret_cast<const float*>(wt->raw_data.data());
  // Transposed row-major: [[1,4],[2,5],[3,6]] -> 1,4,2,5,3,6
  EXPECT_FLOAT_EQ(d[0], 1); EXPECT_FLOAT_EQ(d[1], 4);
  EXPECT_FLOAT_EQ(d[2], 2); EXPECT_FLOAT_EQ(d[3], 5);
  EXPECT_FLOAT_EQ(d[4], 3); EXPECT_FLOAT_EQ(d[5], 6);

  // Original initializer W is left intact (it may have other consumers).
  const inferc::Tensor* w = g.GetTensor("W");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->shape, (inferc::Shape{2, 3}));
  EXPECT_FLOAT_EQ(reinterpret_cast<const float*>(w->raw_data.data())[1], 2);
}

TEST(FoldConstantTranspose, SkipsNonInitializerAndGraphOutput) {
  // Transpose of an activation (non-initializer) is NOT folded.
  Graph g1;
  g1.inputs = {"A"};
  g1.outputs = {"At_sink"};
  g1.nodes.push_back(MakeTranspose("A", "At", {1, 0}));
  g1.nodes.push_back(MakeOp("Identity", {"At"}, {"At_sink"}));
  EXPECT_EQ(FoldConstantTranspose(&g1), 0);
  EXPECT_EQ(CountOp(g1, "Transpose"), 1);

  // Transpose of an initializer whose output IS a graph output is left alone.
  Graph g2;
  AddF32Initializer(&g2, "W", {2, 2}, {1, 2, 3, 4});
  g2.outputs = {"Wt"};
  g2.nodes.push_back(MakeTranspose("W", "Wt", {1, 0}));
  EXPECT_EQ(FoldConstantTranspose(&g2), 0);
  EXPECT_EQ(CountOp(g2, "Transpose"), 1);
}
