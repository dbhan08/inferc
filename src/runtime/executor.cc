#include "runtime/executor.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "frontend/onnx_to_ir.h"
#include "kernels/activation.h"
#include "kernels/attention.h"
#include "kernels/elementwise.h"
#include "kernels/embedding.h"
#include "kernels/fused_matmul_add_gelu.h"
#include "kernels/matmul.h"
#include "kernels/movement.h"
#include "profiler/profiler.h"

namespace inferc {
namespace rt {

namespace {

// Build a runtime Tensor that BORROWS an IR initializer's bytes (no copy).
// The Graph outlives the Executor (it holds a pointer to it), so the view is
// safe; this keeps a 4.4 GB fp32 Llama resident once, not twice.
Tensor MakeFromIR(const ::inferc::Tensor& t) {
  std::shared_ptr<uint8_t[]> storage(const_cast<uint8_t*>(t.raw_data.data()),
                                     [](uint8_t*) {});
  Shape strides(t.shape.size(), 1);
  for (int64_t d = static_cast<int64_t>(t.shape.size()) - 2; d >= 0; --d)
    strides[d] = strides[d + 1] * t.shape[d + 1];
  return Tensor::BorrowingView(t.dtype, t.shape, std::move(storage), 0, std::move(strides));
}

// Read a 1D int64 tensor's contents into a vector. Used for e.g. Reshape's
// shape input.
std::vector<int64_t> ReadInt64Vec(const Tensor& t) {
  if (t.dtype() != DType::kInt64) {
    throw std::runtime_error("Executor: expected int64 tensor");
  }
  Tensor c = t.Contiguous();
  const int64_t n = c.numel();
  std::vector<int64_t> v(n);
  std::memcpy(v.data(), c.bytes(), static_cast<size_t>(n * 8));
  return v;
}

// Build a runtime Tensor from a Constant node's value attribute.
Tensor MaterializeConstant(const Node& node) {
  const auto* val = node.GetAttr("value");
  if (val && val->type() == onnx::AttributeProto::TENSOR) {
    const auto& t = val->t();
    DType dt = DTypeFromOnnx(t.data_type());
    Shape shape(t.dims().begin(), t.dims().end());
    if (!t.raw_data().empty()) {
      return Tensor::FromHostBytes(dt, shape, t.raw_data().data());
    }
    // Typed-data fallback for the common dtypes.
    Tensor out = Tensor::Zeros(dt, shape);
    switch (t.data_type()) {
      case onnx::TensorProto::FLOAT: {
        float* p = out.data<float>();
        for (int i = 0; i < t.float_data_size(); ++i) p[i] = t.float_data(i);
        break;
      }
      case onnx::TensorProto::INT64: {
        int64_t* p = out.data<int64_t>();
        for (int i = 0; i < t.int64_data_size(); ++i) p[i] = t.int64_data(i);
        break;
      }
      case onnx::TensorProto::INT32: {
        int32_t* p = out.data<int32_t>();
        for (int i = 0; i < t.int32_data_size(); ++i) p[i] = t.int32_data(i);
        break;
      }
      default:
        throw std::runtime_error("Constant: unsupported typed-data dtype");
    }
    return out;
  }
  if ((val = node.GetAttr("value_int"))) {
    int64_t v = val->i();
    Tensor out = Tensor::Zeros(DType::kInt64, {});
    *out.data<int64_t>() = v;
    return out;
  }
  if ((val = node.GetAttr("value_float"))) {
    float v = val->f();
    Tensor out = Tensor::Zeros(DType::kFloat32, {});
    *out.data<float>() = v;
    return out;
  }
  throw std::runtime_error("Constant: no recognized value attribute");
}

}  // namespace

Executor::Executor(const Graph& graph) : graph_(&graph) {
  // Materialize all initializers once.
  for (const auto& [name, t] : graph.tensors) {
    if (t.IsInitializer()) {
      initializers_[name] = MakeFromIR(t);
    }
  }
  PrepareBranches(graph);
}

void Executor::PrepareBranches(const Graph& g) {
  for (const auto& node : g.nodes) {
    if (node.op_type != "If") continue;
    for (const auto& attr : node.attributes) {
      if (attr.type() != onnx::AttributeProto::GRAPH) continue;
      auto sub = std::make_unique<Graph>();
      std::string err;
      if (!ConvertGraphProtoToIR(attr.g(), sub.get(), &err)) {
        throw std::runtime_error("If: failed to convert branch '" + attr.name() + "': " + err);
      }
      PrepareBranches(*sub);  // nested Ifs
      branches_[&attr] = std::move(sub);
    }
  }
}

void Executor::RunNodes(const Graph& g, Tape& tape, prof::Profiler* profiler) const {
  for (const auto& node : g.nodes) {
    try {
      ExecNode(node, tape, profiler);
    } catch (const std::runtime_error& e) {
      // Annotate once with the failing node; nested If branches rethrow as-is.
      const std::string what = e.what();
      if (what.rfind("[node ", 0) == 0) throw;
      std::string detail = what + " [node " + node.name + " (" + node.op_type + ")";
      for (const auto& in : node.inputs) {
        auto it = tape.find(in);
        if (it != tape.end()) detail += " " + in + "=" + ShapeToString(it->second.shape());
      }
      throw std::runtime_error("[node " + node.name + "] " + detail + "]");
    }
  }
}

void Executor::ExecNode(const Node& node, Tape& tape, prof::Profiler* profiler) const {
  auto get = [&](const std::string& name) -> const Tensor& {
    auto it = tape.find(name);
    if (it == tape.end()) {
      throw std::runtime_error("Executor: missing tensor '" + name + "'");
    }
    return it->second;
  };
  // Sum bytes of live tensors that are NOT initializers (i.e., activations + inputs).
  auto live_activation_bytes = [&]() -> int64_t {
    int64_t b = 0;
    for (const auto& [name, t] : tape) {
      if (initializers_.count(name) == 0) b += t.byte_size();
    }
    return b;
  };
  {
    const std::string& op = node.op_type;
    if (profiler) profiler->BeginOp(op, node.name);

    // ---- Multi-output ops: handle and continue (skip the single-output handler). ----
    if (op == "Split") {
      const Tensor& x = get(node.inputs[0]);
      int64_t axis = node.GetAttrInt("axis", 0);
      if (axis < 0) axis += x.rank();
      std::vector<int64_t> sizes;
      if (node.inputs.size() >= 2) sizes = ReadInt64Vec(get(node.inputs[1]));
      else sizes = node.GetAttrInts("split");
      if (sizes.empty() && !node.outputs.empty()) {
        int64_t total = x.shape()[axis];
        int64_t k = static_cast<int64_t>(node.outputs.size());
        if (k > 0 && total % k == 0) sizes.assign(k, total / k);
        else throw std::runtime_error("Split: cannot infer split sizes");
      }
      auto parts = Split(x, axis, sizes);
      for (size_t i = 0; i < parts.size() && i < node.outputs.size(); ++i) {
        tape[node.outputs[i]] = std::move(parts[i]);
      }
      if (profiler) {
        profiler->EndOp(profiler->TrackActivationBytes() ? live_activation_bytes() : 0);
      }
      return;
    }

    if (op == "FusedQKV") {
      auto qkv = FusedQKVProjection(
          get(node.inputs[0]), get(node.inputs[1]), get(node.inputs[2]),
          get(node.inputs[3]), get(node.inputs[4]), get(node.inputs[5]),
          get(node.inputs[6]));
      for (size_t i = 0; i < qkv.size() && i < node.outputs.size(); ++i) {
        tape[node.outputs[i]] = std::move(qkv[i]);
      }
      if (profiler) {
        profiler->EndOp(profiler->TrackActivationBytes() ? live_activation_bytes() : 0);
      }
      return;
    }

    Tensor out;

    // Pointwise unary
    if (op == "Sqrt")        out = Sqrt(get(node.inputs[0]));
    else if (op == "Erf")    out = Erf(get(node.inputs[0]));
    else if (op == "Relu")   out = Relu(get(node.inputs[0]));
    else if (op == "Tanh")   out = Tanh(get(node.inputs[0]));
    else if (op == "Neg")    out = Neg(get(node.inputs[0]));
    else if (op == "Abs")    out = Abs(get(node.inputs[0]));
    else if (op == "Sin")    out = Sin(get(node.inputs[0]));
    else if (op == "Cos")    out = Cos(get(node.inputs[0]));
    else if (op == "Sigmoid") out = Sigmoid(get(node.inputs[0]));

    // Binary elementwise (broadcast)
    else if (op == "Add")    out = Add(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Sub")    out = Sub(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Mul")    out = Mul(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Div")    out = Div(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Pow")    out = Pow(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Equal")  out = Equal(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Less")   out = Less(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Greater") out = Greater(get(node.inputs[0]), get(node.inputs[1]));

    // Where (3-arg)
    else if (op == "Where")  out = Where(get(node.inputs[0]),
                                        get(node.inputs[1]),
                                        get(node.inputs[2]));

    // Activations
    else if (op == "Gelu")     out = Gelu(get(node.inputs[0]));
    else if (op == "GeluTanh") out = GeluTanh(get(node.inputs[0]));
    else if (op == "Softmax") {
      int64_t axis = node.GetAttrInt("axis", -1);
      out = Softmax(get(node.inputs[0]), axis);
    }
    else if (op == "ReduceMean") {
      auto axes = node.GetAttrInts("axes");
      // Opset 18+: axes can come from input[1].
      if (axes.empty() && node.inputs.size() >= 2) {
        axes = ReadInt64Vec(get(node.inputs[1]));
      }
      bool keepdims = node.GetAttrInt("keepdims", 1) != 0;
      out = ReduceMean(get(node.inputs[0]), axes, keepdims);
    }

    // Linear algebra
    else if (op == "MatMul") {
      out = MatMul(get(node.inputs[0]), get(node.inputs[1]));
    }
    else if (op == "FusedMatMulAddGELU") {
      out = FusedMatMulAddGELU(get(node.inputs[0]), get(node.inputs[1]),
                               get(node.inputs[2]));
    }
    else if (op == "FusedLayerNorm") {
      float eps = node.GetAttrFloat("epsilon", 1e-5f);
      out = LayerNorm(get(node.inputs[0]), get(node.inputs[1]),
                      get(node.inputs[2]), eps, /*normalized_dims=*/1);
    }
    else if (op == "FusedAttention") {
      int64_t head_dim = node.GetAttrInt("head_dim", 64);
      float fill = node.GetAttrFloat("fill", -3.0e38f);
      static const Tensor kNoMask;  // empty when input absent
      const Tensor& mask = node.inputs.size() >= 4 ? get(node.inputs[3]) : kNoMask;
      out = FusedAttention(get(node.inputs[0]), get(node.inputs[1]),
                           get(node.inputs[2]), mask, head_dim, fill);
    }
    else if (op == "Gemm") {
      float alpha = node.GetAttrFloat("alpha", 1.0f);
      float beta = node.GetAttrFloat("beta", 1.0f);
      bool tA = node.GetAttrInt("transA", 0) != 0;
      bool tB = node.GetAttrInt("transB", 0) != 0;
      const Tensor* c = node.inputs.size() >= 3 ? &get(node.inputs[2]) : nullptr;
      out = Gemm(get(node.inputs[0]), get(node.inputs[1]), c, alpha, beta, tA, tB);
    }

    // Movement / shape ops
    else if (op == "Reshape") {
      auto shape = ReadInt64Vec(get(node.inputs[1]));
      Shape s(shape.begin(), shape.end());
      out = Reshape(get(node.inputs[0]), s);
    }
    else if (op == "Transpose") {
      auto perm = node.GetAttrInts("perm");
      out = Transpose(get(node.inputs[0]), perm);
    }
    else if (op == "Concat") {
      int64_t axis = node.GetAttrInt("axis", 0);
      std::vector<Tensor> parts;
      parts.reserve(node.inputs.size());
      for (const auto& iname : node.inputs) parts.push_back(get(iname));
      out = Concat(parts, axis);
    }
    else if (op == "Slice") {
      // Opset >= 10: starts/ends/axes/steps as inputs.
      std::vector<int64_t> starts, ends, axes, steps;
      if (node.inputs.size() >= 2) starts = ReadInt64Vec(get(node.inputs[1]));
      if (node.inputs.size() >= 3) ends = ReadInt64Vec(get(node.inputs[2]));
      if (node.inputs.size() >= 4) axes = ReadInt64Vec(get(node.inputs[3]));
      if (node.inputs.size() >= 5) steps = ReadInt64Vec(get(node.inputs[4]));
      out = Slice(get(node.inputs[0]), starts, ends, axes, steps);
    }
    else if (op == "Unsqueeze") {
      // Opset 13+: axes from input[1]; older: attribute.
      std::vector<int64_t> axes;
      if (node.inputs.size() >= 2) axes = ReadInt64Vec(get(node.inputs[1]));
      else axes = node.GetAttrInts("axes");
      out = Unsqueeze(get(node.inputs[0]), axes);
    }
    else if (op == "Squeeze") {
      std::vector<int64_t> axes;
      if (node.inputs.size() >= 2) axes = ReadInt64Vec(get(node.inputs[1]));
      else axes = node.GetAttrInts("axes");
      out = Squeeze(get(node.inputs[0]), axes);
    }
    else if (op == "Cast") {
      int64_t to = node.GetAttrInt("to", onnx::TensorProto::FLOAT);
      out = Cast(get(node.inputs[0]),
                 DTypeFromOnnx(static_cast<int32_t>(to)));
    }
    else if (op == "Expand") {
      auto target = ReadInt64Vec(get(node.inputs[1]));
      Shape s(target.begin(), target.end());
      out = Expand(get(node.inputs[0]), s);
    }
    else if (op == "Shape") {
      out = ShapeOf(get(node.inputs[0]));
    }
    else if (op == "ConstantOfShape") {
      // Output shape from input[0] (int64 vec); fill from `value` attribute.
      auto shape_vec = ReadInt64Vec(get(node.inputs[0]));
      Shape shape(shape_vec.begin(), shape_vec.end());
      DType dtype = DType::kFloat32;
      std::vector<uint8_t> fill_bytes;
      const auto* val = node.GetAttr("value");
      if (val && val->type() == onnx::AttributeProto::TENSOR) {
        dtype = DTypeFromOnnx(val->t().data_type());
        if (!val->t().raw_data().empty()) {
          fill_bytes.assign(val->t().raw_data().begin(), val->t().raw_data().end());
        } else {
          // Typed-data fallback for fp32 / int64.
          const auto& tp = val->t();
          int64_t eb = DTypeBytes(dtype);
          if (eb > 0) fill_bytes.resize(static_cast<size_t>(eb));
          if (tp.data_type() == onnx::TensorProto::FLOAT && tp.float_data_size() >= 1) {
            float v = tp.float_data(0);
            std::memcpy(fill_bytes.data(), &v, sizeof(float));
          } else if (tp.data_type() == onnx::TensorProto::INT64 && tp.int64_data_size() >= 1) {
            int64_t v = tp.int64_data(0);
            std::memcpy(fill_bytes.data(), &v, sizeof(int64_t));
          }
        }
      }
      out = ConstantOfShape(shape, dtype, fill_bytes);
    }
    else if (op == "Range") {
      const Tensor& s = get(node.inputs[0]);
      const Tensor& l = get(node.inputs[1]);
      const Tensor& d = get(node.inputs[2]);
      if (s.dtype() == DType::kInt64) {
        out = RangeI64(s.Contiguous().data<int64_t>()[0],
                       l.Contiguous().data<int64_t>()[0],
                       d.Contiguous().data<int64_t>()[0]);
      } else if (s.dtype() == DType::kFloat32) {
        out = RangeF32(s.Contiguous().data<float>()[0],
                       l.Contiguous().data<float>()[0],
                       d.Contiguous().data<float>()[0]);
      } else {
        throw std::runtime_error("Range: unsupported dtype");
      }
    }
    else if (op == "Gather") {
      int64_t axis = node.GetAttrInt("axis", 0);
      out = Gather(get(node.inputs[0]), get(node.inputs[1]), axis);
    }
    else if (op == "Constant") {
      out = MaterializeConstant(node);
    }
    else if (op == "Identity") {
      out = get(node.inputs[0]);
    }
    else if (op == "Trilu") {
      int64_t k = 0;
      if (node.inputs.size() >= 2 && !node.inputs[1].empty())
        k = get(node.inputs[1]).Contiguous().data<int64_t>()[0];
      out = Trilu(get(node.inputs[0]), k, node.GetAttrInt("upper", 1) != 0);
    }
    else if (op == "ScatterND") {
      out = ScatterND(get(node.inputs[0]), get(node.inputs[1]), get(node.inputs[2]));
    }
    else if (op == "If") {
      const Tensor& cond = get(node.inputs[0]);
      if (cond.dtype() != DType::kBool || cond.numel() != 1)
        throw std::runtime_error("If: condition must be a bool scalar");
      const bool take_then = cond.Contiguous().data<uint8_t>()[0] != 0;
      const auto* attr = node.GetAttr(take_then ? "then_branch" : "else_branch");
      auto it = attr ? branches_.find(attr) : branches_.end();
      if (it == branches_.end()) throw std::runtime_error("If: missing branch graph");
      const Graph& br = *it->second;
      // Branch initializers become visible on the (shared) tape; outer-scope
      // tensors are already there. Branch nodes run without the profiler so
      // op records stay flat (the If itself is the recorded op).
      for (const auto& [tname, t] : br.tensors)
        if (t.IsInitializer() && !tape.count(tname)) tape[tname] = MakeFromIR(t);
      RunNodes(br, tape, nullptr);
      for (size_t i = 0; i < node.outputs.size() && i < br.outputs.size(); ++i)
        tape[node.outputs[i]] = get(br.outputs[i]);
      if (profiler) {
        profiler->EndOp(profiler->TrackActivationBytes() ? live_activation_bytes() : 0);
      }
      return;
    }
    else {
      throw std::runtime_error(
          "Executor: unsupported op '" + op + "' (node: " + node.name + ")");
    }

    // For nodes with one output, store under the (single) output name.
    if (!node.outputs.empty()) {
      tape[node.outputs[0]] = std::move(out);
    }
    if (profiler) {
      profiler->EndOp(profiler->TrackActivationBytes() ? live_activation_bytes() : 0);
    }
  }
}

std::map<std::string, Tensor> Executor::Run(
    const std::map<std::string, Tensor>& inputs,
    prof::Profiler* profiler) const {
  Tape tape = initializers_;
  for (const auto& [k, v] : inputs) tape[k] = v;

  if (profiler) profiler->BeginIteration();
  RunNodes(*graph_, tape, profiler);
  if (profiler) profiler->EndIteration();


  // Collect graph outputs.
  std::map<std::string, Tensor> result;
  for (const auto& name : graph_->outputs) {
    auto it = tape.find(name);
    if (it == tape.end()) {
      throw std::runtime_error("Executor: missing graph output '" + name + "'");
    }
    result[name] = it->second;
  }
  return result;
}

}  // namespace rt
}  // namespace inferc
