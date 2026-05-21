#include "runtime/executor.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "kernels/activation.h"
#include "kernels/elementwise.h"
#include "kernels/embedding.h"
#include "kernels/matmul.h"
#include "kernels/movement.h"
#include "profiler/profiler.h"

namespace inferc {
namespace rt {

namespace {

// Build a runtime Tensor from an IR Tensor that carries initializer bytes.
Tensor MakeFromIR(const ::inferc::Tensor& t) {
  return Tensor::FromHostBytes(t.dtype, t.shape, t.raw_data.data());
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
}

std::map<std::string, Tensor> Executor::Run(
    const std::map<std::string, Tensor>& inputs,
    prof::Profiler* profiler) const {
  std::map<std::string, Tensor> tape = initializers_;
  for (const auto& [k, v] : inputs) tape[k] = v;

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

  if (profiler) profiler->BeginIteration();

  for (const auto& node : graph_->nodes) {
    const std::string& op = node.op_type;
    Tensor out;
    if (profiler) profiler->BeginOp(op, node.name);

    // Pointwise unary
    if (op == "Sqrt")        out = Sqrt(get(node.inputs[0]));
    else if (op == "Erf")    out = Erf(get(node.inputs[0]));
    else if (op == "Relu")   out = Relu(get(node.inputs[0]));
    else if (op == "Tanh")   out = Tanh(get(node.inputs[0]));
    else if (op == "Neg")    out = Neg(get(node.inputs[0]));
    else if (op == "Abs")    out = Abs(get(node.inputs[0]));

    // Binary elementwise (broadcast)
    else if (op == "Add")    out = Add(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Sub")    out = Sub(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Mul")    out = Mul(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Div")    out = Div(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Pow")    out = Pow(get(node.inputs[0]), get(node.inputs[1]));
    else if (op == "Equal")  out = Equal(get(node.inputs[0]), get(node.inputs[1]));

    // Where (3-arg)
    else if (op == "Where")  out = Where(get(node.inputs[0]),
                                        get(node.inputs[1]),
                                        get(node.inputs[2]));

    // Activations
    else if (op == "Gelu")   out = Gelu(get(node.inputs[0]));
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
    else {
      throw std::runtime_error(
          "Executor: unsupported op '" + op + "' (node: " + node.name + ")");
    }

    // For nodes with one output, store under the (single) output name.
    if (!node.outputs.empty()) {
      tape[node.outputs[0]] = std::move(out);
    }
    if (profiler) profiler->EndOp(live_activation_bytes());
  }

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
