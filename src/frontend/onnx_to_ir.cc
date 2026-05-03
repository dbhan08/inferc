#include "frontend/onnx_to_ir.h"

#include <cstring>
#include <sstream>

namespace inferc {

namespace {

void ShapeFromValueInfo(const onnx::ValueInfoProto& vi, Shape* out) {
  if (!vi.has_type() || !vi.type().has_tensor_type()) return;
  const auto& tt = vi.type().tensor_type();
  if (!tt.has_shape()) return;
  out->reserve(tt.shape().dim_size());
  for (const auto& dim : tt.shape().dim()) {
    if (dim.has_dim_value()) {
      out->push_back(dim.dim_value());
    } else {
      // symbolic dim → unknown (-1) for now
      out->push_back(kUnknownDim);
    }
  }
}

DType DTypeFromValueInfo(const onnx::ValueInfoProto& vi) {
  if (!vi.has_type() || !vi.type().has_tensor_type()) return DType::kUnknown;
  return DTypeFromOnnx(vi.type().tensor_type().elem_type());
}

// Materialize an initializer's bytes from the various places ONNX may stash
// them: raw_data (preferred), or the typed *_data fields. We only handle the
// common cases for v1.
void CopyInitializerBytes(const onnx::TensorProto& src, Tensor* dst) {
  if (!src.raw_data().empty()) {
    const std::string& bytes = src.raw_data();
    dst->raw_data.assign(bytes.begin(), bytes.end());
    return;
  }
  // Typed-data path. For each dtype that ONNX serializes via repeated fields
  // rather than raw_data, copy out as a contiguous byte buffer in the natural
  // little-endian layout (Apple Silicon is LE; protobuf wire is LE; matches).
  const int64_t n_elems = [&]() {
    int64_t n = 1;
    for (auto d : src.dims()) n *= d;
    return src.dims_size() == 0 ? 1 : n;
  }();
  switch (src.data_type()) {
    case onnx::TensorProto::FLOAT: {
      dst->raw_data.resize(n_elems * sizeof(float));
      auto* p = reinterpret_cast<float*>(dst->raw_data.data());
      for (int i = 0; i < src.float_data_size(); ++i) p[i] = src.float_data(i);
      break;
    }
    case onnx::TensorProto::INT32: {
      dst->raw_data.resize(n_elems * sizeof(int32_t));
      auto* p = reinterpret_cast<int32_t*>(dst->raw_data.data());
      for (int i = 0; i < src.int32_data_size(); ++i) p[i] = src.int32_data(i);
      break;
    }
    case onnx::TensorProto::INT64: {
      dst->raw_data.resize(n_elems * sizeof(int64_t));
      auto* p = reinterpret_cast<int64_t*>(dst->raw_data.data());
      for (int i = 0; i < src.int64_data_size(); ++i) p[i] = src.int64_data(i);
      break;
    }
    case onnx::TensorProto::DOUBLE: {
      dst->raw_data.resize(n_elems * sizeof(double));
      auto* p = reinterpret_cast<double*>(dst->raw_data.data());
      for (int i = 0; i < src.double_data_size(); ++i) p[i] = src.double_data(i);
      break;
    }
    default:
      // Other typed-data dtypes (UINT64, FLOAT16, etc.) are uncommon for v1.
      // Leave raw_data empty; the shape inference / executor will surface this.
      break;
  }
}

}  // namespace

bool ConvertOnnxToIR(const onnx::ModelProto& model, Graph* out, std::string* err) {
  if (!model.has_graph()) {
    if (err) *err = "model has no graph";
    return false;
  }
  const auto& g = model.graph();
  out->name = g.name();

  // 1) Initializers → Tensors with raw_data.
  for (const auto& init : g.initializer()) {
    Tensor& t = out->GetOrCreateTensor(init.name());
    t.dtype = DTypeFromOnnx(init.data_type());
    t.shape.assign(init.dims().begin(), init.dims().end());
    CopyInitializerBytes(init, &t);
  }

  // 2) Graph inputs → Tensors (skip those that are also initializers; some
  //    older exports list both). value_info inputs supply dtype + symbolic shape.
  for (const auto& vi : g.input()) {
    Tensor* existing = out->GetTensor(vi.name());
    if (existing && existing->IsInitializer()) continue;  // weight, not a runtime input
    Tensor& t = out->GetOrCreateTensor(vi.name());
    t.dtype = DTypeFromValueInfo(vi);
    ShapeFromValueInfo(vi, &t.shape);
    out->inputs.push_back(vi.name());
  }

  // 3) Graph outputs.
  for (const auto& vi : g.output()) {
    Tensor& t = out->GetOrCreateTensor(vi.name());
    t.dtype = DTypeFromValueInfo(vi);
    ShapeFromValueInfo(vi, &t.shape);
    out->outputs.push_back(vi.name());
  }

  // 4) value_info → set dtype/shape on intermediates if available.
  for (const auto& vi : g.value_info()) {
    Tensor& t = out->GetOrCreateTensor(vi.name());
    t.dtype = DTypeFromValueInfo(vi);
    if (t.shape.empty()) {
      ShapeFromValueInfo(vi, &t.shape);
    }
  }

  // 5) Nodes (already in topological order in ONNX). Create placeholders for
  //    every output tensor so they exist in the table for the shape inferencer.
  out->nodes.reserve(g.node_size());
  for (const auto& src : g.node()) {
    Node n;
    n.name = src.name();
    n.op_type = src.op_type();
    n.domain = src.domain();
    n.inputs.assign(src.input().begin(), src.input().end());
    n.outputs.assign(src.output().begin(), src.output().end());
    n.attributes.assign(src.attribute().begin(), src.attribute().end());

    for (const auto& out_name : src.output()) {
      out->GetOrCreateTensor(out_name);
    }
    out->nodes.push_back(std::move(n));
  }

  return true;
}

}  // namespace inferc
