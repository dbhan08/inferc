#include "ir/shape_inference.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>

namespace inferc {

bool BroadcastShapes(const Shape& a, const Shape& b, Shape* out) {
  out->clear();
  const size_t ra = a.size(), rb = b.size();
  const size_t r = std::max(ra, rb);
  out->resize(r);
  for (size_t i = 0; i < r; ++i) {
    int64_t da = i < ra ? a[ra - 1 - i] : 1;
    int64_t db = i < rb ? b[rb - 1 - i] : 1;
    int64_t d;
    if (da == kUnknownDim || db == kUnknownDim) {
      d = kUnknownDim;
    } else if (da == db) {
      d = da;
    } else if (da == 1) {
      d = db;
    } else if (db == 1) {
      d = da;
    } else {
      return false;  // hard mismatch
    }
    (*out)[r - 1 - i] = d;
  }
  return true;
}

namespace {

// Read a 1D int64 tensor's bytes into a vector<int64_t>. Returns false if
// the tensor isn't a resolved int64 1D initializer.
bool ReadInt64Tensor(const Tensor& t, std::vector<int64_t>* out) {
  if (t.dtype != DType::kInt64) return false;
  if (t.raw_data.empty()) return false;
  if (t.raw_data.size() % sizeof(int64_t) != 0) return false;
  out->resize(t.raw_data.size() / sizeof(int64_t));
  std::memcpy(out->data(), t.raw_data.data(), t.raw_data.size());
  return true;
}

// Try to obtain a 1D int64 vector from a tensor that may have come from a
// Constant node (already resolved into `tensors`) or a graph initializer.
bool TryReadConstInt64(const Graph& g, const std::string& name,
                       std::vector<int64_t>* out) {
  const Tensor* t = g.GetTensor(name);
  if (!t) return false;
  return ReadInt64Tensor(*t, out);
}

// Output shape == input[0] shape, output dtype == input[0] dtype.
void PassThroughShape(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* in = g.GetTensor(n.inputs[0]);
  if (!in) return;
  out->shape = in->shape;
  out->dtype = in->dtype;
}

// Standard numpy-style broadcast across all inputs. Output dtype follows
// input[0] (or input[1] for Where, where input[1] is `x`).
void BroadcastAll(const Graph& g, const Node& n, Tensor* out, size_t dtype_idx = 0) {
  Shape acc;
  bool first = true;
  for (const auto& iname : n.inputs) {
    const Tensor* t = g.GetTensor(iname);
    if (!t) return;
    if (first) { acc = t->shape; first = false; continue; }
    Shape merged;
    if (!BroadcastShapes(acc, t->shape, &merged)) return;
    acc = merged;
  }
  out->shape = acc;
  if (n.inputs.size() > dtype_idx) {
    const Tensor* t = g.GetTensor(n.inputs[dtype_idx]);
    if (t) out->dtype = t->dtype;
  }
}

// ---------------- Per-op rules ----------------

void Op_MatMul(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.size() < 2) return;
  const Tensor* a = g.GetTensor(n.inputs[0]);
  const Tensor* b = g.GetTensor(n.inputs[1]);
  if (!a || !b) return;
  if (a->shape.size() < 1 || b->shape.size() < 1) return;
  // Special cases: 1D x 1D → scalar, 1D x ND, ND x 1D, ND x ND with batch broadcast.
  Shape sa = a->shape, sb = b->shape;
  // Promote 1D to 2D for the algorithm; remember to squeeze at the end.
  const bool a_is_1d = (sa.size() == 1);
  const bool b_is_1d = (sb.size() == 1);
  if (a_is_1d) sa.insert(sa.begin(), 1);
  if (b_is_1d) sb.push_back(1);
  // Now both are >= 2D. Last two dims do the matmul; leading dims broadcast.
  const int64_t M = sa[sa.size() - 2];
  const int64_t Kb = sa[sa.size() - 1];  // K from a (unused for shape, just sanity)
  (void)Kb;
  const int64_t N = sb[sb.size() - 1];
  Shape lead_a(sa.begin(), sa.end() - 2);
  Shape lead_b(sb.begin(), sb.end() - 2);
  Shape lead;
  if (!BroadcastShapes(lead_a, lead_b, &lead)) return;
  lead.push_back(M);
  lead.push_back(N);
  if (a_is_1d) lead.erase(lead.end() - 2);
  if (b_is_1d) lead.pop_back();
  out->shape = std::move(lead);
  out->dtype = a->dtype;
}

void Op_Gemm(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.size() < 2) return;
  const Tensor* a = g.GetTensor(n.inputs[0]);
  const Tensor* b = g.GetTensor(n.inputs[1]);
  if (!a || !b || a->shape.size() != 2 || b->shape.size() != 2) return;
  const bool transA = n.GetAttrInt("transA", 0) != 0;
  const bool transB = n.GetAttrInt("transB", 0) != 0;
  const int64_t M = transA ? a->shape[1] : a->shape[0];
  const int64_t N = transB ? b->shape[0] : b->shape[1];
  out->shape = {M, N};
  out->dtype = a->dtype;
}

void Op_Reshape(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.size() < 2) return;
  const Tensor* data = g.GetTensor(n.inputs[0]);
  if (!data) return;
  std::vector<int64_t> target;
  if (!TryReadConstInt64(g, n.inputs[1], &target)) {
    // Can't resolve target shape; leave unknown.
    out->dtype = data->dtype;
    return;
  }
  // Resolve special values: 0 = same as input dim at that position, -1 = inferred.
  Shape new_shape(target.begin(), target.end());
  // Replace 0 with input dim
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == 0 && i < data->shape.size()) {
      new_shape[i] = data->shape[i];
    }
  }
  // Replace -1 with the dim that makes sizes match, only if everything else is concrete.
  int neg_idx = -1;
  int64_t known_product = 1;
  bool concrete = true;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      if (neg_idx >= 0) { concrete = false; break; }
      neg_idx = static_cast<int>(i);
    } else if (new_shape[i] < 0) {
      concrete = false;
      break;
    } else {
      known_product *= new_shape[i];
    }
  }
  if (neg_idx >= 0 && concrete && data->NumElements() > 0 && known_product > 0) {
    new_shape[neg_idx] = data->NumElements() / known_product;
  }
  out->shape = new_shape;
  out->dtype = data->dtype;
}

void Op_Transpose(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* in = g.GetTensor(n.inputs[0]);
  if (!in) return;
  std::vector<int64_t> perm = n.GetAttrInts("perm");
  if (perm.empty()) {
    // Default: reverse all dims.
    Shape s = in->shape;
    std::reverse(s.begin(), s.end());
    out->shape = std::move(s);
  } else {
    Shape s(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      const auto p = perm[i];
      if (p < 0 || p >= static_cast<int64_t>(in->shape.size())) return;
      s[i] = in->shape[p];
    }
    out->shape = std::move(s);
  }
  out->dtype = in->dtype;
}

void Op_Concat(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* first = g.GetTensor(n.inputs[0]);
  if (!first) return;
  const int64_t axis_attr = n.GetAttrInt("axis", 0);
  const int64_t rank = static_cast<int64_t>(first->shape.size());
  const int64_t axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
  if (axis < 0 || axis >= rank) return;
  Shape s = first->shape;
  for (size_t i = 1; i < n.inputs.size(); ++i) {
    const Tensor* t = g.GetTensor(n.inputs[i]);
    if (!t || static_cast<int64_t>(t->shape.size()) != rank) return;
    if (s[axis] == kUnknownDim || t->shape[axis] == kUnknownDim) {
      s[axis] = kUnknownDim;
    } else {
      s[axis] += t->shape[axis];
    }
  }
  out->shape = std::move(s);
  out->dtype = first->dtype;
}

void Op_Unsqueeze(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* data = g.GetTensor(n.inputs[0]);
  if (!data) return;
  // Opset 13+: axes is input[1]. Earlier: attribute.
  std::vector<int64_t> axes;
  if (n.inputs.size() >= 2) {
    TryReadConstInt64(g, n.inputs[1], &axes);
  }
  if (axes.empty()) axes = n.GetAttrInts("axes");
  if (axes.empty()) return;
  const int64_t out_rank = static_cast<int64_t>(data->shape.size()) + axes.size();
  for (auto& a : axes) if (a < 0) a += out_rank;
  std::sort(axes.begin(), axes.end());
  Shape s;
  s.reserve(out_rank);
  size_t in_idx = 0;
  size_t ax_idx = 0;
  for (int64_t i = 0; i < out_rank; ++i) {
    if (ax_idx < axes.size() && axes[ax_idx] == i) {
      s.push_back(1);
      ++ax_idx;
    } else if (in_idx < data->shape.size()) {
      s.push_back(data->shape[in_idx++]);
    }
  }
  out->shape = std::move(s);
  out->dtype = data->dtype;
}

void Op_Squeeze(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* data = g.GetTensor(n.inputs[0]);
  if (!data) return;
  std::vector<int64_t> axes;
  if (n.inputs.size() >= 2) TryReadConstInt64(g, n.inputs[1], &axes);
  if (axes.empty()) axes = n.GetAttrInts("axes");
  Shape s;
  if (axes.empty()) {
    // Remove all size-1 dims.
    for (auto d : data->shape) if (d != 1) s.push_back(d);
  } else {
    const int64_t r = static_cast<int64_t>(data->shape.size());
    std::set<int64_t> to_drop;
    for (auto a : axes) to_drop.insert(a < 0 ? a + r : a);
    for (int64_t i = 0; i < r; ++i) {
      if (!to_drop.count(i)) s.push_back(data->shape[i]);
    }
  }
  out->shape = std::move(s);
  out->dtype = data->dtype;
}

void Op_Gather(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.size() < 2) return;
  const Tensor* data = g.GetTensor(n.inputs[0]);
  const Tensor* indices = g.GetTensor(n.inputs[1]);
  if (!data || !indices) return;
  const int64_t axis_attr = n.GetAttrInt("axis", 0);
  const int64_t rank = static_cast<int64_t>(data->shape.size());
  const int64_t axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
  if (axis < 0 || axis >= rank) return;
  Shape s;
  s.reserve(data->shape.size() - 1 + indices->shape.size());
  for (int64_t i = 0; i < axis; ++i) s.push_back(data->shape[i]);
  for (auto d : indices->shape) s.push_back(d);
  for (int64_t i = axis + 1; i < rank; ++i) s.push_back(data->shape[i]);
  out->shape = std::move(s);
  out->dtype = data->dtype;
}

void Op_ReduceMean(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* in = g.GetTensor(n.inputs[0]);
  if (!in) return;
  const bool keepdims = n.GetAttrInt("keepdims", 1) != 0;
  std::vector<int64_t> axes = n.GetAttrInts("axes");
  // Opset 18+: axes can be input[1].
  if (axes.empty() && n.inputs.size() >= 2) {
    TryReadConstInt64(g, n.inputs[1], &axes);
  }
  const int64_t r = static_cast<int64_t>(in->shape.size());
  std::set<int64_t> ax_set;
  if (axes.empty()) {
    // Default: reduce over all axes.
    for (int64_t i = 0; i < r; ++i) ax_set.insert(i);
  } else {
    for (auto a : axes) ax_set.insert(a < 0 ? a + r : a);
  }
  Shape s;
  for (int64_t i = 0; i < r; ++i) {
    if (ax_set.count(i)) {
      if (keepdims) s.push_back(1);
    } else {
      s.push_back(in->shape[i]);
    }
  }
  out->shape = std::move(s);
  out->dtype = in->dtype;
}

void Op_Slice(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* data = g.GetTensor(n.inputs[0]);
  if (!data) return;
  out->dtype = data->dtype;
  // Opset >= 10: starts, ends, axes, steps are inputs[1..4].
  std::vector<int64_t> starts, ends, axes, steps;
  if (n.inputs.size() >= 2) TryReadConstInt64(g, n.inputs[1], &starts);
  if (n.inputs.size() >= 3) TryReadConstInt64(g, n.inputs[2], &ends);
  if (n.inputs.size() >= 4) TryReadConstInt64(g, n.inputs[3], &axes);
  if (n.inputs.size() >= 5) TryReadConstInt64(g, n.inputs[4], &steps);
  if (starts.empty() || ends.empty()) {
    // Can't read slice bounds (they come from runtime-computed tensors).
    // Slice preserves rank — every dim of output corresponds 1:1 to input.
    // Mark all dims unknown so downstream ops at least know the rank.
    out->shape.assign(data->shape.size(), kUnknownDim);
    return;
  }
  const int64_t r = static_cast<int64_t>(data->shape.size());
  if (axes.empty()) {
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) axes[i] = static_cast<int64_t>(i);
  }
  Shape s = data->shape;
  for (size_t i = 0; i < starts.size(); ++i) {
    int64_t ax = axes[i] < 0 ? axes[i] + r : axes[i];
    if (ax < 0 || ax >= r) return;
    int64_t step = i < steps.size() ? steps[i] : 1;
    if (step == 0) return;
    int64_t dim = data->shape[ax];
    if (dim == kUnknownDim) {
      s[ax] = kUnknownDim;
      continue;
    }
    int64_t start = starts[i];
    int64_t end = ends[i];
    // Clamp per ONNX spec.
    if (start < 0) start += dim;
    if (end < 0) end += dim;
    start = std::max<int64_t>(0, std::min(start, dim));
    end = std::max<int64_t>(0, std::min(end, dim));
    int64_t len = step > 0 ? std::max<int64_t>(0, (end - start + step - 1) / step)
                           : std::max<int64_t>(0, (start - end - step - 1) / -step);
    s[ax] = len;
  }
  out->shape = std::move(s);
}

void Op_ConstantOfShape(const Graph& g, const Node& n, Tensor* out) {
  // Output dtype comes from the optional `value` attribute (default fp32).
  out->dtype = DType::kFloat32;
  const auto* val = n.GetAttr("value");
  if (val && val->type() == onnx::AttributeProto::TENSOR) {
    out->dtype = DTypeFromOnnx(val->t().data_type());
  }
  // Output shape comes from input[0] read as int64 vector.
  if (n.inputs.empty()) return;
  std::vector<int64_t> shape_vec;
  if (TryReadConstInt64(g, n.inputs[0], &shape_vec)) {
    out->shape.assign(shape_vec.begin(), shape_vec.end());
  } else {
    // Shape not yet resolvable. The rank equals the length of input[0], which
    // is at most a known 1D shape; mark all dims as unknown.
    const Tensor* in0 = g.GetTensor(n.inputs[0]);
    if (in0 && in0->shape.size() == 1 && in0->shape[0] >= 0) {
      out->shape.assign(in0->shape[0], kUnknownDim);
    }
  }
}

void Op_Range(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.size() < 3) return;
  const Tensor* in0 = g.GetTensor(n.inputs[0]);
  if (!in0) return;
  out->dtype = in0->dtype;
  // Try to read the three scalar inputs. If we can, compute the exact length;
  // else mark as 1D unknown.
  std::vector<int64_t> s, l, d;
  if (in0->dtype == DType::kInt64 &&
      TryReadConstInt64(g, n.inputs[0], &s) &&
      TryReadConstInt64(g, n.inputs[1], &l) &&
      TryReadConstInt64(g, n.inputs[2], &d) &&
      s.size() == 1 && l.size() == 1 && d.size() == 1 && d[0] != 0) {
    int64_t n_out;
    if (d[0] > 0) n_out = (l[0] > s[0]) ? (l[0] - s[0] + d[0] - 1) / d[0] : 0;
    else          n_out = (s[0] > l[0]) ? (s[0] - l[0] + (-d[0]) - 1) / (-d[0]) : 0;
    out->shape = {n_out};
  } else {
    out->shape = {kUnknownDim};
  }
}

void Op_Split(Graph* g, const Node& n) {
  if (n.inputs.empty()) return;
  const Tensor* x = g->GetTensor(n.inputs[0]);
  if (!x) return;
  int64_t axis = n.GetAttrInt("axis", 0);
  const int64_t r = static_cast<int64_t>(x->shape.size());
  if (axis < 0) axis += r;
  if (axis < 0 || axis >= r) return;

  std::vector<int64_t> sizes;
  if (n.inputs.size() >= 2) TryReadConstInt64(*g, n.inputs[1], &sizes);
  if (sizes.empty()) {
    auto attr = n.GetAttrInts("split");
    sizes.assign(attr.begin(), attr.end());
  }
  // If still empty, split evenly into n.outputs.size() parts.
  if (sizes.empty() && !n.outputs.empty()) {
    int64_t total = x->shape[axis];
    int64_t k = static_cast<int64_t>(n.outputs.size());
    if (total != kUnknownDim && k > 0 && total % k == 0) {
      sizes.assign(k, total / k);
    }
  }
  if (sizes.size() != n.outputs.size()) return;
  for (size_t i = 0; i < n.outputs.size(); ++i) {
    Tensor* out_t = g->GetTensor(n.outputs[i]);
    if (!out_t) continue;
    out_t->dtype = x->dtype;
    Shape s = x->shape;
    s[axis] = sizes[i];
    out_t->shape = std::move(s);
  }
}

void Op_Shape(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* in = g.GetTensor(n.inputs[0]);
  if (!in) return;
  out->shape = {static_cast<int64_t>(in->shape.size())};
  out->dtype = DType::kInt64;
  // If shape is fully resolved, materialize the Shape op as a constant so
  // downstream ops that need integer shape data can read it.
  if (ShapeIsResolved(in->shape)) {
    out->raw_data.resize(in->shape.size() * sizeof(int64_t));
    std::memcpy(out->raw_data.data(), in->shape.data(), out->raw_data.size());
  }
}

void Op_Constant(const Node& n, Tensor* out) {
  const auto* val = n.GetAttr("value");
  if (val && val->type() == onnx::AttributeProto::TENSOR) {
    const auto& t = val->t();
    out->dtype = DTypeFromOnnx(t.data_type());
    out->shape.assign(t.dims().begin(), t.dims().end());
    if (!t.raw_data().empty()) {
      const std::string& bytes = t.raw_data();
      out->raw_data.assign(bytes.begin(), bytes.end());
    } else {
      // Typed-data path for common types.
      const int64_t n_elems = out->NumElements();
      if (n_elems > 0) {
        switch (t.data_type()) {
          case onnx::TensorProto::FLOAT:
            out->raw_data.resize(n_elems * sizeof(float));
            for (int i = 0; i < t.float_data_size(); ++i) {
              std::memcpy(out->raw_data.data() + i * sizeof(float),
                          &t.float_data().data()[i], sizeof(float));
            }
            break;
          case onnx::TensorProto::INT64:
            out->raw_data.resize(n_elems * sizeof(int64_t));
            for (int i = 0; i < t.int64_data_size(); ++i) {
              const int64_t v = t.int64_data(i);
              std::memcpy(out->raw_data.data() + i * sizeof(int64_t), &v, sizeof(int64_t));
            }
            break;
          default: break;
        }
      }
    }
    return;
  }
  // Other Constant attribute forms: value_int, value_ints, value_float, value_floats.
  if ((val = n.GetAttr("value_int"))) {
    int64_t v = val->i();
    out->dtype = DType::kInt64;
    out->shape = {};
    out->raw_data.resize(sizeof(int64_t));
    std::memcpy(out->raw_data.data(), &v, sizeof(int64_t));
  } else if ((val = n.GetAttr("value_ints"))) {
    out->dtype = DType::kInt64;
    out->shape = {static_cast<int64_t>(val->ints_size())};
    out->raw_data.resize(val->ints_size() * sizeof(int64_t));
    for (int i = 0; i < val->ints_size(); ++i) {
      int64_t v = val->ints(i);
      std::memcpy(out->raw_data.data() + i * sizeof(int64_t), &v, sizeof(int64_t));
    }
  } else if ((val = n.GetAttr("value_float"))) {
    float v = val->f();
    out->dtype = DType::kFloat32;
    out->shape = {};
    out->raw_data.resize(sizeof(float));
    std::memcpy(out->raw_data.data(), &v, sizeof(float));
  } else if ((val = n.GetAttr("value_floats"))) {
    out->dtype = DType::kFloat32;
    out->shape = {static_cast<int64_t>(val->floats_size())};
    out->raw_data.resize(val->floats_size() * sizeof(float));
    for (int i = 0; i < val->floats_size(); ++i) {
      float v = val->floats(i);
      std::memcpy(out->raw_data.data() + i * sizeof(float), &v, sizeof(float));
    }
  }
}

void Op_Cast(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.empty()) return;
  const Tensor* in = g.GetTensor(n.inputs[0]);
  if (!in) return;
  out->shape = in->shape;
  const int64_t to_attr = n.GetAttrInt("to", onnx::TensorProto::FLOAT);
  out->dtype = DTypeFromOnnx(static_cast<int32_t>(to_attr));
}

void Op_Expand(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.size() < 2) return;
  const Tensor* in = g.GetTensor(n.inputs[0]);
  if (!in) return;
  out->dtype = in->dtype;
  std::vector<int64_t> target;
  if (!TryReadConstInt64(g, n.inputs[1], &target)) {
    // shape data unresolved → can still infer rank from input[1] shape.
    const Tensor* shp = g.GetTensor(n.inputs[1]);
    if (shp && shp->shape.size() == 1 && shp->shape[0] != kUnknownDim) {
      out->shape.assign(shp->shape[0], kUnknownDim);
    }
    return;
  }
  Shape result;
  if (!BroadcastShapes(in->shape, Shape(target.begin(), target.end()), &result)) return;
  out->shape = std::move(result);
}

void Op_Where(const Graph& g, const Node& n, Tensor* out) {
  if (n.inputs.size() < 3) return;
  const Tensor* cond = g.GetTensor(n.inputs[0]);
  const Tensor* x = g.GetTensor(n.inputs[1]);
  const Tensor* y = g.GetTensor(n.inputs[2]);
  if (!cond || !x || !y) return;
  Shape r1, r2;
  if (!BroadcastShapes(cond->shape, x->shape, &r1)) return;
  if (!BroadcastShapes(r1, y->shape, &r2)) return;
  out->shape = std::move(r2);
  out->dtype = x->dtype;
}

void Op_Equal(const Graph& g, const Node& n, Tensor* out) {
  // Boolean output, same broadcast as inputs.
  if (n.inputs.size() < 2) return;
  const Tensor* a = g.GetTensor(n.inputs[0]);
  const Tensor* b = g.GetTensor(n.inputs[1]);
  if (!a || !b) return;
  Shape r;
  if (!BroadcastShapes(a->shape, b->shape, &r)) return;
  out->shape = std::move(r);
  out->dtype = DType::kBool;
}

}  // namespace

bool InferShapes(Graph* graph, std::string* err,
                 std::vector<std::string>* unsupported_ops) {
  std::set<std::string> unsupported_set;

  for (auto& node : graph->nodes) {
    if (node.outputs.empty()) continue;
    Tensor* out0 = graph->GetTensor(node.outputs[0]);
    if (!out0) {
      if (err) *err = "missing output tensor for node " + node.name + " (" + node.op_type + ")";
      return false;
    }

    const std::string& op = node.op_type;

    // Pointwise (no broadcast, output shape = input[0] shape, dtype unchanged).
    if (op == "Erf" || op == "Sqrt" || op == "Relu" || op == "Softmax" ||
        op == "Tanh" || op == "Sigmoid" || op == "Identity" || op == "Gelu" ||
        op == "Neg" || op == "Abs" || op == "LogSoftmax" ||
        op == "FusedLayerNorm") {
      PassThroughShape(*graph, node, out0);
    }
    // Pointwise with broadcasting.
    else if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" ||
             op == "Pow" || op == "Min" || op == "Max") {
      BroadcastAll(*graph, node, out0);
    }
    else if (op == "Equal") { Op_Equal(*graph, node, out0); }
    else if (op == "Where") { Op_Where(*graph, node, out0); }
    else if (op == "MatMul" ||
             op == "FusedMatMulAddGELU") { Op_MatMul(*graph, node, out0); }
    else if (op == "Gemm") { Op_Gemm(*graph, node, out0); }
    else if (op == "Reshape") { Op_Reshape(*graph, node, out0); }
    else if (op == "Transpose") { Op_Transpose(*graph, node, out0); }
    else if (op == "Concat") { Op_Concat(*graph, node, out0); }
    else if (op == "Unsqueeze") { Op_Unsqueeze(*graph, node, out0); }
    else if (op == "Squeeze") { Op_Squeeze(*graph, node, out0); }
    else if (op == "Gather") { Op_Gather(*graph, node, out0); }
    else if (op == "ReduceMean" || op == "ReduceSum" || op == "ReduceMax" ||
             op == "ReduceMin") { Op_ReduceMean(*graph, node, out0); }
    else if (op == "Slice") { Op_Slice(*graph, node, out0); }
    else if (op == "Shape") { Op_Shape(*graph, node, out0); }
    else if (op == "Constant") { Op_Constant(node, out0); }
    else if (op == "Cast") { Op_Cast(*graph, node, out0); }
    else if (op == "Expand") { Op_Expand(*graph, node, out0); }
    else if (op == "ConstantOfShape") { Op_ConstantOfShape(*graph, node, out0); }
    else if (op == "Range") { Op_Range(*graph, node, out0); }
    else if (op == "Split") { Op_Split(graph, node); }   // multi-output
    else {
      unsupported_set.insert(op);
    }
  }

  if (unsupported_ops) {
    unsupported_ops->assign(unsupported_set.begin(), unsupported_set.end());
  }
  return true;
}

}  // namespace inferc
