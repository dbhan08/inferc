#include "ir/graph.h"

#include <sstream>

namespace inferc {

DType DTypeFromOnnx(int32_t onnx_dtype) {
  switch (onnx_dtype) {
    case onnx::TensorProto::FLOAT:    return DType::kFloat32;
    case onnx::TensorProto::FLOAT16:  return DType::kFloat16;
    case onnx::TensorProto::BFLOAT16: return DType::kBFloat16;
    case onnx::TensorProto::DOUBLE:   return DType::kFloat64;
    case onnx::TensorProto::INT8:     return DType::kInt8;
    case onnx::TensorProto::UINT8:    return DType::kUint8;
    case onnx::TensorProto::INT16:    return DType::kInt16;
    case onnx::TensorProto::UINT16:   return DType::kUint16;
    case onnx::TensorProto::INT32:    return DType::kInt32;
    case onnx::TensorProto::UINT32:   return DType::kUint32;
    case onnx::TensorProto::INT64:    return DType::kInt64;
    case onnx::TensorProto::UINT64:   return DType::kUint64;
    case onnx::TensorProto::BOOL:     return DType::kBool;
    default:                          return DType::kUnknown;
  }
}

int32_t DTypeToOnnx(DType dt) {
  switch (dt) {
    case DType::kFloat32:  return onnx::TensorProto::FLOAT;
    case DType::kFloat16:  return onnx::TensorProto::FLOAT16;
    case DType::kBFloat16: return onnx::TensorProto::BFLOAT16;
    case DType::kFloat64:  return onnx::TensorProto::DOUBLE;
    case DType::kInt8:     return onnx::TensorProto::INT8;
    case DType::kUint8:    return onnx::TensorProto::UINT8;
    case DType::kInt16:    return onnx::TensorProto::INT16;
    case DType::kUint16:   return onnx::TensorProto::UINT16;
    case DType::kInt32:    return onnx::TensorProto::INT32;
    case DType::kUint32:   return onnx::TensorProto::UINT32;
    case DType::kInt64:    return onnx::TensorProto::INT64;
    case DType::kUint64:   return onnx::TensorProto::UINT64;
    case DType::kBool:     return onnx::TensorProto::BOOL;
    default:               return onnx::TensorProto::UNDEFINED;
  }
}

int64_t DTypeBytes(DType dt) {
  switch (dt) {
    case DType::kFloat32:
    case DType::kInt32:
    case DType::kUint32:   return 4;
    case DType::kFloat16:
    case DType::kBFloat16:
    case DType::kInt16:
    case DType::kUint16:   return 2;
    case DType::kFloat64:
    case DType::kInt64:
    case DType::kUint64:   return 8;
    case DType::kInt8:
    case DType::kUint8:
    case DType::kBool:     return 1;
    default:               return 0;
  }
}

const char* DTypeName(DType dt) {
  switch (dt) {
    case DType::kFloat32:  return "float32";
    case DType::kFloat16:  return "float16";
    case DType::kBFloat16: return "bfloat16";
    case DType::kFloat64:  return "float64";
    case DType::kInt8:     return "int8";
    case DType::kUint8:    return "uint8";
    case DType::kInt16:    return "int16";
    case DType::kUint16:   return "uint16";
    case DType::kInt32:    return "int32";
    case DType::kUint32:   return "uint32";
    case DType::kInt64:    return "int64";
    case DType::kUint64:   return "uint64";
    case DType::kBool:     return "bool";
    default:               return "unknown";
  }
}

bool ShapeIsResolved(const Shape& s) {
  for (auto d : s) {
    if (d < 0) return false;
  }
  return true;
}

std::string ShapeToString(const Shape& s) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) oss << ", ";
    if (s[i] == kUnknownDim) {
      oss << "?";
    } else {
      oss << s[i];
    }
  }
  oss << "]";
  return oss.str();
}

int64_t Tensor::NumElements() const {
  if (shape.empty()) return 1;  // scalar
  int64_t n = 1;
  for (auto d : shape) {
    if (d < 0) return -1;
    n *= d;
  }
  return n;
}

int64_t Tensor::SizeBytes() const {
  int64_t n = NumElements();
  if (n < 0) return -1;
  return n * DTypeBytes(dtype);
}

const onnx::AttributeProto* Node::GetAttr(const std::string& name) const {
  for (const auto& a : attributes) {
    if (a.name() == name) return &a;
  }
  return nullptr;
}

int64_t Node::GetAttrInt(const std::string& name, int64_t default_value) const {
  const auto* a = GetAttr(name);
  if (!a || a->type() != onnx::AttributeProto::INT) return default_value;
  return a->i();
}

std::vector<int64_t> Node::GetAttrInts(const std::string& name) const {
  std::vector<int64_t> out;
  const auto* a = GetAttr(name);
  if (!a || a->type() != onnx::AttributeProto::INTS) return out;
  out.reserve(a->ints_size());
  for (int i = 0; i < a->ints_size(); ++i) out.push_back(a->ints(i));
  return out;
}

float Node::GetAttrFloat(const std::string& name, float default_value) const {
  const auto* a = GetAttr(name);
  if (!a || a->type() != onnx::AttributeProto::FLOAT) return default_value;
  return a->f();
}

std::string Node::GetAttrString(const std::string& name,
                                const std::string& default_value) const {
  const auto* a = GetAttr(name);
  if (!a || a->type() != onnx::AttributeProto::STRING) return default_value;
  return a->s();
}

Tensor* Graph::GetTensor(const std::string& name) {
  auto it = tensors.find(name);
  return it == tensors.end() ? nullptr : &it->second;
}

const Tensor* Graph::GetTensor(const std::string& name) const {
  auto it = tensors.find(name);
  return it == tensors.end() ? nullptr : &it->second;
}

Tensor& Graph::GetOrCreateTensor(const std::string& name) {
  auto it = tensors.find(name);
  if (it != tensors.end()) return it->second;
  Tensor t;
  t.name = name;
  return tensors.emplace(name, std::move(t)).first->second;
}

}  // namespace inferc
