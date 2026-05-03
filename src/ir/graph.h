#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "onnx.pb.h"

namespace inferc {

// Internal dtype enum. Mirrors the ONNX TensorProto.DataType subset we
// actually care about. Conversions to/from ONNX live in the .cc.
enum class DType : int8_t {
  kUnknown = 0,
  kFloat32,
  kFloat16,
  kBFloat16,
  kFloat64,
  kInt8,  kUint8,
  kInt16, kUint16,
  kInt32, kUint32,
  kInt64, kUint64,
  kBool,
};

DType DTypeFromOnnx(int32_t onnx_dtype);
int32_t DTypeToOnnx(DType dt);
int64_t DTypeBytes(DType dt);
const char* DTypeName(DType dt);

// Shape: vector of dim values. Conventions:
//   >= 0  → concrete dim
//   -1    → unknown / not yet inferred
// We treat all symbolic dims (e.g., "batch") as -1 internally; a separate
// pass binds concrete shapes when the user provides input shapes at run time.
using Shape = std::vector<int64_t>;

constexpr int64_t kUnknownDim = -1;

bool ShapeIsResolved(const Shape& s);  // all dims >= 0
std::string ShapeToString(const Shape& s);

// Tensor is a named typed shape, optionally backed by initializer bytes.
struct Tensor {
  std::string name;
  DType dtype = DType::kUnknown;
  Shape shape;

  // Raw bytes from an ONNX initializer (weights, biases, constants). Empty
  // for activations / inputs / outputs that get computed at run time.
  std::vector<uint8_t> raw_data;
  bool IsInitializer() const { return !raw_data.empty(); }

  int64_t NumElements() const;
  int64_t SizeBytes() const;
};

// Node = one operator. Inputs/outputs are tensor names that resolve via
// Graph::tensors. Attributes are kept as ONNX AttributeProto so we don't
// reinvent the (large) set of attribute types.
struct Node {
  std::string name;
  std::string op_type;
  std::string domain;  // empty == ai.onnx
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<onnx::AttributeProto> attributes;

  // Returns null if no such attribute.
  const onnx::AttributeProto* GetAttr(const std::string& name) const;
  // Convenience accessors with defaults; callers handle absence themselves
  // when they care.
  int64_t GetAttrInt(const std::string& name, int64_t default_value) const;
  std::vector<int64_t> GetAttrInts(const std::string& name) const;
  float GetAttrFloat(const std::string& name, float default_value) const;
  std::string GetAttrString(const std::string& name,
                            const std::string& default_value) const;
};

// Graph = topologically-ordered nodes + a tensor table. Inputs and outputs
// are listed by name (into tensors).
struct Graph {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<Node> nodes;

  // Owning storage of all tensors referenced anywhere in the graph.
  // Keyed by name. Initializers, inputs, outputs, and intermediate
  // activations all live here.
  std::map<std::string, Tensor> tensors;

  Tensor* GetTensor(const std::string& name);
  const Tensor* GetTensor(const std::string& name) const;
  Tensor& GetOrCreateTensor(const std::string& name);
};

}  // namespace inferc
