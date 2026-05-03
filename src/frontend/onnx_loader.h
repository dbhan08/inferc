#pragma once

#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

#include "onnx.pb.h"

namespace inferc {

struct ModelSummary {
  int64_t ir_version = 0;
  std::string producer_name;
  std::string producer_version;
  int64_t opset_version = 0;
  std::string opset_domain;  // empty in proto == "ai.onnx" by convention
  std::string graph_name;

  struct IO {
    std::string name;
    int32_t dtype = 0;            // onnx::TensorProto::DataType enum
    std::vector<int64_t> shape;   // -1 for symbolic / unknown dim
    std::vector<std::string> dim_param;  // symbolic name per dim, empty if numeric
  };
  std::vector<IO> inputs;
  std::vector<IO> outputs;

  int64_t node_count = 0;
  std::map<std::string, int64_t> op_type_counts;

  int64_t initializer_count = 0;
  int64_t initializer_total_bytes = 0;
};

// Returns the byte size of one element of the given ONNX tensor data type, or
// 0 if the dtype is variable-width (string) or unknown.
int64_t OnnxElementBytes(int32_t onnx_dtype);
const char* OnnxDTypeName(int32_t onnx_dtype);

// Reads `path` from disk and parses it as an ONNX ModelProto.
// Returns false on missing file or malformed protobuf.
bool LoadOnnx(const std::string& path, onnx::ModelProto* out_model);

// Builds a summary from an already-loaded model.
ModelSummary SummarizeModel(const onnx::ModelProto& model);

// Pretty-prints the summary to the stream.
void PrintSummary(const ModelSummary& summary, std::ostream& os);

}  // namespace inferc
