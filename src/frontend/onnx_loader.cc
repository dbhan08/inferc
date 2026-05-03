#include "frontend/onnx_loader.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <utility>

namespace inferc {

namespace {

constexpr int64_t kDimUnknown = -1;

void ExtractIO(const onnx::ValueInfoProto& vi, ModelSummary::IO* io) {
  io->name = vi.name();
  if (!vi.has_type()) return;
  const auto& t = vi.type();
  if (!t.has_tensor_type()) return;
  const auto& tt = t.tensor_type();
  io->dtype = tt.elem_type();
  if (!tt.has_shape()) return;
  for (const auto& dim : tt.shape().dim()) {
    if (dim.has_dim_value()) {
      io->shape.push_back(dim.dim_value());
      io->dim_param.emplace_back();
    } else {
      io->shape.push_back(kDimUnknown);
      io->dim_param.push_back(dim.dim_param());
    }
  }
}

int64_t TensorBytes(const onnx::TensorProto& t) {
  int64_t elem_bytes = OnnxElementBytes(t.data_type());
  if (elem_bytes == 0) return 0;
  int64_t count = 1;
  for (auto d : t.dims()) count *= d;
  return count * elem_bytes;
}

std::string FormatShape(const ModelSummary::IO& io) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < io.shape.size(); ++i) {
    if (i) oss << ", ";
    if (io.shape[i] == kDimUnknown) {
      if (!io.dim_param[i].empty()) {
        oss << io.dim_param[i];
      } else {
        oss << "?";
      }
    } else {
      oss << io.shape[i];
    }
  }
  oss << "]";
  return oss.str();
}

}  // namespace

int64_t OnnxElementBytes(int32_t onnx_dtype) {
  switch (onnx_dtype) {
    case onnx::TensorProto::FLOAT:    return 4;
    case onnx::TensorProto::UINT8:    return 1;
    case onnx::TensorProto::INT8:     return 1;
    case onnx::TensorProto::UINT16:   return 2;
    case onnx::TensorProto::INT16:    return 2;
    case onnx::TensorProto::INT32:    return 4;
    case onnx::TensorProto::INT64:    return 8;
    case onnx::TensorProto::BOOL:     return 1;
    case onnx::TensorProto::FLOAT16:  return 2;
    case onnx::TensorProto::DOUBLE:   return 8;
    case onnx::TensorProto::UINT32:   return 4;
    case onnx::TensorProto::UINT64:   return 8;
    case onnx::TensorProto::BFLOAT16: return 2;
    default: return 0;
  }
}

const char* OnnxDTypeName(int32_t onnx_dtype) {
  switch (onnx_dtype) {
    case onnx::TensorProto::FLOAT:    return "float32";
    case onnx::TensorProto::UINT8:    return "uint8";
    case onnx::TensorProto::INT8:     return "int8";
    case onnx::TensorProto::UINT16:   return "uint16";
    case onnx::TensorProto::INT16:    return "int16";
    case onnx::TensorProto::INT32:    return "int32";
    case onnx::TensorProto::INT64:    return "int64";
    case onnx::TensorProto::STRING:   return "string";
    case onnx::TensorProto::BOOL:     return "bool";
    case onnx::TensorProto::FLOAT16:  return "float16";
    case onnx::TensorProto::DOUBLE:   return "float64";
    case onnx::TensorProto::UINT32:   return "uint32";
    case onnx::TensorProto::UINT64:   return "uint64";
    case onnx::TensorProto::BFLOAT16: return "bfloat16";
    default: return "unknown";
  }
}

bool LoadOnnx(const std::string& path, onnx::ModelProto* out_model) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return false;
  return out_model->ParseFromIstream(&f);
}

ModelSummary SummarizeModel(const onnx::ModelProto& model) {
  ModelSummary s;
  s.ir_version = model.ir_version();
  s.producer_name = model.producer_name();
  s.producer_version = model.producer_version();

  for (const auto& op : model.opset_import()) {
    if (op.domain().empty()) {
      s.opset_version = op.version();
      s.opset_domain = "ai.onnx";
      break;
    }
  }

  if (!model.has_graph()) return s;
  const auto& g = model.graph();
  s.graph_name = g.name();

  for (const auto& vi : g.input()) {
    ModelSummary::IO io;
    ExtractIO(vi, &io);
    s.inputs.push_back(std::move(io));
  }
  for (const auto& vi : g.output()) {
    ModelSummary::IO io;
    ExtractIO(vi, &io);
    s.outputs.push_back(std::move(io));
  }

  s.node_count = g.node_size();
  for (const auto& node : g.node()) {
    s.op_type_counts[node.op_type()]++;
  }

  s.initializer_count = g.initializer_size();
  for (const auto& init : g.initializer()) {
    s.initializer_total_bytes += TensorBytes(init);
  }

  // Some old exports list initializers as graph inputs too. Filter those out
  // for cleaner summary display (they're already accounted for in initializers).
  if (g.initializer_size() > 0 && g.input_size() > 0) {
    std::vector<std::string> init_names;
    init_names.reserve(g.initializer_size());
    for (const auto& init : g.initializer()) init_names.push_back(init.name());
    std::sort(init_names.begin(), init_names.end());
    auto is_init = [&](const std::string& n) {
      return std::binary_search(init_names.begin(), init_names.end(), n);
    };
    s.inputs.erase(
        std::remove_if(s.inputs.begin(), s.inputs.end(),
                       [&](const ModelSummary::IO& io) { return is_init(io.name); }),
        s.inputs.end());
  }

  return s;
}

void PrintSummary(const ModelSummary& summary, std::ostream& os) {
  os << "Model: " << summary.graph_name << "\n";
  os << "  IR version:        " << summary.ir_version << "\n";
  os << "  Producer:          " << summary.producer_name;
  if (!summary.producer_version.empty()) os << " " << summary.producer_version;
  os << "\n";
  os << "  Opset:             " << summary.opset_domain
     << " v" << summary.opset_version << "\n";
  os << "\n";

  os << "Inputs (" << summary.inputs.size() << "):\n";
  for (const auto& io : summary.inputs) {
    os << "  " << std::left << std::setw(20) << io.name
       << std::setw(10) << OnnxDTypeName(io.dtype)
       << FormatShape(io) << "\n";
  }
  os << "\n";

  os << "Outputs (" << summary.outputs.size() << "):\n";
  for (const auto& io : summary.outputs) {
    os << "  " << std::left << std::setw(20) << io.name
       << std::setw(10) << OnnxDTypeName(io.dtype)
       << FormatShape(io) << "\n";
  }
  os << "\n";

  os << "Nodes: " << summary.node_count << "\n";
  os << "  Op type counts (descending):\n";
  std::vector<std::pair<std::string, int64_t>> sorted(
      summary.op_type_counts.begin(), summary.op_type_counts.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
              if (a.second != b.second) return a.second > b.second;
              return a.first < b.first;
            });
  for (const auto& [op, count] : sorted) {
    os << "    " << std::left << std::setw(28) << op << count << "\n";
  }
  os << "\n";

  os << "Initializers: " << summary.initializer_count << "  ("
     << std::fixed << std::setprecision(2)
     << summary.initializer_total_bytes / 1.0e6 << " MB)\n";
}

}  // namespace inferc
