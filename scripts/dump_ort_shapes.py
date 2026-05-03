"""Run ONNX's official shape inference on DistilBERT and dump the result.

Produces models/golden_shapes.json — a list of every node, in topological
order, with its op_type, name, and output shapes / dtypes. inferc's
shape-inference test loads this and asserts per-node equality.

Symbolic dims (e.g., "batch", "sequence") are emitted as -1 to match
inferc's internal kUnknownDim convention.

Run via: poetry run python scripts/dump_ort_shapes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx
from onnx import shape_inference

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "distilbert.onnx"
OUT_PATH = ROOT / "models" / "golden_shapes.json"


def vi_to_record(vi: onnx.ValueInfoProto) -> dict:
    dtype = int(vi.type.tensor_type.elem_type)
    # Distinguish "no shape inferred" (None) from "rank-0 scalar" ([]).
    if not vi.type.tensor_type.HasField("shape"):
        return {"name": vi.name, "shape": None, "dtype": dtype}
    dims: list[int] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            # symbolic ("batch", "sequence", "Unsqueezeoutput_dim_1", etc.) → -1
            dims.append(-1)
    return {"name": vi.name, "shape": dims, "dtype": dtype}


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"error: {ONNX_PATH} not found. Run scripts/fetch_distilbert.py first.",
              file=sys.stderr)
        return 1

    model = onnx.load(str(ONNX_PATH))
    # data_prop=True lets the inferencer propagate constant-folded shape data
    # through Shape/Slice/Concat chains, which matters for transformer graphs
    # full of dynamic-looking shape ops that are actually static.
    inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)

    # Build a name → ValueInfoProto map across initializers, inputs, outputs, and value_info.
    name_to_vi: dict[str, onnx.ValueInfoProto] = {}
    for vi in inferred.graph.input: name_to_vi[vi.name] = vi
    for vi in inferred.graph.output: name_to_vi[vi.name] = vi
    for vi in inferred.graph.value_info: name_to_vi[vi.name] = vi

    nodes_out = []
    for i, node in enumerate(inferred.graph.node):
        outs = []
        for o in node.output:
            vi = name_to_vi.get(o)
            if vi is None:
                outs.append({"name": o, "shape": None, "dtype": 0})
            else:
                outs.append(vi_to_record(vi))
        nodes_out.append({
            "index": i,
            "op_type": node.op_type,
            "name": node.name,
            "outputs": outs,
        })

    summary = {
        "model": str(ONNX_PATH.name),
        "node_count": len(nodes_out),
        "graph_inputs": [vi_to_record(vi) for vi in inferred.graph.input],
        "graph_outputs": [vi_to_record(vi) for vi in inferred.graph.output],
        "nodes": nodes_out,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    inferred_count = sum(
        1 for n in nodes_out for o in n["outputs"] if o["shape"] is not None
    )
    total_outputs = sum(len(n["outputs"]) for n in nodes_out)
    print(f"Wrote {OUT_PATH}")
    print(f"  {len(nodes_out)} nodes, {inferred_count}/{total_outputs} output shapes inferred by ORT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
