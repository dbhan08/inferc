#pragma once

#include "ir/graph.h"

namespace inferc {
namespace passes {

// Recognizes BERT-style multi-head self-attention and folds it into a single
// `FusedAttention` op (inferc domain). Anchored on Softmax:
//
//   Qp ─ Reshape ─ Transpose ─ Div(/√d) ─┐
//                                          MatMul ─ Where(mask) ─ Softmax ─┐
//   Kp ─ Reshape ─ Transpose ────────────┘                                 MatMul ─ Transpose ─ Reshape ─→ ctx
//   Vp ─ Reshape ─ Transpose ──────────────────────────────────────────────┘
//
// Replaced with: ctx = FusedAttention(Qp, Kp, Vp, mask_cond)  [attrs: head_dim, fill]
// where Qp/Kp/Vp are the [B,S,H] projection outputs. head_dim is derived from the
// scale constant (√head_dim); fill from the Where constant. Eliminates the 3
// transposes, both attention MatMuls' intermediate materialization, the scale
// Div, the mask Where, and the Softmax buffer — all into one kernel.
//
// Returns the number of attention blocks folded.
int RecognizeAttention(Graph* graph);

// Fuses the Q/K/V projections feeding each FusedAttention — three independent
// `Add(bias, MatMul(X, W))` sharing the same input X — into one multi-output
// `FusedQKV(X, Wq,bq,Wk,bk,Wv,bv)` op whose kernel runs the three sgemms
// concurrently (they were otherwise serial graph nodes). Run after
// RecognizeAttention. Returns the number of QKV triples fused.
int FuseQKVProjection(Graph* graph);

}  // namespace passes
}  // namespace inferc
