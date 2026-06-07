"""Generate paper figures for the AMX prefill GEMM result.

Reproduces two figures from current bench numbers (`bench/amx/prefill_tune.cc`,
`bench/amx/openblas_sanity.cc`, 5-run mean ± std):

  fig_scoreboard.png         — bar chart of GFLOPS by shape × backend
  fig_optimization_curve.png — optimization curve from naive AMX to BLIS+Kc

Reads numbers from this file (hard-coded for reproducibility — they are the
exact numbers cited in docs/PAPER_DRAFT.md so a reviewer can match them).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# GPT-2-style row group of Table 1: paired mean over 30 trials, deployable
# amx_sgemm_auto, single thread (bench/amx/llama_shapes_bench.cc). These are
# the exact GPT-2-style numbers in docs/PAPER_DRAFT.md Table 1.
SHAPES = ["QKV\n[128,2048,2048]", "FFN1\n[128,8192,2048]",
          "FFN2\n[128,2048,8192]", "LM-head\n[128,60000,2048]"]
RESULTS = {
    "OpenBLAS NEON":   {"mean": [197, 230,  70, 293], "std": [0, 0, 0, 0]},
    "Accelerate AMX":  {"mean": [472, 368, 453, 941], "std": [33, 39, 18, 15]},
    "our BLIS+Kc":     {"mean": [587, 463, 627, 724], "std": [43, 72, 25, 20]},
}

# Optimization curve at QKV [128, 2048, 2048]. The final BLIS+Kc bar is the
# deployable kernel's paired mean (587, = Table 1 GPT-2-style QKV); the
# Accelerate reference is the paired mean at the same shape (472).
OPT_CURVE = [
    ("naive AMX 16x16",                  194),
    ("+ cache blocking",                 212),
    ("+ LDX_pair",                       204),  # regression
    ("+ sw-pipelined ping-pong",         211),
    ("+ B-panel packing (Phase 1)",      407),
    ("+ Kc blocking + LDZ/STZ carry",    587),  # the win (deployable, Table 1)
]

def fig_scoreboard():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_shapes = len(SHAPES)
    n_backends = len(RESULTS)
    x = np.arange(n_shapes)
    width = 0.8 / n_backends
    colors = {"OpenBLAS NEON": "#888", "Accelerate AMX": "#1f77b4",
              "our BLIS+Kc": "#d62728"}
    for i, (label, data) in enumerate(RESULTS.items()):
        offsets = x + (i - n_backends / 2 + 0.5) * width
        bars = ax.bar(offsets, data["mean"], width, yerr=data["std"],
                      label=label, color=colors[label], capsize=3)
    # Accelerate "peak" line
    ax.axhline(1400, color="grey", linestyle=":", linewidth=0.8,
               label="M1 AMX fp32 peak (~1400)")
    ax.set_ylabel("GFLOPS (single thread, M1 P-core)")
    ax.set_xticks(x)
    ax.set_xticklabels(SHAPES, fontsize=9)
    ax.set_title("LLM prefill GEMM scoreboard\n(M=128 batch, fp32, bit-exact)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/fig_scoreboard.png", dpi=150)
    print("wrote docs/fig_scoreboard.png")

def fig_opt_curve():
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [step[0] for step in OPT_CURVE]
    gflops = [step[1] for step in OPT_CURVE]
    accel = 472     # QKV Accelerate (paired mean, Table 1)
    x = np.arange(len(labels))
    colors = ["#ff7f0e" if g > accel else "#d62728" if g == max(gflops)
              else "#1f77b4" for g in gflops]
    bars = ax.bar(x, gflops, color=colors)
    ax.axhline(accel, color="grey", linestyle="--",
               label=f"Accelerate ({accel} GFLOPS)")
    for i, g in enumerate(gflops):
        ax.text(i, g + 15, f"{g}", ha="center", fontsize=9)
    ax.set_ylabel("GFLOPS (single thread, M1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_title("Optimization curve at QKV shape [128, 2048, 2048]\n"
                 "all variants bit-exact vs Accelerate")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 800)
    plt.tight_layout()
    plt.savefig("docs/fig_optimization_curve.png", dpi=150)
    print("wrote docs/fig_optimization_curve.png")

if __name__ == "__main__":
    fig_scoreboard()
    fig_opt_curve()
