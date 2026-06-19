# Regenerates the scoreboard figure: GPT-2-style prefill GEMM throughput,
# cblas_sgemm vs BNNSMatMul vs the proposed pre-packed kernel. Values are the
# median GFLOPS over 11 isolated invocations (main-result table).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

shapes = ["QKV\n(2048, 2048)", "FFN-up\n(8192, 2048)",
          "FFN-down\n(2048, 8192)", "LM head\n(60000, 2048)"]
cblas      = [614, 403, 527, 831]
bnnsmatmul = [809, 681, 653, 838]
proposed   = [1221, 1239, 1144, 1211]

x = np.arange(len(shapes)); w = 0.25
fig, ax = plt.subplots(figsize=(6.6, 3.1))
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7)

b1 = ax.bar(x - w, cblas,      w, label="cblas_sgemm",           color="#c7cdd6", edgecolor="none")
b2 = ax.bar(x,     bnnsmatmul, w, label="BNNSMatMul",            color="#6f93bd", edgecolor="none")
b3 = ax.bar(x + w, proposed,   w, label="proposed (pre-packed)", color="#274472", edgecolor="none")

for bars in (b1, b2, b3):
    ax.bar_label(bars, fmt="%d", padding=2, fontsize=6.3, color="#333333")

ax.set_xticks(x); ax.set_xticklabels(shapes, fontsize=8.5)
ax.set_ylabel("GFLOPS (fp32, bit-exact)", fontsize=9)
ax.set_ylim(0, 1380)
ax.tick_params(axis="y", labelsize=8)
ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          frameon=False, handlelength=1.3, columnspacing=1.6, borderaxespad=0.3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")
fig.tight_layout()
fig.savefig("/Users/deyvikbhan/develop/inferc/docs/fig_scoreboard.png", dpi=220, bbox_inches="tight")
print("wrote fig_scoreboard.png")
