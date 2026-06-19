# Regenerates Figure 1: the panel-width sweep. Throughput of the proposed
# pre-packed kernel at the QKV shape (128, 2048, 2048) as the column-panel width
# Nc shrinks from 512 (a coarse width: four panels, P-cluster only) to 64
# (32 panels, both AMX blocks). Every point is a
# median over eleven isolated invocations on macOS Tahoe 26.5.1
# (bench/amx/amx_nc_sweep_fig.cc), bit-exact. Horizontal lines are the
# BNNSMatMul and BNNS Graph medians at this shape (Tables 3, 4).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Nc       = [512, 384, 256, 192, 128, 64]
panels   = [4,   6,   8,   11,  16,  32]
proposed = [631, 888, 1015, 1175, 1129, 1205]  # best-Kc median over 11 isolated invocations, Tahoe 26.5.1
BNNSMATMUL = 809    # Table 3, QKV (2048,2048), median over 11
BNNSGRAPH  = 1116   # Table 4, QKV (2048,2048), median over 11

x = np.arange(len(Nc))  # categorical, left=512 -> right=64 ("finer panels ->")
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7)

# reference baselines
ax.axhline(BNNSGRAPH, color="#6f93bd", ls="--", lw=1.3)
ax.axhline(BNNSMATMUL, color="#9aa3ad", ls="--", lw=1.3)
ax.text(len(Nc)-1, BNNSGRAPH+22, "BNNS Graph (1116)", ha="right", va="bottom",
        fontsize=8, color="#4f6f9b")
ax.text(len(Nc)-1, BNNSMATMUL+22, "BNNSMatMul (809)", ha="right", va="bottom",
        fontsize=8, color="#7a828c")

# proposed kernel curve
ax.plot(x, proposed, "-o", color="#274472", lw=2.2, ms=7, zorder=5,
        label="proposed (pre-packed), bit-exact")
for xi, g in zip(x, proposed):
    ax.annotate(f"{g}", (xi, g), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=7.5, color="#274472")

# annotate the two endpoints of interest
ax.annotate("coarse $N_c$=512\n(P-cluster only)", (x[0], proposed[0]),
            textcoords="offset points", xytext=(6, -30), fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
ax.annotate("deployed $N_c$=64", (x[-1], proposed[-1]),
            textcoords="offset points", xytext=(-8, -30), ha="right", fontsize=8,
            color="#333333", arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
# make explicit that every point is measured on the current macOS
ax.text(0.015, 0.97, "all points: macOS Tahoe 26.5.1", transform=ax.transAxes,
        ha="left", va="top", fontsize=7.5, color="#666666")

ax.set_xticks(x)
ax.set_xticklabels([f"{nc}\n({p} panels)" for nc, p in zip(Nc, panels)], fontsize=8)
ax.set_xlabel("column-panel width $N_c$  (finer $\\rightarrow$)", fontsize=9)
ax.set_ylabel("GFLOPS (fp32, bit-exact)", fontsize=9)
ax.set_ylim(0, 1400)
ax.tick_params(axis="y", labelsize=8)
ax.legend(fontsize=8, loc="lower right", frameon=True, facecolor="white",
          framealpha=1.0, edgecolor="none")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")
fig.tight_layout()
fig.savefig("/Users/deyvikbhan/develop/inferc/docs/fig_panel_sweep.png", dpi=220, bbox_inches="tight")
print("wrote fig_panel_sweep.png")
