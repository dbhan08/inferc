# Paper 2 Fig: profile-driven optimization of the M=16 codebook kernel (fp32).
# baseline (issue/load-bound) -> N-tile blocking -> index-amortization. M1, GFLOP/s.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
labels=["baseline\n(load-bound)","+ N-tile\nblocking","+ index\namortization"]
g=[320,424,1062]
x=np.arange(len(labels))
fig,ax=plt.subplots(figsize=(4.4,3.0)); ax.set_axisbelow(True)
ax.yaxis.grid(True,color="#e6e6e6",linewidth=0.7)
bars=ax.bar(x,g,0.6,color=["#c7cdd6","#6f93bd","#274472"],edgecolor="none")
ax.bar_label(bars,fmt="%d",padding=2,fontsize=8,color="#333")
ax.axhline(1424,color="#c0392b",linewidth=1.0,linestyle="--")
ax.text(len(labels)-0.5,1448,"AMX MATFP ceiling",fontsize=7,color="#c0392b",ha="right")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8)
ax.set_ylabel("GFLOP/s (M=16, single-thread)",fontsize=9); ax.set_ylim(0,1560)
ax.tick_params(axis="y",labelsize=8)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#888"); ax.spines["bottom"].set_color("#888")
fig.tight_layout(); fig.savefig("docs/fig_optwaterfall.png",dpi=220,bbox_inches="tight"); print("wrote fig_optwaterfall.png")
