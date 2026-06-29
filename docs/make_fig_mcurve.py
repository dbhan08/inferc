# Paper 2 Fig: optimized kernel speedup over fair (repacked) ggml Q4 across batch M,
# single-thread and multi-thread. M1, K=2048 N=8192. Measured (amx_mcurve_opt_mt.cc).
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
M=["1","4","16","32","64"]
st=[0.52,1.21,3.94,3.61,2.72]; mt=[0.48,0.67,1.57,1.47,1.04]
x=np.arange(len(M)); w=0.36
fig,ax=plt.subplots(figsize=(6.6,3.0)); ax.set_axisbelow(True)
ax.yaxis.grid(True,color="#e6e6e6",linewidth=0.7)
b1=ax.bar(x-w/2,st,w,label="single-thread",color="#274472",edgecolor="none")
b2=ax.bar(x+w/2,mt,w,label="multi-thread (8)",color="#6f93bd",edgecolor="none")
for b in (b1,b2): ax.bar_label(b,fmt="%.1f",padding=2,fontsize=6.3,color="#333")
ax.axhline(1.0,color="#c0392b",linewidth=1.0,linestyle="--")
ax.text(len(M)-0.5,1.03,"parity (ggml)",fontsize=7,color="#c0392b",ha="right")
ax.set_xticks(x); ax.set_xticklabels(M,fontsize=9)
ax.set_xlabel("batch M (tokens)",fontsize=9); ax.set_ylabel("speedup over ggml Q4",fontsize=9)
ax.set_ylim(0,4.4); ax.tick_params(axis="y",labelsize=8)
ax.legend(fontsize=8,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2,frameon=False,handlelength=1.3,columnspacing=1.6,borderaxespad=0.3)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#888"); ax.spines["bottom"].set_color("#888")
fig.tight_layout(); fig.savefig("docs/fig_mcurve.png",dpi=220,bbox_inches="tight"); print("wrote fig_mcurve.png")
