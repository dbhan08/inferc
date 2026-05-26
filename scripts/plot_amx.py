"""Plot the AMX engagement sweep produced by `inferc amx-probe` (paper Figure 1).

Reads the CSV emitted by `inferc amx-probe --out-csv ...` and produces:

  1. A GFLOPs heatmap over the (M, NK) sgemm grid, with an overlaid contour at
     90% of empirical peak — the AMX engagement threshold curve.
  2. A line plot comparing the M=1 sgemm "decode row" against the sgemv path,
     as a function of feature dim NK. This is the Session-13 lever: where
     cblas_sgemv beats a single-row cblas_sgemm.

Run via:
  poetry run python scripts/plot_amx.py \\
      --csv bench/amx/amx_probe.csv --out bench/amx/amx_figure1.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt
import numpy as np


def load(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def build_gemm_grid(rows: list[dict]):
    gemm = [r for r in rows if r["kernel"] == "sgemm"]
    Ms = sorted({int(r["M"]) for r in gemm})
    NKs = sorted({int(r["N"]) for r in gemm})  # N == K for the square grid
    g = {(int(r["M"]), int(r["N"])): float(r["gflops"]) for r in gemm}
    Z = np.array([[g[(m, nk)] for nk in NKs] for m in Ms])  # [len(Ms), len(NKs)]
    return Ms, NKs, Z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bench/amx/amx_probe.csv")
    ap.add_argument("--out", default="bench/amx/amx_figure1.png")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: {csv_path} not found — run `inferc amx-probe` first", flush=True)
        return 1
    rows = load(csv_path)

    Ms, NKs, Z = build_gemm_grid(rows)
    peak = Z.max()

    has_f16 = any(r["kernel"] == "bnns_f16" for r in rows)
    ncols = 3 if has_f16 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    ax_hm, ax_ln = axes[0], axes[1]

    # ---- Panel 1: GFLOPs heatmap with 90%-of-peak threshold contour. ----
    im = ax_hm.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax_hm.set_xticks(range(len(NKs)))
    ax_hm.set_xticklabels(NKs, rotation=45, ha="right")
    ax_hm.set_yticks(range(len(Ms)))
    ax_hm.set_yticklabels(Ms)
    ax_hm.set_xlabel("feature dim  N = K")
    ax_hm.set_ylabel("rows  M  (M=1 is autoregressive decode)")
    ax_hm.set_title(f"cblas_sgemm GFLOPs on M1 (empirical peak {peak:.0f} GFLOPs)")
    # Threshold contour at 90% of peak: the AMX-engaged region boundary.
    cs = ax_hm.contour(
        np.arange(len(NKs)), np.arange(len(Ms)), Z,
        levels=[0.9 * peak], colors="white", linewidths=2.0, linestyles="--",
    )
    ax_hm.clabel(cs, fmt={0.9 * peak: "90% peak"}, fontsize=9)
    fig.colorbar(im, ax=ax_hm, label="GFLOPs")

    # ---- Panel 2: decode-shape comparison (sgemm M=1 vs sgemv) over NK. ----
    g_m1 = {int(r["N"]): float(r["gflops"])
            for r in rows if r["kernel"] == "sgemm" and int(r["M"]) == 1}
    g_gemv = {int(r["K"]): float(r["gflops"])
              for r in rows if r["kernel"] == "sgemv"}
    xs = sorted(set(g_m1) & set(g_gemv))
    ax_ln.plot(xs, [g_m1[x] for x in xs], "o-", label="sgemm  M=1  (1xNK · NKxNK)")
    ax_ln.plot(xs, [g_gemv[x] for x in xs], "s-", label="sgemv  (NKxNK · NK)")
    ax_ln.axvline(768, color="gray", ls=":", lw=1)
    ax_ln.text(768, ax_ln.get_ylim()[0], " GPT-2 hidden=768", fontsize=8, va="bottom")
    ax_ln.set_xlabel("feature dim  N = K")
    ax_ln.set_ylabel("GFLOPs")
    ax_ln.set_title("Decode-step throughput: sgemm M=1 vs sgemv")
    ax_ln.grid(True, alpha=0.3)
    ax_ln.legend()

    # ---- Panel 3 (optional): fp16 (BNNS) vs fp32 (sgemm) peak GFLOPs over NK. ----
    if has_f16:
        ax_f16 = axes[2]
        f32 = {}
        for r in rows:
            if r["kernel"] == "sgemm":
                f32[int(r["N"])] = max(f32.get(int(r["N"]), 0.0), float(r["gflops"]))
        f16 = {}
        for r in rows:
            if r["kernel"] == "bnns_f16":
                f16[int(r["K"])] = max(f16.get(int(r["K"]), 0.0), float(r["gflops"]))
        xs = sorted(set(f32) & set(f16))
        ax_f16.plot(xs, [f32[x] for x in xs], "o-", label="fp32 sgemm (peak)")
        ax_f16.plot(xs, [f16[x] for x in xs], "s-", label="fp16 BNNS (peak)")
        ax_f16.axvline(768, color="gray", ls=":", lw=1)
        ax_f16.set_xlabel("feature dim  N = K")
        ax_f16.set_ylabel("peak GFLOPs (best over M)")
        ax_f16.set_title("fp16 vs fp32 GEMM on M1 — fp16 loses below NK~1536")
        ax_f16.grid(True, alpha=0.3)
        ax_f16.legend()

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
