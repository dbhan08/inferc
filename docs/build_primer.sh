#!/usr/bin/env bash
#
# build_primer.sh — render docs/PRIMER.md to docs/PRIMER.pdf.
#
# A teaching document ("inferc from scratch"), NOT a two-column paper: single
# column `article` class, table of contents, generous margins, real math. Same
# pandoc -> tectonic pipeline as build_paper.sh but simplified for a primer:
#   1. YAML front matter (title/author/TOC/class options) is prepended.
#   2. Unicode math symbols in the body are rewritten to LaTeX math.
#   3. pandoc -> intermediate primer.tex (--listings for code, --toc).
#   4. tectonic primer.tex -> primer.pdf, copied to docs/PRIMER.pdf.
#
# Requires: pandoc, tectonic. Portable: Latin Modern default, no system fonts.
# Usage: bash docs/build_primer.sh   (run from repo root)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/docs/PRIMER.md"
OUT="$REPO/docs/PRIMER.pdf"
BUILD="$(mktemp -d /tmp/primerbuild.XXXXXX)"

# copy any figures the primer references (added as they are created)
for f in fig_mechanism.png fig_pareto.png; do
  [ -f "$REPO/docs/$f" ] && cp "$REPO/docs/$f" "$BUILD/" || true
done

cat > "$BUILD/primer.md" <<'YAML'
---
title: "inferc, From Scratch"
subtitle: "A Student's Guide to Building an LLM Inference Engine and Beating the Hardware"
author: "Deyvik Bhan"
documentclass: article
classoption: [11pt]
geometry: [margin=1in]
toc: true
toc-depth: 2
numbersections: false
colorlinks: true
linkcolor: RoyalBlue
urlcolor: RoyalBlue
header-includes: |
  \usepackage{microtype}
  \usepackage{booktabs}
  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{mathtools}
  \usepackage{listings}
  \usepackage{xurl}
  \usepackage{mdframed}
  \emergencystretch=2em
  \frenchspacing
  \lstset{basicstyle=\ttfamily\small,frame=single,framesep=4pt,breaklines=true,columns=fullflexible,keepspaces=true}
  \providecommand{\tightlist}{\setlength{\itemsep}{2pt}\setlength{\parskip}{0pt}}
---

YAML

# Rewrite Unicode math -> LaTeX in the body, then append it.
python3 - "$SRC" "$BUILD/primer.md" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
for old, new in [('≥', r'$\geq$'), ('≤', r'$\leq$'), ('≈', r'$\approx$'),
                 ('≫', r'$\gg$'), ('≪', r'$\ll$'), ('×', r'$\times$'),
                 ('∈', r'$\in$'), ('ᵀ', r'$^{\top}$'), ('→', r'$\rightarrow$'),
                 ('∑', r'$\sum$'), ('√', r'$\sqrt{\ }$'), ('∞', r'$\infty$'),
                 ('α', r'$\alpha$'), ('σ', r'$\sigma$'), ('μ', r'$\mu$'),
                 ('θ', r'$\theta$'), ('λ', r'$\lambda$'), ('±', r'$\pm$')]:
    src = src.replace(old, new)
with open(sys.argv[2], 'a') as f:
    f.write(src)
PY

pandoc "$BUILD/primer.md" -o "$BUILD/primer.tex" --standalone \
  -f markdown+autolink_bare_uris+tex_math_dollars \
  --toc --listings

( cd "$BUILD" && tectonic primer.tex 2>&1 | tail -3 )
cp "$BUILD/primer.pdf" "$OUT"
pdfinfo "$OUT" 2>/dev/null | grep -E '^Pages|^File' || true
echo "Wrote $OUT"
