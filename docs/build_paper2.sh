#!/usr/bin/env bash
#
# build_paper2.sh — render docs/PAPER2_DRAFT.md to docs/PAPER2_DRAFT.pdf.
# Mirrors build_paper.sh (Paper 1): pandoc -> paper.tex -> tectonic, two-column
# IEEEtran. Figures: fig_mcurve.png, fig_optwaterfall.png. Tables are narrow
# (<=4 col) so all stay single-column. Standalone "---" rules are stripped.
# Requires: pandoc, tectonic. Usage: bash docs/build_paper2.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/docs/PAPER2_DRAFT.md"
OUT="$REPO/docs/PAPER2_DRAFT.pdf"
BUILD="$(mktemp -d /tmp/paper2build.XXXXXX)"

cp "$REPO/docs/fig_mcurve.png" "$BUILD/"
cp "$REPO/docs/fig_optwaterfall.png" "$BUILD/"

cat > "$BUILD/paper.md" <<'YAML'
---
title: "A Fused Codebook-Gather Matmul on the Apple AMX Coprocessor"
author: "AUTHORBLOCK"
abstract: |
  __ABSTRACT__
documentclass: IEEEtran
classoption: [conference]
indent: true
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes: |
  \usepackage{microtype}
  \usepackage{booktabs}
  \usepackage{caption}
  \usepackage{listings}
  \usepackage{xurl}
  \emergencystretch=2em
  \frenchspacing
  \captionsetup{font=footnotesize}
  \lstset{basicstyle=\ttfamily\scriptsize,frame=single,framesep=4pt,xleftmargin=4pt,xrightmargin=4pt,columns=fullflexible,keepspaces=true}
---

YAML

python3 - "$SRC" "$BUILD/paper.md" <<'PY'
import re, sys
src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
src = re.sub(r'(?m)^---\s*$', '', src)          # drop standalone hrules
_a = src[src.find('## Abstract'):]
_ab = re.sub(r'\s+', ' ', _a[_a.find('\n')+1:_a.find('## 1. Introduction')].strip())
_hdr = open(out_path).read().replace('__ABSTRACT__', _ab)
open(out_path, 'w').write(_hdr)
body = src[src.find('## 1. Introduction'):]
for old, new in [('≥', r'$\geq$'), ('≤', r'$\leq$'), ('≫', r'$\gg$'),
                 ('≪', r'$\ll$'), ('∈', r'$\in$'), ('×', 'x')]:
    body = body.replace(old, new)
# Relocate "*Table N. ...*" captions to a pandoc caption after the table.
tcap = re.compile(r'\*(Table \d+\.[^*]*?)\*\s*\n+(\|[^\n]*(?:\n\|[^\n]*)*)')
def reloc(m):
    cap = re.sub(r'\s+', ' ', m.group(1)).strip()
    return m.group(2) + '\n\n: ' + cap + '\n'
body = tcap.sub(reloc, body)
# Merge "*Figure N. caption*" into the preceding image alt text.
pat = re.compile(r'!\[[^\]]*\]\(([^)]+)\)\s*\n\s*\n\*Figure \d+\.\s*(.*?)\*', re.DOTALL)
def merge(m):
    cap = re.sub(r'\s+', ' ', m.group(2).replace('\n', ' ')).strip()
    return '![' + cap + '](' + m.group(1) + ')'
body = pat.sub(merge, body)
with open(out_path, 'a') as f:
    f.write(body)
PY

pandoc "$BUILD/paper.md" -o "$BUILD/paper.tex" --standalone \
  -f markdown+autolink_bare_uris --listings

python3 - "$BUILD/paper.tex" <<'PY'
import re, sys
path = sys.argv[1]
s = open(path).read()
s = s.replace('\\begin{minipage}[b]', '\\begin{minipage}[t]')
s = s.replace('\\author{AUTHORBLOCK}',
              '\\author{\\IEEEauthorblockN{Deyvik Bhan}\\IEEEauthorblockA{'
              'Georgia Institute of Technology\\\\dbhan6@gatech.edu}}')
s = s.replace('pdfauthor={AUTHORBLOCK}', 'pdfauthor={Deyvik Bhan}')

def extract_caption(body):
    key = '\\caption{'; idx = body.find(key)
    if idx < 0: return body, ''
    i = idx + len(key); depth = 1
    while i < len(body) and depth:
        if body[i] == '{': depth += 1
        elif body[i] == '}': depth -= 1
        i += 1
    cap = body[idx + len(key):i - 1]
    after = re.sub(r'^\s*\\tabularnewline', '', body[i:])
    return body[:idx] + after, cap

def delstinline(s):
    return re.sub(r'\\passthrough\{\\lstinline(.)(.*?)\1\}',
                  lambda m: '\\texttt{' + m.group(2) + '}', s)

def fix_lt(t):
    p = re.compile(r'\\begin\{longtable\}(?:\[[^\]]*\])?\{([^}]*)\}(.*?)\\end\{longtable\}', re.DOTALL)
    def f(m):
        cols, body = m.group(1), m.group(2)
        body, cap = extract_caption(body); cap = delstinline(cap).strip()
        body = re.sub(r'\\endfirsthead.*?\\endhead', '', body, flags=re.DOTALL)
        body = re.sub(r'\\bottomrule\s*(?:\\noalign\{\})?\s*\\endlastfoot', '', body, flags=re.DOTALL)
        for mk in ('\\endhead', '\\endfirsthead', '\\endlastfoot', '\\endfoot'):
            body = body.replace(mk, '')
        body = body.replace('\\noalign{}', '').strip()
        cap_tex = ('\\caption*{' + cap + '}\n') if cap else ''
        return ('\\begin{table}[t]\n\\centering\n' + cap_tex + '\\footnotesize\n'
                '\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{' + cols + '}\n'
                + body + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
    return p.sub(f, t)
s = fix_lt(s)
s = s.replace('\\begin{figure}', '\\begin{figure}[t]\n\\centering')

def wrap_lst(t):
    p = re.compile(r'(\\begin\{lstlisting\}.*?\\end\{lstlisting\})', re.DOTALL)
    return p.sub(lambda m: '\\begin{figure*}[t]\n\\centering\n\\begin{minipage}{0.78\\textwidth}\n'
                 + m.group(1) + '\n\\end{minipage}\n\\end{figure*}', t)
s = wrap_lst(s)

def break_paths(t):
    p = re.compile(r'\\passthrough\{\\lstinline(\S)(.*?)\1\}')
    def f(m):
        c = m.group(2)
        return '\\nolinkurl{' + c + '}' if ('/' in c and ' ' not in c) else m.group(0)
    return p.sub(f, t)
s = break_paths(s)
open(path, 'w').write(s)
PY

( cd "$BUILD" && tectonic paper.tex 2>&1 | tail -3 )
cp "$BUILD/paper.pdf" "$OUT"
pdfinfo "$OUT" 2>/dev/null | grep -E '^Pages' || true
echo "Wrote $OUT"
