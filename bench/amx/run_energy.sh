#!/usr/bin/env bash
# Perf-per-watt: AMX codebook MATFP vs NEON sdot codebook, K=2048 N=8192 M=64.
# Must run as root (powermetrics needs it). From repo root:
#     ! sudo bash bench/amx/run_energy.sh
# Prints GFLOP/s, average CPU power (W), and GFLOP/s-per-watt for each engine.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=/tmp/eload_e
DUR=6                                   # seconds of sustained load per arm
[ "$(id -u)" = "0" ] || { echo "run with sudo: ! sudo bash bench/amx/run_energy.sh"; exit 1; }

clang++ -O3 -std=c++17 -I"$REPO/third_party" "$REPO/bench/amx/amx_energy_load.cc" -o "$BIN"

arm() {  # $1=mode $2=threads
  local mode=$1 th=$2
  "$BIN" "$mode" "$th" "$DUR" > /tmp/eload_$mode.txt 2>&1 &
  local lp=$!
  # sample CPU power for ~DUR seconds (200 ms interval), capture "CPU Power: N mW"
  powermetrics --samplers cpu_power -i 200 -n $((DUR*5)) 2>/dev/null \
    | grep -i "^CPU Power:" > /tmp/pm_$mode.txt || true
  wait "$lp"
  local gf w
  gf=$(grep -oE '[0-9]+ GFLOP/s' /tmp/eload_$mode.txt | grep -oE '[0-9]+' | head -1)
  # average the mW samples -> watts
  w=$(awk '{s+=$3; n++} END{ if(n) printf "%.2f", (s/n)/1000 }' /tmp/pm_$mode.txt)
  printf "%-5s  threads=%-2s  %6s GFLOP/s   %6s W   %6.1f GFLOP/s per W\n" \
    "$mode" "$th" "$gf" "$w" "$(echo "$gf / $w" | bc -l)"
}

echo "== perf/watt (sustained ${DUR}s, K=2048 N=8192 M=64) =="
arm amx 4
arm amx 2
arm neon 8
echo "(AMX reaches its throughput on fewer active cores; the per-watt ratio is the energy story.)"
