#!/bin/bash
# CPU phase: score, vote, compare and report, for each model that has samples.
# No GPU needed.
#   scripts/cpu_phase.sh <name> ...
set -u
cd "$(dirname "$0")/.."
export EBMC=${EBMC:-$HOME/opt/src/hw-cbmc/src/ebmc/ebmc}
export PATH=$PATH:$HOME/opt/spot/bin          # appended: Spot's env also ships a python3
PY=${PY:-python3}
for name in "$@"; do
  [ -s "results/raw/$name.jsonl" ] && $PY scripts/score.py "results/raw/$name.jsonl" --workers 4
  [ -s "results/raw/$name.jsonl" ] && $PY scripts/score.py "results/raw/$name.jsonl" --workers 4 --fix-semicolon
  [ -s "results/scores/$name.verdicts.jsonl" ] && [[ "$name" != *-codev ]] && \
    $PY scripts/vote.py "results/scores/$name.verdicts.jsonl" --workers 4
  [ -s "results/raw/$name.repaired.jsonl" ] && $PY scripts/score.py "results/raw/$name.repaired.jsonl" --workers 4
done
votes=()                                      # in the order given, so reruns match
for name in "$@"; do
  [ -s "results/scores/$name.vote_tasks.jsonl" ] && votes+=("results/scores/$name.vote_tasks.jsonl")
done
[ ${#votes[@]} -gt 0 ] && $PY scripts/compare.py "${votes[@]}" --out results/scores/pooled.compare.json
[ -s results/scores/qwen3-8b-codev.verdicts.jsonl ] && $PY scripts/calibrate.py results/scores/qwen3-8b-codev.verdicts.jsonl
$PY scripts/report.py
