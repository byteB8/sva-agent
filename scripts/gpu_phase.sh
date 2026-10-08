#!/bin/bash
# GPU phase: generation and repair for each model, back to back, so the GPUs
# are released as early as possible. Scoring and voting come later
# (scripts/cpu_phase.sh). A step whose output already exists is skipped, so the
# script can be rerun after an interruption.
#
#   scripts/gpu_phase.sh [--after-pid PID] <hf-model>=<name>=<samples per task> ...
#
# Extra vLLM options for every model in the call go in $GEN_ARGS, e.g.
#   GEN_ARGS="--max-num-seqs 64 --gpu-mem 0.93" scripts/gpu_phase.sh ...
# $SUITES picks the task suites (default: FVEval's human and machine), and
# SKIP_REPAIR=1 skips the repair step, e.g. for calibration runs.
set -u
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--after-pid" ]; then          # wait for a generation already running
  while kill -0 "$2" 2>/dev/null; do sleep 30; done
  shift 2
fi
export EBMC=${EBMC:-$HOME/opt/src/hw-cbmc/src/ebmc/ebmc}
export PATH=$PATH:$HOME/opt/spot/bin          # appended: Spot's env also ships a python3
export VLLM_USE_FLASHINFER_SAMPLER=0          # PyTorch sampler; FlashInfer's needs a JIT build
export HF_HUB_OFFLINE=1                       # models are cached; do not depend on the network
PY=$HOME/env/vllm/bin/python
RUN="taskset -c 0-3"
mkdir -p results/raw logs

for spec in "$@"; do
  IFS== read -r model name n <<< "$spec"
  if [ ! -s "results/raw/$name.jsonl" ]; then
    $RUN $PY scripts/generate.py --model "$model" --tp 2 --n "$n" --scorable ${GEN_ARGS:-} \
        --suites ${SUITES:-human machine} \
        --out "results/raw/$name.jsonl" > "logs/$name.gen.log" 2>&1 || echo "generation failed: $name"
  fi
  if [ -z "${SKIP_REPAIR:-}" ] && [ -s "results/raw/$name.jsonl" ] && [ ! -s "results/raw/$name.repaired.jsonl" ]; then
    $RUN $PY scripts/repair.py --model "$model" --tp 2 --scorable ${GEN_ARGS:-} --samples "results/raw/$name.jsonl" \
        --out "results/raw/$name.repaired.jsonl" > "logs/$name.repair.log" 2>&1 || echo "repair failed: $name"
  fi
  echo "$(date '+%F %T') GPU work done: $name"
done
echo "$(date '+%F %T') GPU_PHASE_DONE"
