#!/usr/bin/env python3
"""Generate SVA for FVEval NL2SVA tasks with a local model through vLLM.

    python scripts/generate.py --model Qwen/Qwen3-8B --suites human machine \
        --n 8 --out results/raw/qwen3-8b.jsonl [--tp 2] [--limit 5]

Writes one JSON line per sample: task key, sample index, the final answer
(text after any </think>), token count and finish reason. Seeds are derived
from the task key, so a rerun with the same settings reproduces the samples.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sva_agent import data, prompts  # noqa: E402
from sva_agent.checker import final_answer  # noqa: E402


def seed_for(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--suites", nargs="+", default=["human", "machine"])
    ap.add_argument("--n", type=int, default=8, help="samples per task")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks per suite (smoke tests)")
    ap.add_argument("--scorable", action="store_true",
                    help="only tasks the checker can score (saves GPU time)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="cap on concurrent sequences; hybrid models such as Qwen3.8 need one "
                         "linear-attention state block per sequence, so the vLLM default (256) "
                         "can exceed what fits")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    keep = data.scorable_keys() if args.scorable else None
    tasks = []
    for suite in args.suites:
        ts = [t for t in data.load(suite) if keep is None or t.key in keep]
        tasks += ts[:args.limit] if args.limit else ts

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, seed=0,
              **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
              # The GPUs on the target server meet only through the CPU (PCIe,
              # no NVLink); vLLM's custom all-reduce needs direct GPU-to-GPU
              # access and hangs there, so plain NCCL is used instead.
              disable_custom_all_reduce=True)
    params = [SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                             max_tokens=args.max_tokens, seed=seed_for(t.key)) for t in tasks]
    start = time.monotonic()
    outputs = llm.chat([prompts.messages(t) for t in tasks], params,
                       chat_template_kwargs={"enable_thinking": args.thinking == "on"})
    seconds = time.monotonic() - start

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w") as f:
        for t, o in zip(tasks, outputs):
            for i, c in enumerate(o.outputs):
                total += len(c.token_ids)
                f.write(json.dumps({
                    "key": t.key, "sample": i, "model": args.model,
                    "thinking": args.thinking, "temperature": args.temperature,
                    "answer": final_answer(c.text, c.finish_reason != "length"),
                    "tokens": len(c.token_ids),
                    "finish": c.finish_reason, "raw": c.text,
                }) + "\n")
    print(f"{len(tasks)} tasks x {args.n} samples, {total} output tokens in {seconds:.0f} s "
          f"({total / seconds:.0f} tok/s); wrote {out}")


if __name__ == "__main__":
    main()
