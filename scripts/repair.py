#!/usr/bin/env python3
"""Repair agent: feed the checker's error back to the model and ask again.

    python scripts/repair.py --model Qwen/Qwen3-8B --samples results/raw/qwen3-8b.jsonl \
        [--rounds 1] [--scorable] --out results/raw/qwen3-8b.repaired.jsonl [--tp 2]

Every sample whose answer does not elaborate with its testbench, or contains
no assertion at all, gets a follow-up turn quoting EBMC's error and asking for
a corrected assertion. Only the error is fed back, never the reference, so a
deployed tool could do the same. The output has the same rows as the input,
with failed answers replaced by the last repair attempt; score it with
scripts/score.py to measure the effect.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import seed_for  # noqa: E402
from sva_agent import checker, data, prompts  # noqa: E402
from sva_agent.checker import final_answer  # noqa: E402

FOLLOW_UP = """The SystemVerilog checker could not use your assertion:
{error}

Write a corrected assertion that checks the same requirement. Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.
Answer:"""


def syntax_error(task, answer: str, timeout: int = 60) -> str | None:
    code = checker.extract_code(answer)
    if checker.extract_property(code) is None:
        return "no `assert property (...)` was found in your answer"
    return checker.elaborates(checker.insert(task.testbench, code),
                              checker.top_module(task.testbench), timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--scorable", action="store_true",
                    help="only repair samples of tasks the checker can score")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="cap on concurrent sequences; hybrid models such as Qwen3.8 need one "
                         "linear-attention state block per sequence, so the vLLM default (256) "
                         "can exceed what fits")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.samples)]
    tasks = data.tasks_for(r["key"] for r in rows)
    keep = data.scorable_keys() if args.scorable else None
    convo = {}                        # row index -> conversation so far
    for i, r in enumerate(rows):
        r["answer"] = final_answer(r["answer"])
        r["repair_rounds"] = 0
        if keep is not None and r["key"] not in keep:
            continue
        err = syntax_error(tasks[r["key"]], r["answer"])
        if err is not None:
            convo[i] = prompts.messages(tasks[r["key"]]) + [
                {"role": "assistant", "content": r["answer"]},
                {"role": "user", "content": FOLLOW_UP.format(error=err)}]
    print(f"{len(convo)} of {len(rows)} samples fail to elaborate")

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, seed=0,
              **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
              # The GPUs on the target server meet only through the CPU (PCIe,
              # no NVLink); vLLM's custom all-reduce needs direct GPU-to-GPU
              # access and hangs there, so plain NCCL is used instead.
              disable_custom_all_reduce=True)
    start = time.monotonic()
    for rnd in range(1, args.rounds + 1):
        todo = sorted(convo)
        if not todo:
            break
        params = [SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                                 max_tokens=args.max_tokens,
                                 seed=seed_for(f"{rows[i]['key']}/{rows[i]['sample']}/{rnd}"))
                  for i in todo]
        outs = llm.chat([convo[i] for i in todo], params,
                        chat_template_kwargs={"enable_thinking": args.thinking == "on"})
        for i, o in zip(todo, outs):
            answer = final_answer(o.outputs[0].text)
            rows[i].update(answer=answer, repair_rounds=rnd)
            err = syntax_error(tasks[rows[i]["key"]], answer)
            if err is None:
                del convo[i]
            else:
                convo[i] += [{"role": "assistant", "content": answer},
                             {"role": "user", "content": FOLLOW_UP.format(error=err)}]
        print(f"round {rnd}: {len(todo) - len(convo)} of {len(todo)} repaired")

    Path(args.out).write_text("".join(json.dumps({**r, "raw": None}) + "\n" for r in rows))
    print(f"{time.monotonic() - start:.0f} s; {len(convo)} still failing; wrote {args.out}")


if __name__ == "__main__":
    main()
