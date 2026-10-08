#!/usr/bin/env python3
"""Score generated assertions with the EBMC checker.

    scripts/score.py results/raw/qwen3-8b.jsonl [--workers 3] [--mode free]

Scores every sample of every scorable task: one whose reference passed the
checker validation (equivalent to itself) and uses no liveness operator, which
EBMC cannot compare (see sva_agent/checker.py). Identical answers to the same
task are checked once. Writes per-sample verdicts and a summary next to the
input, under results/scores/.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sva_agent import checker, data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of P(at least one of k samples is correct), from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("samples")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mode", choices=["free", "context"], default="free")
    ap.add_argument("--bound", type=int, default=20)
    ap.add_argument("--fix-semicolon", action="store_true",
                    help="lenient: append a missing final ';' before checking")
    ap.add_argument("--validation", nargs="*", default=None,
                    help="checker validation reports (default: all in data.VALIDATIONS)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples)]
    tasks = data.tasks_for(r["key"] for r in rows)
    keep = data.scorable_keys(args.validation)
    rows = [r for r in rows if r["key"] in keep]
    for r in rows:
        r["answer"] = checker.final_answer(r["answer"])   # empty if reasoning was cut off

    unique = sorted({(r["key"], r["answer"]) for r in rows})
    start = time.monotonic()

    def run(item):
        key, answer = item
        t = tasks[key]
        return item, checker.check(t.testbench, t.reference, answer, mode=args.mode,
                                   bound=args.bound, fix_semicolon=args.fix_semicolon).as_dict()

    with ThreadPoolExecutor(args.workers) as pool:
        verdict = dict(pool.map(run, unique))
    seconds = time.monotonic() - start

    per_task = defaultdict(list)
    for r in rows:
        per_task[r["key"]].append(verdict[(r["key"], r["answer"])])

    # Answers that crash EBMC itself are counted as failures, like any answer
    # that does not elaborate, but reported so their effect can be bounded.
    crashes = sum("invariant violation" in (verdict[(r["key"], r["answer"])]["error"] or "")
                  for r in rows)
    summary = {"samples_file": args.samples, "mode": args.mode, "bound": args.bound,
               "fix_semicolon": args.fix_semicolon, "checker_crash_samples": crashes,
               "unique_answers": len(unique), "check_seconds": round(seconds, 1), "suites": {}}
    for suite in data.SUITES:
        ks = [k for k in per_task if k.startswith(suite + "/")]
        if not ks:
            continue
        n = min(len(per_task[k]) for k in ks)
        s = {"tasks": len(ks), "samples_per_task": n}
        for metric in ("syntax", "equivalent", "relaxed"):
            for k in sorted({1, n}):
                s[f"{metric}@{k}"] = round(100 * sum(
                    pass_at_k(n, sum(v[metric] for v in per_task[t][:n]), k) for t in ks) / len(ks), 1)
        summary["suites"][suite] = s

    out_dir = ROOT / "results" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (Path(args.samples).stem + ("" if args.mode == "free" else f"_{args.mode}")
            + ("_semicolon" if args.fix_semicolon else ""))
    (out_dir / f"{stem}.verdicts.jsonl").write_text("".join(
        json.dumps({**r, **verdict[(r["key"], r["answer"])], "raw": None}) + "\n" for r in rows))
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for suite, s in summary["suites"].items():
        print(suite, json.dumps(s))
    print(f"{len(unique)} unique answers checked in {seconds:.0f} s")


if __name__ == "__main__":
    main()
