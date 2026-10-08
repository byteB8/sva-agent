#!/usr/bin/env python3
"""Paired bootstrap comparison of answer-selection methods, per suite.

    scripts/compare.py results/scores/qwen3-8b.vote_tasks.jsonl [more files ...]

Each file holds one model's per-task outcomes (from scripts/vote.py). Tasks
are resampled with replacement. Given several files (models), a resampled task
brings all models' outcomes for it, so tasks shared between models are not
treated as independent. Each resample computes the difference in mean accuracy
between two methods on the same items. Reported: the observed difference, its
95% percentile interval, and the fraction of resamples in which the difference
is <= 0 (a one-sided bootstrap p-value).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PAIRS = [("semantic_vote", "compiling"), ("semantic_vote", "text_vote"),
         ("semantic_vote", "single"), ("compiling", "single")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("per_task", nargs="+")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="JSON report path (default: next to the single input)")
    args = ap.parse_args()

    # suite -> task -> list of rows (one per model)
    by_suite = defaultdict(lambda: defaultdict(list))
    for path in args.per_task:
        for r in map(json.loads, open(path)):
            suite, task = r["key"].split("/", 1)
            by_suite[suite][task].append(r)

    rng = random.Random(args.seed)
    report = {"inputs": args.per_task}
    for suite in sorted(by_suite):
        tasks = sorted(by_suite[suite])
        n = len(tasks)
        items = sum(len(by_suite[suite][t]) for t in tasks)
        samples = [[tasks[rng.randrange(n)] for _ in range(n)] for _ in range(args.resamples)]
        report[suite] = {"tasks": n, "items": items}
        for a, b in PAIRS:
            diff = {t: [r[a] - r[b] for r in by_suite[suite][t]] for t in tasks}
            observed = 100 * sum(sum(d) for d in diff.values()) / items
            boot = []
            for sample in samples:
                tot = sum(sum(diff[t]) for t in sample)
                cnt = sum(len(diff[t]) for t in sample)
                boot.append(100 * tot / cnt)
            boot.sort()
            lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
            p = sum(x <= 0 for x in boot) / len(boot)
            report[suite][f"{a} - {b}"] = {"diff": round(observed, 1),
                                           "ci95": [round(lo, 1), round(hi, 1)], "p_le_0": round(p, 4)}
            print(f"{suite:14s} {a:>14s} - {b:<10s} {observed:+5.1f} pts  "
                  f"95% CI [{lo:+5.1f}, {hi:+5.1f}]  p={p:.4f}  ({n} tasks, {items} items)")
    out = Path(args.out) if args.out else Path(args.per_task[0]).with_name(
        Path(args.per_task[0]).name.replace(".vote_tasks.jsonl", ".compare.json"))
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
