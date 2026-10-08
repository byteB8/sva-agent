#!/usr/bin/env python3
"""Compare this scorer with published Jasper scores on the same tasks.

    scripts/calibrate.py results/scores/qwen3-8b-codev.verdicts.jsonl

Reads the verdicts of a model run on the CodeV-SVA authors' corrected suites,
computes Func@1 per suite with a 95% bootstrap interval over tasks, and sets it
beside the Func@1 those authors published for the same model (Jasper, all
tasks). Writes results/scores/calibration.json.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# Qwen3-8B, Func@1, from the CodeV-SVA model card (wyt2000/CodeV-SVA-14B).
PUBLISHED = {"human_codev": 32.3, "machine_codev": 46.1}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("verdicts")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per_task = defaultdict(list)
    for r in map(json.loads, open(args.verdicts)):
        per_task[r["key"]].append(r["equivalent"])

    rng = random.Random(args.seed)
    out = {}
    for suite, published in PUBLISHED.items():
        acc = [sum(v) / len(v) for k, v in sorted(per_task.items()) if k.startswith(suite + "/")]
        if not acc:
            continue
        n = len(acc)
        mean = 100 * sum(acc) / n
        boot = sorted(100 * sum(acc[rng.randrange(n)] for _ in range(n)) / n
                      for _ in range(args.resamples))
        lo, hi = boot[int(0.025 * args.resamples)], boot[int(0.975 * args.resamples) - 1]
        out[suite] = {"tasks": n, "func_at_1": round(mean, 1), "ci95": [round(lo, 1), round(hi, 1)],
                      "published": published}
        print(f"{suite}: Func@1 {mean:.1f}, 95% CI [{lo:.1f}, {hi:.1f}], published {published} "
              f"({'inside' if lo <= published <= hi else 'outside'} the interval)")
    path = Path(args.verdicts).resolve().parent / "calibration.json"
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
