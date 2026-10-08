#!/usr/bin/env python3
"""Validate the EBMC checker before trusting it with model output.

1. Self-equivalence: every reference must come out equivalent to itself. Tasks
   where it does not are ones EBMC cannot handle; they are excluded from
   scoring, and the exclusion list is written out.
2. Sensitivity: small edits that change a reference's meaning (|-> to |=>,
   ##1 to ##2, && to ||, == to !=, ...) must not come out equivalent.

    scripts/validate_checker.py [--suites human machine] [--workers 3] [--bound 20]
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sva_agent import checker, data  # noqa: E402

# (name, pattern, replacement): each applied once, at its first match.
MUTATIONS = [
    ("overlap_to_next", r"\|->", "|=>"),
    ("next_to_overlap", r"\|=>", "|->"),
    ("delay_1_to_2", r"##1\b", "##2"),
    ("and_to_or", r"&&", "||"),
    ("or_to_and", r"\|\|", "&&"),
    ("eq_to_neq", r"(?<![=!<>])==(?!=)", "!="),
    ("neq_to_eq", r"!=(?!=)", "=="),
    ("case_neq_to_eq", r"!==", "==="),
]


def mutants(reference: str):
    body = checker.extract_property(reference) or ""
    for name, pat, rep in MUTATIONS:
        if re.search(pat, body):
            yield name, reference.replace(body, re.sub(pat, rep, body, count=1), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suites", nargs="+", default=["human", "machine"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bound", type=int, default=20)
    ap.add_argument("--mode", choices=["free", "context"], default="free")
    ap.add_argument("--out", default="results/validation.json")
    args = ap.parse_args()

    tasks = [t for s in args.suites for t in data.load(s)]
    start = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        selfeq = dict(zip([t.key for t in tasks], pool.map(
            lambda t: checker.check(t.testbench, t.reference, t.reference, mode=args.mode, bound=args.bound), tasks)))
    usable = [t for t in tasks if selfeq[t.key].equivalent]

    jobs = [(t, name, m) for t in usable for name, m in mutants(t.reference)]
    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = list(pool.map(
            lambda j: checker.check(j[0].testbench, j[0].reference, j[2], mode=args.mode, bound=args.bound), jobs))
    elapsed = time.monotonic() - start

    report = {"mode": args.mode, "bound": args.bound, "seconds": round(elapsed, 1), "suites": {}}
    for suite in args.suites:
        st = [t for t in tasks if t.suite == suite]
        excluded = {t.key: selfeq[t.key].as_dict() for t in st if not selfeq[t.key].equivalent}
        js = [(j, v) for j, v in zip(jobs, verdicts) if j[0].suite == suite]
        missed = [{"task": j[0].key, "mutation": j[1], "verdict": v.as_dict()}
                  for j, v in js if v.equivalent]
        undecided = [{"task": j[0].key, "mutation": j[1]} for j, v in js
                     if v.syntax and (v.ref_implies_cand is None or v.cand_implies_ref is None)]
        report["suites"][suite] = {
            "tasks": len(st), "self_equivalent": len(st) - len(excluded),
            "excluded": excluded, "mutants": len(js),
            "mutants_detected": len(js) - len(missed),
            "missed": missed, "undecided": undecided,
        }
        print(f"{suite}: self-equivalent {len(st) - len(excluded)}/{len(st)}; "
              f"mutants detected {len(js) - len(missed)}/{len(js)} "
              f"({len(undecided)} undecided)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{elapsed:.0f} s; wrote {args.out}")


if __name__ == "__main__":
    main()
