#!/usr/bin/env python3
"""Semantic voting: pick each task's answer by meaning, without the reference.

    scripts/vote.py results/scores/qwen3-8b.verdicts.jsonl [--workers 3]

For each task, the samples that elaborate are grouped into classes of
mutually equivalent assertions, using the same model checker that scores
them but comparing candidates only with each other. The largest class wins,
and its representative is the task's answer. That answer is then looked up in
the existing verdicts, which do compare it with the reference.

This is majority voting where votes are counted by proven meaning rather than
by string match: "a |=> b" and "a |-> ##1 b" vote together.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sva_agent import checker, data  # noqa: E402


def classes(testbench: str, answers: Counter, bound: int) -> list[tuple[str, int]]:
    """Group answers into equivalence classes: [(representative, votes)], largest first."""
    reps: list[list] = []          # [representative property, votes, representative answer]
    for answer, votes in answers.most_common():
        prop = checker.extract_property(checker.extract_code(answer))
        for c in reps:
            if (checker.implies(testbench, c[0], prop, bound=bound)
                    and checker.implies(testbench, prop, c[0], bound=bound)):
                c[1] += votes
                break
        else:
            reps.append([prop, votes, answer])
    reps.sort(key=lambda c: -c[1])
    return [(c[2], c[1]) for c in reps]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("verdicts")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bound", type=int, default=20)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.verdicts)]
    tasks = data.tasks_for(r["key"] for r in rows)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["key"]].append(r)

    def text_key(answer: str) -> str:
        """An answer's property with all whitespace removed: equal keys mean equal text."""
        prop = checker.extract_property(checker.extract_code(answer)) or answer
        return "".join(prop.split())

    def decide(key):
        samples = by_task[key]
        compiling = [r for r in samples if r["syntax"]]
        if not compiling:
            return {"key": key, "semantic": samples[0], "text": samples[0], "classes": 0,
                    "compiling_eq": 0.0}
        # Baseline 1: a random sample that compiles (uses only the syntax check).
        compiling_eq = sum(r["equivalent"] for r in compiling) / len(compiling)
        # Baseline 2: majority by exact text, among samples that compile.
        by_text = Counter(text_key(r["answer"]) for r in compiling)
        top_text = by_text.most_common(1)[0][0]
        text_pick = next(r for r in compiling if text_key(r["answer"]) == top_text)
        # Semantic voting: majority by proven meaning.
        ranked = classes(tasks[key].testbench, Counter(r["answer"] for r in compiling), args.bound)
        winner = ranked[0][0]
        return {"key": key, "semantic": next(r for r in compiling if r["answer"] == winner),
                "text": text_pick, "classes": len(ranked), "compiling_eq": compiling_eq}

    with ThreadPoolExecutor(args.workers) as pool:
        decided = list(pool.map(decide, sorted(by_task)))

    def pct(x: float) -> float:
        return round(100 * x, 1)

    summary = {"verdicts_file": args.verdicts, "suites": {}}
    for suite in data.SUITES:
        ds = [d for d in decided if d["key"].startswith(suite + "/")]
        if not ds:
            continue
        n = len(ds)
        summary["suites"][suite] = {
            "tasks": n,
            # expected accuracy of one random sample
            "single_sample_equivalent": pct(sum(
                sum(r["equivalent"] for r in by_task[d["key"]]) / len(by_task[d["key"]]) for d in ds) / n),
            # baselines that need no equivalence checking
            "compiling_sample_equivalent": pct(sum(d["compiling_eq"] for d in ds) / n),
            "text_vote_equivalent": pct(sum(d["text"]["equivalent"] for d in ds) / n),
            # semantic voting
            "voted_equivalent": pct(sum(d["semantic"]["equivalent"] for d in ds) / n),
            # upper bound: some sample is right
            "any_sample_equivalent": pct(sum(any(r["equivalent"] for r in by_task[d["key"]]) for d in ds) / n),
            "mean_classes": round(sum(d["classes"] for d in ds) / n, 2),
        }
        print(suite, json.dumps(summary["suites"][suite]))
    out = Path(args.verdicts).with_name(Path(args.verdicts).name.replace(".verdicts.jsonl", ".vote.json"))
    out.write_text(json.dumps(summary, indent=2) + "\n")
    # Per-task outcomes of every method, for paired significance tests (scripts/compare.py).
    per_task = out.with_name(out.name.replace(".vote.json", ".vote_tasks.jsonl"))
    per_task.write_text("".join(json.dumps({
        "key": d["key"],
        "single": sum(r["equivalent"] for r in by_task[d["key"]]) / len(by_task[d["key"]]),
        "compiling": d["compiling_eq"],
        "text_vote": float(d["text"]["equivalent"]),
        "semantic_vote": float(d["semantic"]["equivalent"]),
        "any": float(any(r["equivalent"] for r in by_task[d["key"]])),
    }) + "\n" for d in decided))


if __name__ == "__main__":
    main()
