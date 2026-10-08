"""Load NL2SVA tasks.

Two sources:

* FVEval (NVIDIA, Apache-2.0), the benchmark itself: suites `human` (79 tasks
  on real designs) and `machine` (300 synthetic tasks).
  `scripts/fetch_fveval.sh` clones it at a pinned commit into `data/FVEval`.
* The CodeV-SVA authors' corrected copy of it: suites `human_codev` (73) and
  `machine_codev` (283), with reworded prompts, fixed references and a minimal
  testbench per synthetic task. Their published scores are on this version, so
  it is used to calibrate this scorer against Jasper.
  `scripts/fetch_codev_sva.sh` clones it into `data/CodeV-SVA`.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FVEVAL = ROOT / "data" / "FVEval"
CODEV = ROOT / "data" / "CodeV-SVA"

FVEVAL_SUITES = {
    "human": "data_nl2sva/data/nl2sva_human.csv",
    "machine": "data_nl2sva/data/nl2sva_machine.csv",
}
CODEV_SUITES = {
    "human_codev": "SVAClient/datasets/fveval_nl2sva_human.jsonl",
    "machine_codev": "SVAClient/datasets/fveval_nl2sva_machine.jsonl",
}
SUITES = [*FVEVAL_SUITES, *CODEV_SUITES]

# Checker validation reports (scripts/validate_checker.py); a task is scored
# only if its reference passed.
VALIDATIONS = [ROOT / "results" / "validation_free_ebmc61.json",
               ROOT / "results" / "validation_codev_ebmc61.json"]


@dataclass(frozen=True)
class Task:
    suite: str
    task_id: str         # unique within the suite
    design: str
    prompt: str          # natural-language description of the assertion
    reference: str       # reference SVA
    testbench: str       # SystemVerilog module the assertion is written against

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task_id}"


def load(suite: str) -> list[Task]:
    if suite in FVEVAL_SUITES:
        path = FVEVAL / FVEVAL_SUITES[suite]
        if not path.exists():
            raise SystemExit(f"{path} not found; run scripts/fetch_fveval.sh first")
        csv.field_size_limit(1 << 30)
        with open(path, newline="") as f:
            return [Task(suite, f"{r['design_name']}_{r['task_id']}", r["design_name"],
                         r["prompt"], r["ref_solution"], r["testbench"])
                    for r in csv.DictReader(f)]
    path = CODEV / CODEV_SUITES[suite]
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/fetch_codev_sva.sh first")
    with open(path) as f:
        return [Task(suite, r["name"], r["name"].rsplit("_", 1)[0], r["problem"],
                     r["ground_truth"], r["testbench"]) for r in map(json.loads, f)]


def tasks_for(keys) -> dict[str, Task]:
    """Task objects for the given task keys, loading only the suites they use."""
    suites = {k.split("/", 1)[0] for k in keys}
    return {t.key: t for s in suites for t in load(s)}


def scorable_keys(validations=None) -> set[str]:
    """Tasks the checker can score: the reference passed validation (equivalent
    to itself) and uses no liveness operator, which EBMC cannot compare."""
    from .checker import uses_liveness

    keys = set()
    for path in validations or VALIDATIONS:
        path = Path(path)
        if not path.exists():
            continue
        for suite, rep in json.loads(path.read_text())["suites"].items():
            keys |= {t.key for t in load(suite)
                     if t.key not in rep["excluded"] and not uses_liveness(t.reference)}
    return keys
