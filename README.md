# sva-agent

Open LLMs writing SystemVerilog Assertions (SVA) from English, with an
open-source model checker as both the judge and the helper.

NVIDIA's [FVEval](https://github.com/NVlabs/FVEval) benchmark asks a model to
turn a natural-language requirement ("the arbiter's grant is one-hot") into an
SVA assertion, and scores it by **formal equivalence** with an expert's
reference assertion. Its official scorer needs Cadence Jasper, a commercial
tool, and its equivalence script ships encrypted. This project:

1. **Rebuilds the scorer on open-source tools** (the EBMC model checker, with
   Spot for temporal logic), and validates it before trusting it.
2. **Benchmarks open models** on it: general models against SVA specialists.
3. **Uses the checker to improve answers without the reference:** a repair
   loop that feeds compiler errors back to the model, and *semantic voting*,
   which groups sampled answers by proven meaning and picks the majority.

## Results

All numbers below come from [`results/summary.md`](results/summary.md), which
`scripts/report.py` generates. Func@1 is the expected fraction of tasks a
single sampled answer gets **proven equivalent** to the reference. All models
ran in thinking mode (T=0.8, top-p 0.95).

| Model | Func@1, human (62) | Func@1, machine (276) | after one repair round |
|---|---|---|---|
| Qwen3-8B | 22.2 | 30.7 | 23.2 / 35.2 |
| CodeV-SVA-8B (SVA specialist) | 40.5 | 59.9 | 41.1 / 61.1 |
| Qwen3.8-27B (FP8) | **41.9** | **63.9** | **44.4 / 66.1** |

- **The open-source scorer agrees with Jasper.** On the CodeV-SVA authors'
  corrected copy of FVEval, with their prompts, Qwen3-8B scores 34.6 / 49.3
  here. Their published Jasper numbers are 32.3 / 46.1, inside both of this
  scorer's 95% intervals ([26.8, 42.5] and [45.3, 53.2]).
- **The checker improves answers without seeing the reference.**
  - Picking a sample that compiles, then the largest class of mutually
    equivalent samples, adds +4.4 (human) and +7.4 (machine) points over a
    single sample (pooled over three models, p < 0.01).
  - On the machine suite most of that gain is the compile check itself.
  - Voting by proven meaning beats voting by identical text on both suites
    (+2.2, p = 0.015; +1.1, p = 0.023). It helps most where a model's
    samples disagree in meaning: Qwen3-8B averages 2.9 (human) and 1.8
    (machine) classes of mutually equivalent samples per task, the 27B
    model 1.1 with half as many samples, leaving a vote little to decide.
- **The 27B model's human-suite score is limited by the token budget.** It
  used up all 16,384 tokens on 20% of its human-suite samples (Qwen3-8B on
  none), and an answer cut off mid-reasoning counts as a failure.
- **Two benchmark bugs found:** a reference whose operator precedence makes it
  check half of its prompt, and synthetic testbenches that wire their signals
  together. The corrected copy also keeps tasks whose assertions are
  tautologies. See [`docs/findings.md`](docs/findings.md).

## How an answer is scored

A candidate assertion C and the reference R are compared with two
model-checking runs:

| Run | Question | Meaning when it holds |
|---|---|---|
| assume R, assert C | does R imply C? | C is no stronger than R |
| assume C, assert R | does C imply R? | C is no weaker than R |

Both holding means C and R accept exactly the same behaviours (FVEval's
"functionality"); one holding is its "relaxed" credit. As in FVEval's scorer,
only the property bodies are compared (clocking and `disable iff` stripped),
and every signal they mention is an independent free input. Details and the
deliberate differences are in [`sva_agent/checker.py`](sva_agent/checker.py).

## Validating the scorer

`scripts/validate_checker.py` runs two checks with EBMC 6.1:

- **Self-equivalence:** every reference must be equivalent to itself.
- **Sensitivity:** small edits that change a reference's meaning (`|->` to
  `|=>`, `##1` to `##2`, `&&` to `||`, `==` to `!=`, ...) must be caught.

| Suite | References self-equivalent | Meaning-changing edits caught |
|---|---|---|
| NL2SVA-Human (79) | 62 | 153 / 155 |
| NL2SVA-Machine (300) | 284 | 484 / 484 |

The two human-suite edits not caught are genuinely equivalent: one is a
tautology under the testbench's parameters, and the other exposes a precedence
bug in the reference itself (see [`docs/findings.md`](docs/findings.md)).
Tasks whose reference uses liveness operators are not scored, because EBMC
cannot compare two liveness properties (findings, again). That leaves **62
human and 276 machine tasks**.

## Layout

```
sva_agent/   checker (EBMC equivalence), data loaders, prompts
scripts/     validate_checker, generate and repair (vLLM), score, vote, compare, report,
             gpu_phase.sh / cpu_phase.sh (the two halves of a run)
tests/       unit tests for the checker
results/     validation reports, scores, summary.md (generated; raw samples are not committed)
docs/        findings
```

## Running it

Requirements: EBMC 6.1 or newer, Spot (`ltl2tgba`), Python 3.11, and for
generation vLLM on a GPU. `scripts/fetch_fveval.sh` fetches the benchmark at
the pinned commit.

```bash
scripts/fetch_fveval.sh                     # FVEval at the pinned commit
scripts/fetch_codev_sva.sh                  # corrected copy, for calibration only
scripts/validate_checker.py                 # scorer validation

# GPU half: generation and one repair round per model, back to back
scripts/gpu_phase.sh Qwen/Qwen3-8B=qwen3-8b=8
# CPU half: score, vote, report
scripts/cpu_phase.sh qwen3-8b
```

On the server used here (two RTX A5000s over PCIe, driver 550), vLLM needed
four workarounds, listed in [`docs/findings.md`](docs/findings.md).

FVEval is © NVIDIA, Apache-2.0; it is fetched, not redistributed.
