# Findings

## About the benchmark

**A reference that checks half of its prompt.** Task
`fifo_1r1w_bypass_fifo_2` asks for an assertion "that the fifo output and read
data are consistent". Its reference is

```systemverilog
(!fifo_empty && rd_pop && (fifo_out_data != rd_data)) ||
(fifo_empty && rd_pop && wr_push && (wr_data != rd_data)) !== 1'b1
```

In SystemVerilog, `!==` binds tighter than `||`, so this parses as
`case1 || !case2`. Whenever case 1 happens (a pop from a non-empty FIFO with
mismatched data), the assertion is *true*: the first inconsistency it was
written to catch is never flagged. The checker found this while validating
itself. Changing `!=` to `==` inside case 1 produced an assertion it proved
equivalent, which is correct for the reference as written. An answer that
implements the prompt correctly is scored as not equivalent to this reference.

**Synthetic testbenches alias their signals.** In NL2SVA-Machine, `sig_F` to
`sig_J` are wired as copies of `sig_A` to `sig_E`
(`assign sig_I = sig_D;`). Compared inside the testbench, `sig_I` and
`sig_I || sig_D` are the same signal, so an in-testbench check accepts wrong
answers. FVEval's scorer treats every signal as independent, and so does
this one (the `free` mode). The `context` mode remains available, and is
documented as the more lenient check.

## About the open-source scorer

**Liveness cannot be compared with EBMC.** An implication between two liveness
properties (`s_eventually`, `strong(##[0:$] ...)`) needs the assumed property
honoured on infinite, lasso-shaped traces. Neither engine gets this right:

| Engine | R implies R | R implies a weaker property | weaker implies R (false) |
|---|---|---|---|
| `--buechi`, bounded | refuted | refuted | refuted |
| `--liveness-to-safety`, bounded | proved | proved | proved |

One refutes everything, even R ⇒ R; the other proves everything, including a
false implication. Tasks whose reference uses liveness operators (11 human,
12 machine) are therefore not scored. A liveness *premise* paired with a
safety conclusion is fine: a finite counterexample to the safety property can
always be extended to satisfy the assumed liveness.

**EBMC 6.0's `$onehot0` is wrong** (`$onehot0(4'b0111)` is true), as found in
the formal-interconnect project. Several FVEval references use `$onehot0`, so
this project uses EBMC built from source at commit `03eea66b63` (6.1
development), where it is fixed.

**Checking is cheap.** A full check (elaboration plus two bounded runs)
takes about 0.03 s on these free-signal models, so scoring thousands of
samples takes seconds, not hours.

## About running vLLM on a PCIe-only GPU server

Four separate problems had to be fixed before two RTX A5000s (driver 550,
CUDA 12.4, GPUs connected only through the CPU) would generate:

1. **CUDA 13 wheels.** vLLM's default wheel and its PyTorch need a newer
   driver ("the NVIDIA driver on your system is too old"). The fix was the
   `+cu129` vLLM wheel with `torch 2.13.0+cu129`, which runs on driver 550
   through CUDA minor-version compatibility.
2. **torchcodec built for CUDA 13.** vLLM imports it for audio, and its
   library fails to load (`libnvrtc.so.13`). vLLM catches `ImportError` there
   but not `OSError`. Removing it, since this project needs no audio, makes
   the import fail cleanly.
3. **FlashInfer JIT.** Top-p sampling triggered a FlashInfer kernel build
   that needs `ninja` and a matching CUDA toolkit. `VLLM_USE_FLASHINFER_SAMPLER=0`
   selects the PyTorch sampler.
4. **Custom all-reduce hangs.** With two-GPU tensor parallelism, generation
   hung with both GPUs at "100% utilisation" but only ~105 W (busy-waiting).
   Plain NCCL all-reduce works on this machine with and without
   peer-to-peer. vLLM's custom all-reduce, which needs direct GPU-to-GPU
   access, does not. `disable_custom_all_reduce=True` fixed it, and the GPUs
   then drew ~225 W while generating.

## About comparing with published scores

**The published scores are on a different version of the benchmark.** The
CodeV-SVA authors, whose model card reports the reference Func@1 numbers,
re-checked FVEval by hand ("correcting or removing erroneous tests") and
scored that version with Jasper. Compared with FVEval at commit `141afe7`:

| | Human | Machine |
|---|---|---|
| Tasks | 73 (FVEval: 79) | 283 (FVEval: 300) |
| Prompts reworded | 25 | 125 |
| References changed (of tasks with an unchanged prompt) | 6 of 48 | 51 of 158 |
| Testbenches changed | 23 | every task: one minimal testbench per task, signals independent and multi-bit where the prompt needs it |

Their prompts also differ. Their machine prompt shows the task's testbench,
which FVEval's zero-shot prompt does not. Their human prompt adds "You should
use `tb_reset` as the disable condition signal." Our scores on FVEval
therefore cannot be compared one-to-one with their numbers. A calibration run
on their version (suites `human_codev`, `machine_codev`, with their prompts)
does that comparison.

**On their version, this scorer reproduces their Jasper scores.** Qwen3-8B,
4 samples per task, scored strictly:

| Suite | Func@1 here (EBMC) | 95% CI (bootstrap over tasks) | Published (Jasper) |
|---|---|---|---|
| Human, corrected (57 scorable of 73) | 34.6 | [26.8, 42.5] | 32.3 |
| Machine, corrected (269 scorable of 283) | 49.3 | [45.3, 53.2] | 46.1 |

Both published values fall inside the intervals. Adding a missing semicolon
before checking moves this scorer *away* from Jasper (39.5 / 58.9), which
suggests Jasper also treats that slip as a syntax error. The residual
differences come from different samples, 4 samples per task, and the
liveness tasks this scorer leaves out. Their repository has no license file, so the data is
fetched by `scripts/fetch_codev_sva.sh` for local evaluation and not
redistributed.

Their verification server inserts the model's code into the testbench
verbatim, as FVEval does, and counts the answer as failed when Jasper reports
a syntax error. Answers are not repaired before checking in either pipeline.

**A missing semicolon explains most of the synthetic-suite gap.** 942 of
Qwen3-8B's 2,704 scored answers (34.8%) do not elaborate. The largest group,
407 answers, ends `assert property (...)` without the closing `;`, which is not
valid SystemVerilog. `scripts/score.py --fix-semicolon` appends it before
checking:

| Qwen3-8B, 8 samples | Func@1 strict | Func@1 with `;` added | Published (other version) |
|---|---|---|---|
| Human (62) | 22.2 | 23.4 | 32.3 |
| Machine (276) | 30.7 | 42.2 | 46.1 |

**Most other compile failures are the model's, not the checker's.** EBMC's
"CONVERSION ERROR"s on Qwen3-8B's answers are hallucinated functions
(`prev(x)`, `$xor`), wrongly capitalised names (`Sig_F` for `sig_F`;
SystemVerilog is case-sensitive), invented signals (`result`, `sig_J_next`)
and method calls (`sig_I.all()`). One construct does crash EBMC itself:
`$clog2` of a non-constant (7 samples). It is counted as a failure and reported
as `checker_crash_samples` in each score summary, so its effect stays visible
(at most 1.4 points of the human suite).

**Some tasks are tautologies.** In the counter tasks, the testbench sets
`width = 1`, `min = 0` and `max = 1`, so `count <= max` and `count >= min` can
never be false, and every edit of the reference is still equivalent to it. The
corrected version keeps these tasks and keeps the precedence bug described
above.

## About choosing among samples without the reference

Qwen3-8B, 8 samples per task. Each method picks one answer per task; the
table gives the fraction of tasks where that answer is equivalent to the
reference.

| | Random sample | Random compiling sample | Majority by exact text | Semantic vote | Any sample right |
|---|---|---|---|---|---|
| Human (62) | 22.2 | 24.7 | 22.6 | **27.4** | 51.6 |
| Machine (276) | 30.7 | 46.9 | 45.3 | **48.9** | 67.8 |

Paired bootstrap over tasks (10,000 resamples, `scripts/compare.py`):

| Difference | Human | Machine |
|---|---|---|
| semantic vote - text vote | +4.8, 95% CI [+0.0, +11.3], p = 0.043 | +3.6, [+1.1, +6.2], p = 0.003 |
| semantic vote - compiling sample | +2.7, [-2.8, +8.7], p = 0.17 | +2.1, [-0.5, +4.7], p = 0.060 |
| compiling sample - random sample | +2.5, [+1.3, +3.9], p < 0.001 | +16.2, [+13.6, +18.8], p < 0.001 |

Discarding answers that do not compile is the large, certain gain. Grouping
the rest by proven meaning beats grouping them by text on both suites. Its
edge over simply picking any compiling answer is not significant for one
model; results from the other models will add power.
