"""Build FVEval's NL2SVA prompts, zero-shot, exactly as its launcher does.

The prompt strings live in FVEval's `fv_eval/prompts_nl2sva_*.py`. They are
read by parsing those files (`ast`), not by importing them, so no code from
the downloaded benchmark is executed.
"""

import ast
from functools import cache

from .data import FVEVAL, Task


@cache
def _strings(suite: str) -> dict[str, str]:
    """Top-level string constants of FVEval's prompt module for a suite."""
    tree = ast.parse((FVEVAL / f"fv_eval/prompts_nl2sva_{suite}.py").read_text())
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, str):
                out[node.targets[0].id] = value
    return out


def messages(task: Task) -> list[dict]:
    """Chat messages for one task.

    FVEval suites get FVEval's system prompt and zero-shot user prompt; the
    corrected suites get the CodeV-SVA authors' prompts (below).
    """
    if task.suite.endswith("_codev"):
        return _codev_messages(task)
    s = _strings(task.suite)
    question = s["SVAGEN_QUESTION_PREAMBLE"] + task.prompt + "\n"
    if task.suite == "human":
        question += s["SVAGEN_QUESTION_POSTAMBLE"]
        user = "\n\n" + s["SVAGEN_TB_PREAMBLE"] + "\n" + task.testbench + "\n" + question
    else:
        # NL2SVA-Machine questions are about free signals; the testbench only
        # declares them, and FVEval does not show it to the model.
        question += s["SVAGEN_QUESTION_POSTAMBLE_ZERO_SHOT"]
        user = "\n" + question
    return [{"role": "system", "content": s["SVAGEN_HEADER"]},
            {"role": "user", "content": user}]


# ----------------------------------------------------------------------------
# The CodeV-SVA authors' prompts, for their corrected suites
# ----------------------------------------------------------------------------
# Reproduced from SVAClient/src/SVAClient/Prompter.py and Few_Shots.py in
# github.com/wyt2000/CodeV-SVA (commit 764cb3d), so that calibration runs see
# the prompts their published scores were measured with. Their human-suite
# agent always uses the sequential example and the tb_reset instruction; their
# machine prompt shows each task's testbench.
_CODEV_SYSTEM = ("You are an AI assistant tasked with formal verification of register transfer "
                 "level (RTL) designs.\nYour job is to translate a description of an assertion "
                 "to concrete SystemVerilog Assertion (SVA) implementation.\n")
_CODEV_HUMAN_EXAMPLE = """asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);"""
_CODEV_MACHINE_EXAMPLE = """assert property (@(posedge clk)
    (sig_A && sig_B) != 1'b1
);"""
_CODEV_TAIL = """Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
{example}
```
Answer:"""


def _codev_messages(task: Task) -> list[dict]:
    user = ("Here is the testbench to perform your translation:\n" + task.testbench
            + "\nQuestion: Create a SVA assertion that checks: " + task.prompt + "\n")
    if task.suite == "human_codev":
        user += "You should use `tb_reset` as the disable condition signal. "
        user += _CODEV_TAIL.format(example=_CODEV_HUMAN_EXAMPLE)
    else:
        user += _CODEV_TAIL.format(example=_CODEV_MACHINE_EXAMPLE)
    return [{"role": "system", "content": _CODEV_SYSTEM}, {"role": "user", "content": user}]
