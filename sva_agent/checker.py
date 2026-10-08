"""Check LLM-written SystemVerilog assertions with the EBMC model checker.

For a task with testbench TB and reference assertion R, a candidate assertion
C is asked three questions:

    syntax   does TB elaborate with C inserted?
    R => C   if R holds on every cycle, does C hold too?  (C is no stronger than R)
    C => R   if C holds on every cycle, does R hold too?  (C is no weaker than R)

Each implication is one model-checking run with `assume property (premise)`
and `assert property (conclusion)`. Both holding means the two assertions
accept exactly the same behaviours: FVEval's "functionality" credit. Either
holding is its "relaxed" credit.

The implications are checked in one of two modes:

  free     (default; FVEval's semantics) The two property bodies are compared
           on their own. Clocking and `disable iff` are stripped, as FVEval's
           scorer does, and every signal the bodies mention becomes an
           independent free input with the width the testbench declares.
           Testbench parameters keep their values. (FVEval's encrypted Jasper
           script treats parameters as free variables too; a parameter is a
           constant of the design, so it is not copied here.)
  context  The assertions are compared inside the testbench, so signals keep
           the relations the testbench gives them. More lenient: assertions
           that differ only on values the testbench can never produce count
           as equivalent.

Both modes are bounded: "holds" means no counterexample within `bound` cycles.
FVEval's properties span a few cycles, but liveness properties (`s_eventually`,
`##[1:$]`, `strong`) are only weakly distinguished by a bounded check.
"""

import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass

EBMC = os.environ.get("EBMC", "ebmc")

_FENCE = re.compile(r"```(?:systemverilog|verilog|sv)?\s*\n(.*?)```", re.S | re.I)
_ASSERT = re.compile(r"\bassert\s+property\s*\(")
_MODULE = re.compile(r"\bmodule\s+(\w+)")

# Words that can appear in a property body but are not signals.
_KEYWORDS = set("""
posedge negedge edge disable iff not and or intersect within throughout first_match
if else s_eventually eventually strong weak until s_until until_with s_until_with
nexttime s_nexttime always s_always implies accept_on reject_on sync_accept_on
sync_reject_on case endcase default property sequence assert assume cover begin end
inside dist matches with type int integer logic bit signed unsigned const local var
null this new
""".split())


def final_answer(text: str) -> str:
    """A model's answer without its reasoning block.

    Empty if the reasoning was cut off (by the token limit) before an answer.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if text.lstrip().startswith("<think>"):
        return ""
    return text.strip()


def extract_code(response: str) -> str:
    """The SystemVerilog in a model response: the first fenced block, else all of it."""
    m = _FENCE.search(response)
    return (m.group(1) if m else response).strip()


def _balanced(text: str, start: int) -> int | None:
    """Index just past the parenthesis that closes the one opened before `start`."""
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_property(code: str) -> str | None:
    """The expression inside the first `assert property ( ... )`, without the parentheses."""
    m = _ASSERT.search(code)
    if not m:
        return None
    end = _balanced(code, m.end())
    return code[m.end():end - 1].strip() if end else None


def body(prop: str) -> str:
    """A property expression without its leading clocking event and `disable iff`."""
    text = prop.strip()
    while True:
        m = re.match(r"(@\s*\(|disable\s+iff\s*\()", text)
        if not m:
            return text
        end = _balanced(text, m.end())
        if end is None:
            return text
        text = text[end:].strip()


def identifiers(expr: str) -> set[str]:
    """Names an expression refers to: signals and parameters, not keywords or literals."""
    expr = re.sub(r"\d*\s*'[sS]?[bBoOdDhH]\s*[0-9a-fA-FxXzZ_?]+", " ", expr)  # 4'b1010
    expr = re.sub(r"'[01xXzZ]\b", " ", expr)                                   # '0 '1
    expr = re.sub(r"\$\w+", " ", expr)                                         # $past
    names = set(re.findall(r"(?<![\w$'])[A-Za-z_]\w*", expr))
    return {n for n in names if n not in _KEYWORDS}


_LIVENESS = re.compile(r"s_eventually|\beventually\b|strong\s*\(|s_until|s_nexttime|\$\s*\]")


def uses_liveness(prop: str) -> bool:
    """Whether a property uses liveness or unbounded operators.

    EBMC cannot decide an implication whose premise is a liveness property:
    with --buechi it ignores the assumed liveness and refutes even R => R,
    and with --liveness-to-safety it proves even false implications. Tasks
    whose reference is such a property are therefore not scored.
    """
    return bool(_LIVENESS.search(prop))


def top_module(testbench: str) -> str:
    return _MODULE.search(testbench).group(1)


def insert(testbench: str, code: str) -> str:
    """Testbench with `code` added just before its final endmodule."""
    i = testbench.rfind("endmodule")
    return testbench[:i] + "\n" + code + "\n" + testbench[i:]


# ----------------------------------------------------------------------------
# Testbench declarations, for building the free-signal model
# ----------------------------------------------------------------------------

_PARAM = re.compile(r"\b(?:parameter|localparam)\b[^;]*;", re.S)
_DECL = re.compile(
    r"\b(?:input|output|inout|wire|reg|logic|bit)\b"
    r"((?:\s+(?:wire|reg|logic|bit|signed|unsigned|var))*)"
    r"\s*(\[[^\]]*\])?\s*([^;]*?);", re.S)


_HEADER = re.compile(r"\bmodule\s+\w+\s*(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?\((.*?)\)\s*;", re.S)
_PORT_WORDS = {"input", "output", "inout", "wire", "reg", "logic", "bit", "signed", "unsigned", "var"}


def _header_ports(port_list: str) -> dict[str, tuple[str, str]]:
    """Ports declared in an ANSI-style header: `input [3:0] a, b, output y`.

    A port without its own direction or type keeps the previous port's range,
    as in SystemVerilog. A header that only lists names (non-ANSI style)
    declares nothing here; those ports are declared in the body.
    """
    ports, rng, ansi = {}, "", False
    for item in re.split(r",(?![^\[]*\])", port_list):
        item = item.strip()
        m = re.match(r"((?:\b(?:%s)\b\s*)*)(\[[^\]]*\])?\s*([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$"
                     % "|".join(_PORT_WORDS), item)
        if not m:
            continue
        if m.group(1).strip() or m.group(2):
            ansi = True
            rng = m.group(2) or ""
        if ansi:
            ports[m.group(3)] = (rng, m.group(4).strip())
    return ports


def declarations(testbench: str):
    """(parameter statements, parameter names, {signal: (packed range, unpacked dims)})."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", " ", testbench, flags=re.S)
    params = _PARAM.findall(text)
    params_names = {n for p in params for n in re.findall(r"(\w+)\s*=", p)}
    signals = {}
    header = _HEADER.search(text)
    if header:
        signals.update(_header_ports(header.group(1)))
        text = text[:header.start()] + text[header.end():]      # body declarations only below
    for m in _DECL.finditer(text):
        rng = m.group(2) or ""
        for item in re.split(r",(?![^\[]*\])", m.group(3)):
            item = item.split("=", 1)[0].strip()
            nm = re.match(r"([A-Za-z_]\w*)\s*((?:\[[^\]]*\]\s*)*)$", item)
            if nm and nm.group(1) not in params_names:
                signals.setdefault(nm.group(1), (rng, nm.group(2).strip()))
    return params, params_names, signals


def free_model(testbench: str, premise: str, conclusion: str) -> str | None:
    """A module where both bodies' signals are independent inputs, or None if one is undeclared."""
    params, param_names, signals = declarations(testbench)
    names, decls = ["clk"], ["input logic clk;"]
    for name in sorted(identifiers(premise) | identifiers(conclusion)):
        if name in param_names or name == "clk":
            continue
        if name not in signals:
            return None
        rng, dims = signals[name]
        names.append(name)
        decls.append(" ".join(f"input logic {rng} {name} {dims}".split()) + ";")
    # Non-ANSI ports, as in the testbenches: parameters are declared before the
    # port declarations whose widths use them.
    return (f"module fveval_free ({', '.join(names)});\n"
            + "\n".join(params) + "\n" + "\n".join(decls) + "\n"
            f"fveval_premise: assume property (@(posedge clk) {premise});\n"
            f"fveval_conclusion: assert property (@(posedge clk) {conclusion});\n"
            "endmodule\n")


# ----------------------------------------------------------------------------
# Running EBMC
# ----------------------------------------------------------------------------

def _first_error(output: str) -> str:
    lines = [l for l in output.splitlines() if re.search(r"error|unsupported", l, re.I)]
    if not lines:
        lines = output.strip().splitlines()[-1:] or ["no output"]
    return re.sub(r"/tmp/\S+?/check\.sv", "check.sv", lines[0].strip())[:300]


def _ebmc(sv: str, args, timeout: int):
    """Run EBMC on one SystemVerilog text: (returncode, output, json_result | None)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "check.sv")
        result = os.path.join(tmp, "result.json")
        with open(path, "w") as f:
            f.write(sv)
        try:
            # errors="replace": EBMC echoes source text in its messages and may cut
            # a non-ASCII character from a model's answer in half.
            proc = subprocess.run([EBMC, path, "--json-result", result, *map(str, args)],
                                  capture_output=True, text=True, errors="replace",
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout", None
        data = json.load(open(result)) if os.path.exists(result) else None
    return proc.returncode, proc.stdout + proc.stderr, data


def elaborates(sv: str, top: str, timeout: int) -> str | None:
    """None if EBMC parses and type-checks `sv`, else its first error (or "timeout")."""
    rc, out, _ = _ebmc(sv, ["--top", top, "--show-properties"], timeout)
    if rc is None:
        return "timeout"
    return None if rc == 0 else _first_error(out)


def _conclusion_holds(sv: str, top: str, bound: int, timeout: int, buechi: bool) -> bool | None:
    args = ["--top", top, "--bound", bound, "--property", f"{top}.fveval_conclusion"]
    if buechi:
        args.append("--buechi")
    _, _, data = _ebmc(sv, args, timeout)
    if not data:
        return None
    status = next((p["status"] for p in data["properties"]
                   if p["identifier"].endswith(".fveval_conclusion")), "")
    if status.startswith("PROVED"):
        return True
    if status.startswith("REFUTED"):
        return False
    return None


def implies(testbench: str, premise: str, conclusion: str, mode: str = "free",
            bound: int = 20, timeout: int = 60, buechi: bool = True) -> bool | None:
    """Does assuming `premise` on every cycle make `conclusion` hold, up to `bound` cycles?

    `premise` and `conclusion` are property expressions as written inside
    `assert property ( ... )`. None means undecided (error or timeout).
    """
    if mode == "free":
        sv = free_model(testbench, body(premise), body(conclusion))
        if sv is None:
            return None
        return _conclusion_holds(sv, "fveval_free", bound, timeout, buechi)
    code = (f"fveval_premise: assume property ({premise});\n"
            f"fveval_conclusion: assert property ({conclusion});")
    return _conclusion_holds(insert(testbench, code), top_module(testbench), bound, timeout, buechi)


@dataclass
class Verdict:
    syntax: bool                     # TB elaborates with the candidate
    ref_implies_cand: bool | None    # None: not decided (error or timeout)
    cand_implies_ref: bool | None
    error: str | None = None         # first EBMC error, for syntax failures

    @property
    def equivalent(self) -> bool:
        return bool(self.ref_implies_cand and self.cand_implies_ref)

    @property
    def relaxed(self) -> bool:
        return bool(self.ref_implies_cand or self.cand_implies_ref)

    def as_dict(self):
        return {**asdict(self), "equivalent": self.equivalent, "relaxed": self.relaxed}


def check(testbench: str, reference: str, response: str, mode: str = "free",
          bound: int = 20, timeout: int = 60, buechi: bool = True,
          fix_semicolon: bool = False) -> Verdict:
    """Score one model response against the reference assertion of a task.

    fix_semicolon: append the final `;` if the answer's code lacks one. Off by
    default: without it the code is not valid SystemVerilog, and FVEval's
    Jasper flow counts it as a syntax error too. The lenient score measures how
    much of a model's failure rate is that one formatting slip.
    """
    code = extract_code(response)
    if fix_semicolon and code and not code.rstrip().endswith(";"):
        code = code.rstrip() + ";"
    cand = extract_property(code)
    if cand is None:
        return Verdict(False, None, None, "no `assert property (...)` found in response")
    # Syntax: the testbench must elaborate with the response's own code.
    err = elaborates(insert(testbench, code), top_module(testbench), timeout)
    if err is not None:
        return Verdict(False, None, None, err)
    ref = extract_property(reference)
    kw = dict(mode=mode, bound=bound, timeout=timeout, buechi=buechi)
    return Verdict(True, implies(testbench, ref, cand, **kw), implies(testbench, cand, ref, **kw))
