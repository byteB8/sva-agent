"""Unit tests for the text handling in sva_agent.checker (no EBMC needed),
plus end-to-end checks that run EBMC when it is available."""

import os
import shutil

import pytest

from sva_agent import checker

TB = """module tb (clk, reset_, a, b, v);
    parameter W = 4;
    localparam M = W - 1;
input clk;
input reset_;
input a, b;
input [W-1:0] v;
wire tb_reset;
assign tb_reset = (reset_ == 1'b0);
reg [M:0] q, r [0:3];
endmodule
"""


def test_final_answer_drops_reasoning_and_cut_off_outputs():
    assert checker.final_answer("<think>\nplan\n</think>\n\nassert property (a);") \
        == "assert property (a);"
    assert checker.final_answer("<think>\nplan, cut off") == ""
    # Templates that open <think> in the prompt: the output starts mid-reasoning,
    # and a draft in it is not the answer when generation hit the token limit.
    cut = "We need ... ```systemverilog\nassert property (a);\n``` ... maybe"
    assert checker.final_answer(cut, finished=False) == ""
    assert checker.final_answer("plan\n</think>\nassert property (b);", finished=False) \
        == "assert property (b);"
    assert checker.sample_answer({"raw": cut, "finish": "length", "answer": cut}) == ""
    assert checker.sample_answer({"raw": None, "answer": "assert property (a);"}) \
        == "assert property (a);"


def test_extract_code_prefers_fenced_block():
    assert checker.extract_code("text\n```systemverilog\nassert property (a);\n```\nmore") \
        == "assert property (a);"
    assert checker.extract_code("assert property (a);") == "assert property (a);"


def test_extract_property_handles_nesting_and_labels():
    code = "asrt: assert property (@(posedge clk) disable iff (tb_reset) (a && (b || c)) |-> d);"
    assert checker.extract_property(code) == \
        "@(posedge clk) disable iff (tb_reset) (a && (b || c)) |-> d"
    assert checker.extract_property("assume property (a);") is None
    assert checker.extract_property("assert property ((a);") is None


def test_body_strips_clocking_and_disable():
    assert checker.body("@(posedge clk) disable iff (tb_reset) a |=> b") == "a |=> b"
    assert checker.body("@(posedge clk)\n\t(sig_A && sig_B)") == "(sig_A && sig_B)"
    assert checker.body("a |-> b") == "a |-> b"


def test_identifiers_skip_keywords_literals_and_system_functions():
    ids = checker.identifiers("$onehot0(tb_gnt) && (cnt == 4'b1010) |-> s_eventually(x) ##1 '1")
    assert ids == {"tb_gnt", "cnt", "x"}


def test_declarations_resolve_widths_and_params():
    params, pnames, sigs = checker.declarations(TB)
    assert pnames == {"W", "M"}
    assert sigs["v"] == ("[W-1:0]", "")
    assert sigs["a"] == ("", "") and sigs["b"] == ("", "")
    assert sigs["q"] == ("[M:0]", "")
    assert sigs["r"] == ("[M:0]", "[0:3]")
    assert "tb_reset" in sigs


def test_free_model_declares_only_used_signals():
    sv = checker.free_model(TB, "a |-> v == W", "b")
    assert "input logic a;" in sv and "input logic [W-1:0] v;" in sv and "input logic b;" in sv
    assert "input logic q" not in sv
    assert checker.free_model(TB, "undeclared_signal", "a") is None


def test_liveness_detection():
    assert checker.uses_liveness("a |-> s_eventually b")
    assert checker.uses_liveness("a |-> strong(##[0:$] b)")
    assert checker.uses_liveness("a |-> ##[1:$] b")
    assert not checker.uses_liveness("a |-> ##[1:3] b")


needs_ebmc = pytest.mark.skipif(
    not (shutil.which(checker.EBMC) or os.path.exists(checker.EBMC)), reason="EBMC not available")


@needs_ebmc
def test_equivalent_rewrites_are_accepted():
    ref = "assert property (@(posedge clk) disable iff (tb_reset) a |=> b);"
    for cand in ["assert property (@(posedge clk) disable iff (tb_reset) a |-> ##1 b);",
                 "assert property (@(posedge clk) disable iff (tb_reset) !(a && !b) || 1'b1 |-> (a |=> b));"]:
        v = checker.check(TB, ref, cand)
        assert v.syntax
    v = checker.check(TB, ref, "assert property (@(posedge clk) disable iff (tb_reset) a |-> ##1 b);")
    assert v.equivalent


@needs_ebmc
def test_weaker_and_wrong_answers_are_told_apart():
    ref = "assert property (@(posedge clk) a |-> b);"
    weaker = checker.check(TB, ref, "assert property (@(posedge clk) a |-> (b || v[0]));")
    assert weaker.syntax and weaker.ref_implies_cand and not weaker.cand_implies_ref
    assert weaker.relaxed and not weaker.equivalent
    wrong = checker.check(TB, ref, "assert property (@(posedge clk) a |-> !b);")
    assert wrong.syntax and not wrong.relaxed
    broken = checker.check(TB, ref, "assert property (@(posedge clk) a |-> nonexistent);")
    assert not broken.syntax and broken.error


@needs_ebmc
def test_non_ascii_answers_do_not_crash_the_checker():
    # EBMC may echo a cut-off multi-byte character in its error message.
    v = checker.check(TB, "assert property (@(posedge clk) a |-> b);",
                      "assert property (@(posedge clk) a |-> b ≤ c → d);")
    assert not v.syntax and v.error


def test_ansi_header_ports_and_inherited_ranges():
    tb = """module testbench (
    input clk,
    input [3:0] sig_A, sig_B,
    input logic sig_C,
    output [1:0] y
);
endmodule
"""
    _, _, sigs = checker.declarations(tb)
    assert sigs["clk"] == ("", "")
    assert sigs["sig_A"] == ("[3:0]", "") and sigs["sig_B"] == ("[3:0]", "")
    assert sigs["sig_C"] == ("", "")
    assert sigs["y"] == ("[1:0]", "")


def test_non_ansi_header_lists_names_only():
    _, _, sigs = checker.declarations(TB)        # names in the header, declared in the body
    assert sigs["v"] == ("[W-1:0]", "")
