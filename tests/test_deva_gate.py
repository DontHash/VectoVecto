"""
test_deva_gate.py — W1 gate metric definitions.

digit-exact is compared on the *digit sequence* of digit-bearing lines (so a
matra error does not count as a digit error, and a correct digit read with a
wrong word does not either). BagCER/CER are means over all GT-bearing lines.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from deva_crnn.gate import line_stats  # noqa: E402


def test_line_stats_digit_exact_ignores_nondigit_errors():
    gts = ["मिति २०८१-०४-२७", "धारा १२ को उपधारा"]
    hyps = ["मिति २०८१-०४-२७", "धारा १२ को उपधारा"]  # both perfect
    s = line_stats(gts, hyps)
    assert s["digit_lines"] == 2 and s["digit_exact"] == 1.0
    assert s["cer"] == 0.0

    hyps2 = ["मिति २०८१-०४-२८", "धारा १३ को उपधारा"]  # digits wrong
    s2 = line_stats(gts, hyps2)
    assert s2["digit_exact"] == 0.0

    hyps3 = ["मिति २०८१-०४-२७", "धारा १२ को उपधार"]  # word wrong, digits right
    s3 = line_stats(gts, hyps3)
    assert s3["digit_exact"] == 1.0
    assert s3["cer"] > 0.0


def test_line_stats_ignores_lines_without_digits_for_the_digit_bar():
    s = line_stats(["नेपाल सरकार"], ["नेपाल सरकार"])
    assert s["digit_lines"] == 0 and s["digit_exact"] is None
