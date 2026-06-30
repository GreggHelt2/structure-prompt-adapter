"""Unit tests for the contig→motif-index parser (``spa.eval.generate``, dev 14 §1).

Pure-function (no RFD3, no CUDA, no scoring deps — ``generate`` imports RFD3 lazily), so this always
runs. It pins the design-frame indices the Run-B hard⊕soft eval derives from a contig — the SPA
prompt-mask (§2) and ``score.motif_rmsd`` both consume them, so a wrong index here silently
mis-masks/mis-scores the motif.
"""

import pytest

from spa.eval.generate import _parse_contig_motif_indices as idx


def test_single_segment():
    # "59,A60-71,79": 12-res motif at design indices 59..70; total length 59+12+79 = 150
    assert idx("59,A60-71,79") == list(range(59, 71))


def test_1ctt_multi_single_residue():
    # the 1CTT spec (dev 12 §4): four single-residue islands -> [74, 76, 101, 104] (total 180)
    assert idx("74,A102,1,A104,24,A129,2,A132,75") == [74, 76, 101, 104]


def test_leading_motif_no_gap():
    assert idx("A1-5,20") == [0, 1, 2, 3, 4]


def test_multi_segment():
    # 10 gap, A11-13 (3), 5 gap, A19-20 (2), 30 gap
    assert idx("10,A11-13,5,A19-20,30") == [10, 11, 12, 18, 19]


def test_whitespace_tolerant():
    assert idx(" 59 , A60-71 , 79 ") == list(range(59, 71))


def test_rejects_variable_gap():
    with pytest.raises(ValueError):
        idx("10-20,A21-25,30")      # variable 'min-max' gap -> design length undefined


def test_rejects_no_motif():
    with pytest.raises(ValueError):
        idx("100")                  # no motif segment at all


def test_rejects_junk_token():
    with pytest.raises(ValueError):
        idx("59,A60-71,x,79")       # unparseable token


def test_parse_contig_motif_tuples_and_length():
    # (design_index, chain, author_resid) per motif residue + total design length (review #1 / dev 14 §3).
    from spa.eval.generate import _contig_length, _parse_contig_motif

    # 1CTT: design idx [74,76,101,104]; chain 'A'; AUTHOR resids [102,104,129,132]; total length 180.
    parsed = _parse_contig_motif("74,A102,1,A104,24,A129,2,A132,75")
    assert [d for d, _, _ in parsed] == [74, 76, 101, 104]
    assert [c for _, c, _ in parsed] == ["A", "A", "A", "A"]
    assert [r for _, _, r in parsed] == [102, 104, 129, 132]      # author numbers, NOT design indices
    assert _contig_length("74,A102,1,A104,24,A129,2,A132,75") == 180

    # range token A60-71 -> 12 residues; design idx 59..70 ↔ author resids 60..71; total 150.
    parsed2 = _parse_contig_motif("59,A60-71,79")
    assert [d for d, _, _ in parsed2] == list(range(59, 71))
    assert [r for _, _, r in parsed2] == list(range(60, 72))
    assert _contig_length("59,A60-71,79") == 150
    # and _parse_contig_motif_indices is just the design-index projection
    assert idx("59,A60-71,79") == [d for d, _, _ in parsed2]
