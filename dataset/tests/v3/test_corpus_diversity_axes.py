"""Axes 15 and 16: the two failures the numeric gates are blind to by design.

Every gate on the board before these reads *one row against its own pack*, and
a repainted scenario satisfies all of them. Its pack recomputes, because the
pack is real. Its figures are the pack's, because they came off the same fact
computer. Its share of the corpus is one row like any other. The only thing
wrong with it is that the corpus already contains it twenty times -- which is a
statement about a *pair*, and nothing row-shaped can make it.

Register collapse is the same shape one level up: four voices written as one
voice is a property of the slice, and no single row in it is wrong.

So both are measured after the rows are read, and both are tested here by
planting the failure rather than by asserting on a clean corpus -- a gate
nobody has watched fire catches nothing.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3 import config  # noqa: E402
from pipelines.v3.packs import COMPUTERS, FAMILIES, PackError, compute_pack  # noqa: E402
from pipelines.v3.verification.register import (  # noqa: E402
    REGISTER_MIN_SEPARATION,
    profile_distance,
    register_profile,
)
from pipelines.v3.verify_v3 import (  # noqa: E402
    _measure_corpus_near_dup,
    _measure_register_separation,
)

#: Two passages that differ the way two registers should.
TERSE = "Cost is 12 bp. Work it patiently. The printable mid is not achievable."
MEMO = (
    "Finding: the book is long duration against its policy weight and the carry "
    "no longer compensates for the convexity being given up over this horizon. "
    "Evidence: the bridge puts the whole of the gap in allocation rather than "
    "selection, which is a mandate question and not a stock-picking one. "
    "Our call is to trim the overweight into the next print."
)


def _board(axis: str) -> dict:
    return {axis: {"checked": 0, "failures": [], "note": None}}


def _rows(texts, *, work_type="w", kind="analysis", register="desk_chat", start=0):
    return [
        {
            "id": f"cosimov3_{kind}_{start + i:016x}",
            "record_type": kind,
            "work_type": work_type,
            "register": register,
            "answer": text,
        }
        for i, text in enumerate(texts)
    ]


# --------------------------------------------------------------------------
# axis 15: repaint
# --------------------------------------------------------------------------


def test_distinct_answers_in_one_cell_are_not_flagged():
    rows = _rows(
        [f"The cost is {n} bp and the schedule is the risk." for n in range(12)]
    )
    board = _board("corpus near-dup")
    _measure_corpus_near_dup(board, rows)
    axis = board["corpus near-dup"]
    assert axis["failures"] == []
    assert axis["checked"] == 12
    assert "at or over threshold" in axis["note"]


def test_the_v1_pathology_is_caught_one_number_swapped():
    """ "Same paragraph, different last number" -- what v1 shipped 71,000 of."""
    body = (
        "Square-root impact scaling puts the cost at 28.16 bp against the arrival "
        "benchmark, with participation measured against ADV and the schedule "
        "itself becoming the binding risk past the cap we assume here today."
    )
    rows = _rows(
        [body] * 4
        + [body.replace("28.16", "28.17")] * 4
        + ["quite different prose entirely, about other things"] * 4
    )
    board = _board("corpus near-dup")
    _measure_corpus_near_dup(board, rows)
    assert board["corpus near-dup"]["failures"], "a repaint must be a finding"
    assert "repaint" in board["corpus near-dup"]["failures"][0]["problem"]


def test_a_thin_cell_reports_rather_than_certifies():
    """The share axes' discipline: below support, measure and say so."""
    rows = _rows(["identical prose about the book"] * (config.NEAR_DUP_MIN_ROWS - 1))
    board = _board("corpus near-dup")
    _measure_corpus_near_dup(board, rows)
    axis = board["corpus near-dup"]
    assert axis["failures"] == [], "a cell too thin to judge must not shade red"
    assert "not swept" in axis["note"]


def test_the_implementation_lane_is_exempt_because_its_answer_must_repeat():
    """Its answer is the reviewed reference instrument, identical by contract.

    The board re-executes those exact bytes; a lane that wrote a different
    instrument per variant would be a lane nobody reviewed. Its diversity is in
    the suites and the dirty fixture, which the hidden-test axis proves.
    """
    rows = _rows(["def solve(inputs):\n    return {}"] * 12, kind="implementation")
    board = _board("corpus near-dup")
    _measure_corpus_near_dup(board, rows)
    assert board["corpus near-dup"]["failures"] == []
    assert board["corpus near-dup"]["checked"] == 0


def test_the_sweep_stays_inside_a_cell_and_does_not_cross_work_types():
    """Different questions should differ; comparing them costs a quadratic
    sweep over the whole corpus to learn nothing. The repaint this axis exists
    to catch lives between variants of one family and between families of one
    computer -- both inside the cell.

    Counted, not asserted by absence: two ten-row cells are 2 * C(10,2) = 90
    in-cell pairs, where the cross product would be C(20,2) = 190.
    """
    shared = (
        "The cost is twenty-eight basis points against the arrival benchmark and "
        "the schedule itself becomes the binding risk well before the cap."
    )
    alpha = _rows([shared] * 10, work_type="alpha", start=0)
    beta = _rows([shared] * 10, work_type="beta", start=100)
    board = _board("corpus near-dup")
    _measure_corpus_near_dup(board, alpha + beta)
    axis = board["corpus near-dup"]
    assert axis["failures"], "ten identical rows in a cell is a repaint"
    # Every flagged pair names two rows of the *same* cell. A cross-cell
    # comparison would pair an alpha id with a beta id, and none does.
    alpha_ids = {r["id"] for r in alpha}
    beta_ids = {r["id"] for r in beta}
    for finding in axis["failures"]:
        left = finding["id"]
        right = finding["problem"].split("against ")[1].split()[0]
        assert (left in alpha_ids) == (right in alpha_ids), (
            f"{left} was compared with {right} across work types"
        )
        assert left in alpha_ids or left in beta_ids
    # The report names only the first few; the count carries the rest.
    assert len(axis["failures"]) <= config.NEAR_DUP_MAX_REPORTED


# --------------------------------------------------------------------------
# axis 16: register collapse
# --------------------------------------------------------------------------


def test_registers_that_write_differently_are_reported_separated():
    rows = _rows([TERSE] * 10, register="desk_chat") + _rows(
        [MEMO] * 10, register="ic_memo"
    )
    board = _board("register separation")
    _measure_register_separation(board, rows)
    axis = board["register separation"]
    assert axis["note"].startswith("separated")
    assert axis["failures"] == []


def test_one_voice_under_three_labels_is_reported_as_collapse():
    """The evaluation's complaint, executable: "the field changes; the voice does not"."""
    rows = (
        _rows([MEMO] * 10, register="desk_chat")
        + _rows([MEMO] * 10, register="ic_memo")
        + _rows([MEMO] * 10, register="risk_committee")
    )
    board = _board("register separation")
    _measure_register_separation(board, rows)
    axis = board["register separation"]
    assert "read alike" in axis["note"]
    # Reported, never red: a collapsed register is a finding about the *briefs*,
    # and failing the board would block a publish on prose nobody has read.
    assert axis["failures"] == []


def test_separation_needs_two_registers_with_support():
    rows = _rows([TERSE] * 20, register="desk_chat")
    board = _board("register separation")
    _measure_register_separation(board, rows)
    assert "fewer than two registers" in board["register separation"]["note"]


def test_the_profile_separates_prose_and_does_not_separate_a_voice_from_itself():
    near, far = register_profile([TERSE] * 8), register_profile([MEMO] * 8)
    assert profile_distance(near, far) >= REGISTER_MIN_SEPARATION
    assert profile_distance(near, register_profile([TERSE] * 8)) == 0.0


# --------------------------------------------------------------------------
# the root cause the axes exist to watch: scenario diversity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("work_type", sorted(COMPUTERS))
def test_a_family_does_not_repaint_one_entity_across_its_variants(work_type):
    """Entity names cycle over a pool rather than repeating one literal.

    Measured before the fix: 262 packs drew on 46 distinct names and
    ``US Small Cap Core`` was the book of all twenty ``us_small_cap``
    variants. Same three sectors, same mandate, different arithmetic -- which
    is v1's "71 stems x 1,000 paints" at a smaller constant, and exactly what
    axis 15 above cannot repair, only report.
    """
    for family in FAMILIES[work_type]:
        names = []
        for variant in range(12):
            try:
                pack = compute_pack(work_type, family, variant)
            except PackError:
                continue  # the guard's verdict; that coordinate ships nothing
            names.extend(e.get("name") for e in pack.entities if e.get("name"))
        if len(names) < 6:
            continue  # a family the guard thinned past measuring
        distinct = len(set(names))
        assert distinct >= len(names) * 0.75, (
            f"{work_type}/{family}: {distinct} distinct entities across "
            f"{len(names)} variants -- the scenario is being repainted"
        )
