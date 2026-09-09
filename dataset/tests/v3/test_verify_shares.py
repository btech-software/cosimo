"""Axes 6-8 measured: the arithmetic of a mix, certified only at support.

These three axes are properties of the *slice*, not of a row -- so their tests
are statistics made on synthetic corpora of known composition, not renders.
The synthetic rows wear the shape the measurement reads (record_type,
work_type, scenario_id, answer) and nothing more: the per-row axes will
grumble about their absence elsewhere, and this file asserts only on its own
three axes, which is exactly how the board separates them.

Two properties the suite must pin down, because both were argued, not obvious:

* a slice **below support reports and never shades red** -- a gate that
  certifies noise is a gate somebody switches off;
* a slice **at support certifies both ways** -- in-band is green is not the
  same proposition as out-of-band is red, and both directions need a test.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipelines.v3 import config, verify_v3, write  # noqa: E402

WORK_TYPES = ("valuation.equity.dcf", "risk.market.var_es")
FAMILIES = (
    "mature_consumer",
    "cyclical_industrial",
    "fade_required",
    "rates_book",
    "equity_longonly",
    "credit_focused",
    "global_equity_long",
    "us_small_cap",
    "em_multi_asset",
    "large_cap_intraday",
)

_EXAM_ANSWER = "Work the strip. FINAL ANSWER: A -- 100.00 $M"
_LITURGICAL = (
    "ASSUMPTIONS: the pack's own.\nStep 1. pv 50.00.\nFINAL ANSWER: A -- 100.00 $M"
)
_PROSE_ANSWER = "The read is patient and the figures are the pack's own."


def _row(kind, work_type, family, serial, answer=_PROSE_ANSWER):
    return {
        "id": f"{config.SUPERVISED_ID_PREFIX}_{kind}_{serial:016d}",
        "record_type": kind,
        "work_type": work_type,
        "scenario_id": f"{work_type}.{family}",
        "variant": serial,
        "question": "q",
        "answer": answer,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": answer},
        ],
        "register": "desk_chat",
        "verification": {"computed_by": "synthetic", "pack_seed": "0" * 16},
    }


def _corpus(tmp_path, name, rows):
    out = os.path.join(str(tmp_path), name)
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["record_type"], []).append(row)
    for kind, kind_rows in by_kind.items():
        write.append_unique(write.path_for("sft", kind, out), kind_rows)
    return out


def _shares(out):
    report = verify_v3.verify_dir(
        out,
        kinds=tuple(
            sorted(
                {
                    "analysis",
                    "memo",
                    "critique",
                    "grounded",
                    "abstention",
                    "agentic",
                    "exam",
                }
            )
        ),
    )
    return {
        name: report["axes"][name]
        for name in ("exam liturgy share", "family share", "exam share")
    }


def _mix(total, exam_count, liturgical_count=0):
    """``total`` rows over all ten families, ``exam_count`` of them exam rows,
    ``liturgical_count`` of those carrying the ceremony's marks.

    Round-robin families and work types, and the exam seats spread evenly by
    stride, so no accident of construction tips a share axis: the numbers the
    tests assert are the numbers the fixtures intended, ±1 from integer
    division and nothing more.
    """
    is_exam = [False] * total
    if exam_count:
        stride = total / exam_count
        for seat in range(exam_count):
            is_exam[min(total - 1, int(seat * stride))] = True
    rows, seen_exam = [], 0
    for index in range(total):
        serial = index + 1
        family = FAMILIES[index % len(FAMILIES)]
        work_type = WORK_TYPES[index % len(WORK_TYPES)]
        if is_exam[index]:
            seen_exam += 1
            answer = _LITURGICAL if seen_exam <= liturgical_count else _EXAM_ANSWER
            rows.append(_row("exam", work_type, family, serial, answer))
        else:
            rows.append(_row("analysis", work_type, family, serial))
    return rows


def test_a_representative_slice_certifies_and_ships_green(tmp_path):
    """At support, in-band is green: 17% exam, one fifth liturgical, families
    balanced -- every share within its bound, the whole corpus quiet."""
    rows = _mix(1000, 170, liturgical_count=34)
    axes = _shares(_corpus(tmp_path, "clean", rows))
    assert axes["exam share"]["failures"] == []
    assert axes["exam liturgy share"]["failures"] == []
    assert axes["family share"]["failures"] == []
    assert "note" not in axes["exam share"], "support was reached: certify"
    assert axes["exam share"]["checked"] >= config.SHARE_MIN_ROWS


def test_an_exam_heavy_slice_shades_red_where_v1_collapsed(tmp_path):
    """33% exam rows is the v1 shape; at support the band must bite, loudly."""
    rows = _mix(1000, 330)
    axes = _shares(_corpus(tmp_path, "heavy", rows))
    assert axes["exam share"]["failures"], "a third exam corpus must not pass"
    assert any(
        "outside the band" in f["problem"] for f in axes["exam share"]["failures"]
    )


def test_an_exam_starved_slice_shades_red_too(tmp_path):
    """The band is a band: 6% is as much a drift from the plan as 33% --
    certified (60 ≥ the 50-row floor) and out of band, so red."""
    rows = _mix(1000, 60)
    axes = _shares(_corpus(tmp_path, "starved", rows))
    assert axes["exam share"]["failures"], (
        "a 4% exam slice is the collapse run backwards; the band binds both ways"
    )


def test_the_liturgy_cap_counts_markers_not_the_flag(tmp_path):
    """30% of exam answers carry the ceremony's marks while every row *claims*
    prose: the measurement reads the text, so the cap bites anyway."""
    rows = _mix(1000, 200, liturgical_count=60)
    for row in rows:
        render = row["verification"].setdefault("render", {})
        render["liturgy"] = False  # the stored flag lies; the markers do not
    axes = _shares(_corpus(tmp_path, "liturgy", rows))
    assert axes["exam liturgy share"]["failures"], (
        "a lying flag must not launder a liturgical slice"
    )
    assert any("cap" in f["problem"] for f in axes["exam liturgy share"]["failures"])


def test_one_stem_dominating_the_field_is_the_v2_pathology(tmp_path):
    """130 rows on one stem against 100 on the rest is a ratio of 1.30 -- over
    the balance tolerance, and the axis says so by naming the dominator."""
    rows = _mix(1000, 170)
    serial = 10_000
    work_type = WORK_TYPES[0]
    for _ in range(31):
        serial += 1
        rows.append(_row("analysis", work_type, FAMILIES[0], serial))
    axes = _shares(_corpus(tmp_path, "dominant", rows))
    assert axes["family share"]["failures"], "the 1,000x71 pathology must not pass"
    assert any("dominating" in f["problem"] for f in axes["family share"]["failures"])


@pytest.mark.parametrize(
    "total,exam_count",
    [(config.SHARE_MIN_ROWS - 1, 60), (900, config.SHARE_MIN_EXAM_ROWS - 1)],
)
def test_a_thin_slice_reports_and_never_shades_red(tmp_path, total, exam_count):
    """Below support the share axes state what they measured and withhold the
    verdict: notes, not failures -- even for a mix that would fail at support.

    The parametrised pairs each break exactly one floor: a corpus one row shy
    of ``SHARE_MIN_ROWS`` with a certified-size exam slice, and a full-size
    corpus whose exam slice is one row short of ``SHARE_MIN_EXAM_ROWS``. The
    family axis keeps its own counsel in these corpora (balanced and, in the
    second case, at support) -- this test is about the two exam-driven axes
    and the promise that noise never shades red.
    """
    rows = _mix(total, exam_count, liturgical_count=exam_count)  # would-be-red
    axes = _shares(_corpus(tmp_path, "thin", rows))
    for name in ("exam share", "exam liturgy share"):
        assert axes[name]["failures"] == [], f"{name} certified below support"
        assert "note" in axes[name], f"{name} must say it could not certify"
