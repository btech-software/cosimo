"""The v3.2 capture is what the amendment's acceptance criteria look like.

`live_v3_2.jsonl` is the answer to `live_three_registers.jsonl`: the same three
work types and the same three registers, rendered by the same teacher
(`deepseek-v4-flash-0731`, think off) against the v3.2 packs, briefs and gates.
The older file is kept because it fails; this one is kept because it passes,
and the pair is the whole argument for the amendment.

What is asserted here is §F's acceptance list, read off the rows -- plus the
two-surface split and the schema, as for any captured sample. Deliberately not
the full sixteen-axis board: `live_analysis.jsonl` is the standing proof that a
captured sample can stop clearing a gate that was later tightened, and a test
asserting otherwise turns every future tightening into a red test in this file
rather than a finding about the corpus. The §F criteria below are different in
kind -- they read the *current* tables (`word_budget`, `KIND_SENTENCE_CAPS`),
so they move when the contract moves, which is what makes them worth pinning.
"""

from __future__ import annotations

import collections
import json
import os

import pytest

from pipelines.v3 import config, row as rowlib
from pipelines.v3.packs import compute_pack
from pipelines.v3.teacher.prompts import word_budget
from pipelines.v3.verification.contradictions import contradiction_violations
from pipelines.v3.verify_v3 import ROW_REQUIRED

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(_HERE))
CAPTURE = os.path.join(_DATASET, "examples", "v3", "live_v3_2.jsonl")


def _rows() -> list[dict]:
    with open(CAPTURE, encoding="utf8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_capture_covers_three_registers_and_three_work_types():
    rows = _rows()
    assert len(rows) == 8, "eight, not nine: memo x desk_chat is no longer legal"
    assert set(r["register"] for r in rows) == {
        "desk_chat",
        "ic_memo",
        "risk_committee",
    }
    assert set(r["record_type"] for r in rows) == {"analysis", "grounded", "memo"}
    # The pair §C forbids does not appear, which is why there are eight.
    assert not [
        r for r in rows if r["record_type"] == "memo" and r["register"] == "desk_chat"
    ]


@pytest.mark.parametrize("field", sorted(ROW_REQUIRED))
def test_every_captured_row_carries_the_columns_prepare_joins_on(field):
    for row in _rows():
        assert field in row, f"{row.get('id')} is missing {field}"


def test_no_captured_row_carries_the_factory():
    for row in _rows():
        assert not rowlib.carries_teacher_brief(row), row.get("id")
        assert config.TEACHER_FINGERPRINT not in json.dumps(row)
        assert "messages" not in row, "a prose row's prompt is its question"


def test_the_capture_records_that_thinking_was_off():
    """§B's claim, and the bake-off's verdict, on the rows themselves."""
    for row in _rows():
        teacher = row["verification"]["teacher"]
        assert teacher["think_present"] is False, row.get("id")
        assert teacher["model"] == "deepseek-v4-flash-0731"
        # §F: completion under 900 tokens. The old capture's median was 476 and
        # its memo rows ran past 900.
        assert teacher["usage"]["completion_tokens"] < 900, row.get("id")


def test_every_row_sits_in_the_band_its_kind_and_register_allow():
    """§F: a desk analysis is 160 words where a committee analysis is 220."""
    for row in _rows():
        low, high = word_budget(row["record_type"], row["register"])
        words = len(row["answer"].split())
        assert low <= words <= high, f"{row['id']}: {words} words"


def test_no_row_contradicts_its_own_pack():
    """§D, against the recomputed pack rather than the stored one.

    The three tags are the three failures the old capture shipped board-green:
    a schedule called the risk under the cap, a negative drift read as a gain,
    and an effect named worth acting on out of pieces that do not add up.
    """
    for row in _rows():
        pack = compute_pack(
            row["work_type"], rowlib.family_of(row), row["variant"]
        ).to_dict()
        assert contradiction_violations(pack, row["answer"]) == [], row["id"]


def test_analysis_and_grounded_do_not_open_on_the_same_sentence():
    """§F's sharpest criterion: two record types, two jobs, two first lines."""
    openings: dict[tuple, dict[str, str]] = collections.defaultdict(dict)
    for row in _rows():
        if row["record_type"] not in ("analysis", "grounded"):
            continue
        head = " ".join(row["answer"].split()[:8]).casefold()
        openings[(row["work_type"], row["variant"])][row["record_type"]] = head
    assert openings, "the capture carries no analysis/grounded pair to compare"
    for coord, pair in openings.items():
        assert len(pair) == 2, coord
        assert pair["analysis"] != pair["grounded"], coord
