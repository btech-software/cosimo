"""The committed live capture is a student row set, and stays one.

`live_three_registers.jsonl` is a *captured* sample: nine rows a real teacher
wrote, kept verbatim. Like `live_analysis.jsonl` it is not regenerable, so it
cannot be asserted byte-identical to a harness the way the eight
`<record_type>.jsonl` files are.

What it can be held to is the set of properties that do not depend on how any
gate is currently tuned -- the two-surface split (§A), the schema the prepare
step joins on, and the register coverage that is the file's whole reason for
existing. Deliberately *not* the full board: `live_analysis.jsonl` is the
standing proof that a captured sample can stop clearing a gate that was later
tightened, and a test asserting otherwise would turn every future tightening
into a failure in this file rather than a finding about the corpus.
"""

from __future__ import annotations

import collections
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3 import config, row as rowlib  # noqa: E402
from pipelines.v3.verify_v3 import ROW_REQUIRED  # noqa: E402

CAPTURE = os.path.join(_DATASET, "examples", "v3", "live_three_registers.jsonl")


def _rows() -> list[dict]:
    with open(CAPTURE, encoding="utf8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_capture_is_present_and_holds_every_register():
    """Its reason for existing: one sample that is not all `desk_chat`.

    Every live sample before it was single-register, because `render` had no
    `--work-type` and `--limit` cuts after a sort that groups by work type.
    """
    rows = _rows()
    assert len(rows) == 9
    registers = collections.Counter(r["register"] for r in rows)
    assert set(registers) == {"desk_chat", "ic_memo", "risk_committee"}
    assert min(registers.values()) == 3


@pytest.mark.parametrize("field", sorted(ROW_REQUIRED))
def test_every_captured_row_carries_the_columns_prepare_joins_on(field):
    for row in _rows():
        assert field in row, f"{row.get('id')} is missing {field}"


def test_no_captured_row_carries_the_factory():
    """§A, on the surface that would actually be trained."""
    for row in _rows():
        assert not rowlib.carries_teacher_brief(row), row.get("id")
        assert config.TEACHER_FINGERPRINT not in json.dumps(row)
        assert "messages" not in row, (
            f"{row.get('id')}: a prose row's prompt is its question"
        )


def test_the_capture_is_supervised_and_claims_its_own_verification():
    for row in _rows():
        assert row.get("verified") is True, row.get("id")
        assert row.get("holdout") is False, (
            f"{row.get('id')} is a holdout row sitting in an sft sample (§E)"
        )


def test_the_capture_records_that_thinking_was_off():
    """The one fact this file exists to carry beside the prose.

    Every earlier capture reasoned unconditionally -- not by choice, but
    because the think flag was sent in a dialect the serving stack ignores. A
    sample that quietly lost this field would lose the evidence for §B.
    """
    for row in _rows():
        teacher = row["verification"]["teacher"]
        assert teacher["think_present"] is False, row.get("id")
        assert teacher["model"] == "deepseek-v4-flash-0731"
