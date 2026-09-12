"""The examples are student rows, and the old ones must fail the new gate (§A, §H).

Two duties, and the second is the amendment's own acceptance test: "tests on the
9-row file (those rows must *fail* the new prepare gate)". A schema change that
only makes new rows right is a schema change nobody can tell happened -- what
proves the two-surface split landed is that the previous shape is now refused,
loudly, by both sides of the join.

The first duty is that the committed examples regenerate and pass everything a
shipped row passes. They are the only picture of the corpus a reader gets
before running anything, and the old picture was of the teacher's brief.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(_HERE))
_EXAMPLES = os.path.join(_DATASET, "examples", "v3")
for _p in (_HERE, os.path.join(_HERE, "fixtures"), _EXAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3 import config, row as rowlib, write  # noqa: E402
from pipelines.v3.teacher.prompts import BRIEF_KINDS  # noqa: E402
from pipelines.v3.verify_v3 import ROW_REQUIRED, _check_row  # noqa: E402

#: Every record type the corpus publishes. `examples/v3/` must hold one file
#: per entry: a reader who cannot see a record type has no reason to believe it
#: exists, and the old single-kind file is exactly how `memo`, `critique` and
#: `grounded` went three PRs without a committed example.
EXPECTED_KINDS = (*BRIEF_KINDS, "exam", "agentic", "implementation")

LEGACY = os.path.join(_EXAMPLES, "_teacher_logs", "legacy_analysis_9rows.jsonl")


def _example(kind: str) -> dict:
    path = os.path.join(_EXAMPLES, f"{kind}.jsonl")
    with open(path, encoding="utf8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 1, f"{kind}: an example file holds exactly one row"
    return rows[0]


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_every_record_type_has_one_committed_example(kind):
    row = _example(kind)
    assert row["record_type"] == kind
    missing = [field for field in ROW_REQUIRED if field not in row]
    assert not missing, f"{kind}: example is missing {missing}"


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_no_example_carries_the_factory(kind):
    """The whole point of §H's replacement, asserted per file."""
    row = _example(kind)
    assert not rowlib.carries_teacher_brief(row), (
        f"{kind}: the committed example shows a reader the labelling protocol"
    )
    assert config.TEACHER_FINGERPRINT not in json.dumps(row)


@pytest.mark.parametrize("kind", sorted(BRIEF_KINDS))
def test_a_prose_example_has_no_transcript_at_all(kind):
    """A prose row's prompt is its question. There is nothing else to show."""
    row = _example(kind)
    assert "messages" not in row
    # The question the row was *asked*: the pack's own, except on the
    # abstention lane, which is asked the one the pack cannot answer so that a
    # refusal is a refusal rather than an analysis with a caveat.
    from pipelines.v3.teacher.prompts import question_for

    assert row["question"] == question_for(row["fact_pack"], kind)
    if kind == "abstention":
        assert row["question"] == row["fact_pack"]["abstention_question"]
        assert row["question"] != row["fact_pack"]["question"]


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_every_example_clears_the_board(kind, tmp_path):
    """Not "looks right": passes the same axes a shipped row passes.

    Run through ``_check_row`` rather than through eight hand-written
    assertions, because the contract is the board's and a second copy of it
    here would be a second contract that drifts.
    """
    row = _example(kind)
    out = str(tmp_path)
    write.write_jsonl(write.path_for("sft", kind, out), [row])
    failures = _check_row(row, frozenset(), quick=True)
    for axis, problems in failures.items():
        assert not problems, f"{kind} / {axis}: {problems}"


# --------------------------------------------------------------------------
# the amendment's own acceptance test
# --------------------------------------------------------------------------


def test_the_legacy_nine_row_file_is_kept_where_debug_belongs():
    """It moved rather than vanishing: a post-mortem still wants to read it."""
    assert os.path.isfile(LEGACY), (
        "the old example_generation.jsonl is the record of what the corpus used "
        "to ship; it belongs under _teacher_logs/, not deleted"
    )
    assert "_teacher_logs" in LEGACY


def test_every_legacy_row_fails_the_new_gate():
    """§J PR-A, verbatim: "those rows must *fail* the new prepare gate".

    Three separate refusals, and each row trips at least one. Together they are
    the statement that the previous shape is no longer a student row:

    * it carries the teacher's system turn -- the fingerprint gate;
    * it does not claim ``verified`` -- the corpus now asserts that per row
      rather than leaving the harness to infer it from a stamp;
    * it has no ``holdout``, ``family`` or ``fact_pack`` -- the columns that let
      a reader refuse a leak and score a row without re-running the generator.
    """
    with open(LEGACY, encoding="utf8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows, "the legacy file is the evidence; an empty one proves nothing"
    for row in rows:
        reasons = []
        if rowlib.carries_teacher_brief(row):
            reasons.append("teacher fingerprint")
        if row.get("verified") is not True:
            reasons.append("no verified claim")
        missing = [field for field in ROW_REQUIRED if field not in row]
        if missing:
            reasons.append(f"missing {', '.join(missing)}")
        assert reasons, f"{row.get('id')} would still be accepted as a student row"


def test_the_legacy_rows_are_refused_by_the_board_too():
    """Both sides of the join refuse them, not only the harness."""
    with open(LEGACY, encoding="utf8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        failures = _check_row(row, frozenset(), quick=True)
        assert failures["schema"], f"{row.get('id')} passed axis 1 unchanged"
