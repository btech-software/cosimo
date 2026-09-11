"""The committed implementation dummy's two duties: regenerate byte-identical,
and carry the lane's gate -- the whole committed train slice rendered offline,
every hidden suite executed by the board, the pass rate exactly one.

The drift test is the same argument the prose fixture makes: a fixture nobody
can regenerate is folklore. This lane adds what no other fixture here needs:
the harness *executes* before it admits an entry, so a byte-identical
regeneration is also re-proof that every composed suite still passes its
reference -- the references are reviewed code, and the fixture is their
witness table kept current.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_impl_fixture as impl_harness  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.render.implementation import (  # noqa: E402
    IMPL_KIND,
    run_impl_stage,
    select_impl_jobs,
)
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verify_v3 import verify_dir  # noqa: E402

COMMITTED = os.path.join(_HERE, "fixtures", impl_harness.IMPL_FIXTURE_NAME)


def _regenerate_bytes(payload: dict, tmp_path) -> bytes:
    path = os.path.join(str(tmp_path), "regen.json")
    with open(path, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    with open(path, "rb") as handle:
        return handle.read()


def test_the_committed_fixture_is_byte_identical_to_its_harness(tmp_path):
    payload = impl_harness.build_fixture(impl_harness.DEFAULT_LIMIT)
    with open(COMMITTED, "rb") as handle:
        committed = handle.read()
    assert _regenerate_bytes(payload, tmp_path) == committed, (
        "the committed fixture drifted from its harness; regenerate it with "
        ".venv/bin/python dataset/tests/v3/fixtures/make_impl_fixture.py and "
        "inspect the diff before accepting it"
    )


def test_the_whole_slice_renders_offline_and_the_board_certifies_it(
    tmp_path, monkeypatch
):
    """The PR4 gate in one test: no network, no GPU, no wall clock -- every
    committed row, hidden-test pass rate exactly 1.0, every axis green.

    The row count and the job window both come from the committed fixture
    rather than from literals, because both are *plan* numbers rather than
    contracts. Under the old plan the family cap truncated the implementation
    lane to 29 jobs, so "every planned job" and "every captured job" were the
    same set and the test could say `limit=None`. The amendment's WIP counts
    (analysis 120 -> 20, exam 60 -> 36) leave every family at 228 planned rows
    against a 285 ceiling, so nothing truncates and the lane plans 120 -- of
    which this fixture still commits the first 30. Walking `limit=None` against
    a 30-entry table is a fixture miss, not a finding.

    What the gate promises is unchanged and is what the rest of this test
    says: every committed entry renders offline, no row dead-letters, every
    hidden suite executes, and the board is green across the slice.
    """
    payload = json.load(open(COMMITTED, encoding="utf8"))
    expected = payload["meta"]["entries"]
    assert expected > 0, "the slice is a whole, not a sample"
    assert expected * 2 >= payload["meta"]["jobs_examined"], (
        "more than half the planned draws were vetoed by the pack guards -- "
        "that is a computer defect, not a verdict"
    )
    assert payload["meta"]["record_type"] == IMPL_KIND

    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(inventory.load_plan())
    selected = select_impl_jobs(jobs, limit=payload["meta"]["limit"])
    stage.run_pack_stage(out, selected)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, payload["model"])
    monkeypatch.delenv(config.LIVE_ENV, raising=False)

    report = run_impl_stage(out, selected, Teacher(FixtureTransport(path=COMMITTED)))
    assert report["rendered"] == expected and report["dead_lettered"] == 0

    rows = write.read_jsonl(write.path_for("sft", IMPL_KIND, out))
    assert len(rows) == expected == len({row["id"] for row in rows})
    for row in rows:
        assert row["limitations"] == impl_harness._LIMITATIONS
        assert row["answer"] == row["reference_code"]
        assert row["verification"]["render"]["sandbox"] == "passed"
        assert row["hidden_tests"], "a row without a hidden suite is a row unsuited"

    board = verify_dir(out, kinds=(IMPL_KIND,))
    assert board["ok"] is True, board["axes"]
    assert board["axes"]["hidden tests"]["failures"] == []
    assert board["axes"]["hidden tests"]["checked"] == expected
