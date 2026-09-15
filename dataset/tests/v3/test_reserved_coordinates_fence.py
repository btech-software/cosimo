"""The eval reservation, enforced in *every* render lane -- not just in prose.

``reserved_coordinates.json`` says "generate no row at all on this pack",
because the trap suite interrogates the whole scenario and a single row
rendered on it puts the evaluation's own figures into training. For a long
while only ``render/prose.py`` read the declaration: ``exam``, ``agentic`` and
``implementation`` walked straight past it, and three trap rows reached a
training shard while ``verify`` reported sixteen clean axes -- a recall result
dressed as generalisation, with no gate anywhere that could see it.

So the fence gets a test per lane rather than one test on the declaration
(``test_contradictions.py`` already checks the declaration round-trips against
``traps.jsonl``). The assertion is deliberately stronger than "the row is
absent": the guard must fire *before* authorship, so each lane is handed a
teacher that detonates on contact. A lane that consults the reservation late
-- after paying a teacher, or after composing from the pack -- still writes no
row, but it has already spent the call, and on a full run that is the whole
trap suite billed to prove a fence held.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.v3 import inventory, verify_v3, write  # noqa: E402
from pipelines.v3.render.agentic import KIND as AGENTIC_KIND  # noqa: E402
from pipelines.v3.render.agentic import run_agentic_stage  # noqa: E402
from pipelines.v3.render.exam import EXAM_KIND, run_exam_stage  # noqa: E402
from pipelines.v3.render.implementation import IMPL_KIND  # noqa: E402
from pipelines.v3.render.implementation import run_impl_stage  # noqa: E402
from pipelines.v3.render.prose import run_render_stage  # noqa: E402


class _Detonating:
    """A teacher that must never be reached: reserved jobs stop before authorship."""

    def __getattr__(self, name):
        raise AssertionError(
            f"the reserved-coordinate fence let a job through to teacher.{name}"
        )


def _jobs_for(record_type, count=3):
    jobs = [
        job
        for job in inventory.expand_jobs(inventory.load_plan())
        if job.record_type == record_type and not job.holdout
    ]
    if len(jobs) < count:
        pytest.skip(f"plan carries no {record_type} jobs to reserve")
    return jobs[:count]


def _reserve(monkeypatch, tmp_path, jobs):
    """Point the loader at a declaration covering exactly *jobs*' coordinates."""
    path = tmp_path / "reserved.json"
    path.write_text(
        json.dumps(
            {
                "why": "the fence under test",
                "reserved": [
                    {
                        "work_type": job.work_type,
                        "family": job.family,
                        "variant": job.variant,
                    }
                    for job in jobs
                ],
            }
        ),
        encoding="utf8",
    )
    # conftest points this at a file reserving nothing, so every other test
    # renders the plan it declares; this test is one of the few *about* the
    # reservation and says so explicitly.
    monkeypatch.setenv("COSIMO_V3_RESERVED_COORDS", str(path))
    return path


@pytest.mark.parametrize(
    "record_type,kind,run",
    [
        ("exam", EXAM_KIND, lambda out, jobs: run_exam_stage(out, jobs)),
        (
            "agentic",
            AGENTIC_KIND,
            lambda out, jobs: run_agentic_stage(out, jobs, _Detonating()),
        ),
        (
            "implementation",
            IMPL_KIND,
            lambda out, jobs: run_impl_stage(out, jobs, _Detonating()),
        ),
        (
            "analysis",
            "analysis",
            lambda out, jobs: run_render_stage(out, jobs, _Detonating()),
        ),
    ],
)
def test_every_lane_refuses_a_reserved_coordinate(
    record_type, kind, run, tmp_path, monkeypatch
):
    jobs = _jobs_for(record_type)
    _reserve(monkeypatch, tmp_path, jobs)
    out = str(tmp_path / "corpus")

    # No pack stage on purpose: the fence must precede pack composition too, so
    # a reserved job is neither a render nor a "missing pack" complaint.
    report = run(out, jobs)

    assert report["reserved_for_eval"] == len(jobs), (
        f"{record_type}: the lane did not count the reserved jobs it dropped"
    )
    assert report["rendered"] == 0
    assert report["dead_lettered"] == 0
    assert report["missing_packs"] == []
    assert report["skipped_by_pack_gate"] == []
    assert not os.path.isfile(write.path_for("sft", kind, out)), (
        f"{record_type}: a reserved coordinate reached a training shard"
    )


def test_axis_17_reds_a_shard_that_already_carries_one(tmp_path, monkeypatch):
    """The fence read back off disk, for the rows that predate the guard.

    The renderers refusing to write these is the fix; the board catching one
    already written is what makes a tree auditable rather than merely
    well-intentioned -- three such rows were on disk when the guard landed.
    """
    jobs = _jobs_for("exam", count=2)
    monkeypatch.setenv(
        "COSIMO_V3_RESERVED_COORDS", os.path.join(_HERE, "_no_reserved.json")
    )
    out = str(tmp_path / "corpus")
    from pipelines.v3 import stage

    stage.run_pack_stage(out, jobs)
    rendered = run_exam_stage(out, jobs)
    assert rendered["rendered"] > 0, "need a row on disk to then reserve under it"
    assert verify_v3.verify_dir(out, kinds=(EXAM_KIND,), quick=True)["ok"] is True

    # Now reserve the coordinates those rows sit on: same bytes, new verdict.
    _reserve(monkeypatch, tmp_path, jobs)
    report = verify_v3.verify_dir(out, kinds=(EXAM_KIND,), quick=True)
    axis = report["axes"]["eval-reserved coordinates"]
    assert report["ok"] is False
    assert len(axis["failures"]) == rendered["rendered"]
    assert all("reserved for evaluation" in f["problem"] for f in axis["failures"])


def test_the_axis_says_so_when_nothing_is_reserved(tmp_path, monkeypatch):
    """An empty declaration is a note, not a vacuous pass over every row."""
    jobs = _jobs_for("exam", count=2)
    monkeypatch.setenv(
        "COSIMO_V3_RESERVED_COORDS", os.path.join(_HERE, "_no_reserved.json")
    )
    out = str(tmp_path / "corpus")
    from pipelines.v3 import stage

    stage.run_pack_stage(out, jobs)
    run_exam_stage(out, jobs)
    axis = verify_v3.verify_dir(out, kinds=(EXAM_KIND,), quick=True)["axes"][
        "eval-reserved coordinates"
    ]
    assert axis["failures"] == []
    assert "nothing to hold out" in axis["note"]
