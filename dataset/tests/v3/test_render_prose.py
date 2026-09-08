"""The repair loop, the dead letter, and the resume contract (spec §5.5, §6.3-8).

The scripted transports here are the point of PR2's test design: the repair
loop is a *policy* (how many strikes, how far the temperature falls, what the
repair turn says), and a policy is only testable with a teacher whose
behaviour you choose. The fixture-transport tests underneath then prove the
same stage against the *real* replay path the offline cluster will use.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.render.prose import (  # noqa: E402
    render_prose_row,
    row_id,
    row_id_from_coords,
    run_render_stage,
    select_prose_jobs,
)
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport, TeacherError  # noqa: E402
from pipelines.v3.verification.prose import gate_violations  # noqa: E402

PINNED = ("valuation.equity.dcf", "mature_consumer", 0)
INVENTED = " The terminal multiple is 8153.7729."


class Scripted:
    """A teacher whose every answer is chosen by the test."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def post(self, body):
        self.calls.append(body)
        if not self._replies:
            raise AssertionError("renderer asked the teacher more times than scripted")
        return self._replies.pop(0)


class CountingFixture(FixtureTransport):
    """Replay, and count -- so 'resume asked nobody' is an assertion, not a hope."""

    def __init__(self, entries):
        super().__init__(entries=entries)
        self.calls = 0

    def post(self, body):
        self.calls += 1
        return super().post(body)


def _reply(text, model="scripted-t", think=None):
    message = {"content": text, "role": "assistant"}
    if think is not None:
        message["reasoning_content"] = think
    return {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"total_tokens": max(1, len(text.split()))},
    }


def _pinned():
    pack = compute_pack(*PINNED).to_dict()
    pack_line = stage.pack_record(*PINNED)
    clean = prose_harness.compliant_text(pack, "analysis")
    assert gate_violations(pack, clean, "analysis") == []
    return pack, pack_line, clean


def _pin_lane(monkeypatch):
    """Offline convention (shared with make_teacher_echo): the replay table is
    keyed on the dummy's own model name, so the prose lane points at it."""
    monkeypatch.setenv(config.TEACHER_PROSE_ENV, prose_harness.DUMMY_MODEL)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, prose_harness.DUMMY_MODEL)


def _plan_jobs():
    return inventory.expand_jobs(inventory.load_plan())


def _fixture_slice(n):
    payload = prose_harness.build_fixture(("analysis",), n)
    examined = payload["meta"]["jobs_examined"]
    selected = select_prose_jobs(_plan_jobs(), types=("analysis",), limit=None)[
        :examined
    ]
    return payload, selected


# --------------------------------------------------------------------------- loop


def test_clean_first_attempt_ships_the_row_verbatim():
    pack, pack_line, clean = _pinned()
    transport = Scripted([_reply(clean, think="checked against the pack")])
    outcome = render_prose_row(Teacher(transport), pack_line, kind="analysis")
    row = outcome["row"]
    assert outcome["dead_letter"] is None
    assert row["id"] == row_id("analysis", pack)
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["answer"] == clean == row["messages"][-1]["content"]
    assert row["verification"]["render"]["attempts"] == 1
    assert row["verification"]["teacher"]["model"] == "scripted-t"
    assert row["verification"]["teacher"]["think_present"] is True
    assert row["verification"]["pack_seed"] == pack_line["verification"]["pack_seed"]
    assert (
        row["verification"]["computed_by"] == pack_line["verification"]["computed_by"]
    )
    assert transport.calls[0]["temperature"] == config.PROSE_TEMPERATURES[0]
    assert transport.calls[0]["thinking"] == {"type": "enabled"}, "analysis thinks"
    assert transport.calls[0]["model"]  # the routed lane name, not a literal here


def test_a_violated_draft_buys_one_repair_turn_naming_the_violations():
    pack, pack_line, clean = _pinned()
    bad = clean + INVENTED
    assert any("invented numbers" in v for v in gate_violations(pack, bad, "analysis"))
    transport = Scripted([_reply(bad), _reply(clean)])
    outcome = render_prose_row(Teacher(transport), pack_line, kind="analysis")
    assert outcome["row"]["verification"]["render"]["attempts"] == 2
    repair_turn = transport.calls[1]["messages"][-1]["content"]
    assert "failed the contract" in repair_turn
    assert "8153.7729" in repair_turn, "the repair prompt must name the offender"
    assert bad in repair_turn, "and show the draft it is repairing"
    assert transport.calls[1]["temperature"] == config.PROSE_TEMPERATURES[1]
    assert (
        transport.calls[0]["temperature"],
        transport.calls[1]["temperature"],
    ) == config.PROSE_TEMPERATURES[:2], "the ladder is config's, not a literal"


def test_three_strikes_write_the_dead_letter_whole():
    _, pack_line, clean = _pinned()
    bad = clean + INVENTED
    transport = Scripted([_reply(bad)] * config.PROSE_ATTEMPTS)
    outcome = render_prose_row(Teacher(transport), pack_line, kind="analysis")
    assert outcome["row"] is None
    dead = outcome["dead_letter"]
    assert dead["attempts"] == config.PROSE_ATTEMPTS
    assert "invented numbers" in dead["reason"]
    assert dead["violations"], "the last verdict travels with the body"
    assert [m["role"] for m in dead["exchange"]][-1] == "user"
    assert len(dead["exchange"]) == 2 + 2 * config.PROSE_ATTEMPTS
    assert dead["temperatures"] == list(config.PROSE_TEMPERATURES)


def test_an_outage_is_an_outage_not_a_verdict():
    """TeacherError propagates: the transport breaking must not dead-letter
    scenarios that never got a fair question."""

    class Down:
        def post(self, body):
            raise TeacherError("endpoint down")

    _, pack_line, _ = _pinned()
    with pytest.raises(TeacherError, match="endpoint down"):
        render_prose_row(Teacher(Down()), pack_line, kind="analysis")


def test_unrouted_record_type_is_refused_before_any_question():
    _, pack_line, _ = _pinned()
    with pytest.raises(ValueError, match="exam"):
        render_prose_row(Teacher(Scripted([])), pack_line, kind="exam")


# --------------------------------------------------------------------------- stage


def test_stage_renders_replays_and_asks_nobody_twice(tmp_path, monkeypatch):
    _pin_lane(monkeypatch)
    out = str(tmp_path)
    payload, selected = _fixture_slice(6)
    stage.run_pack_stage(out, selected)
    transport = CountingFixture(entries=payload["entries"])
    report = run_render_stage(out, selected, Teacher(transport))
    assert report["rendered"] == 6
    assert report["missing_packs"] == [] and report["dead_lettered"] == 0
    rows = write.read_jsonl(write.path_for("sft", "analysis", out))
    assert len(rows) == 6
    assert len({row["id"] for row in rows}) == 6
    calls_after_first = transport.calls
    replay = run_render_stage(out, selected, Teacher(transport))
    assert replay["rendered"] == 0 and replay["existing"] == 6
    assert transport.calls == calls_after_first, "resume must not re-bill the teacher"


def test_rows_reproduce_from_their_own_coordinates(tmp_path, monkeypatch):
    _pin_lane(monkeypatch)
    out = str(tmp_path)
    payload, selected = _fixture_slice(6)
    stage.run_pack_stage(out, selected)
    run_render_stage(
        out, selected, Teacher(CountingFixture(entries=payload["entries"]))
    )
    for row in write.read_jsonl(write.path_for("sft", "analysis", out)):
        family = row["scenario_id"][len(row["work_type"]) + 1 :]
        assert row["id"] == row_id_from_coords(
            row["record_type"], row["work_type"], family, row["variant"]
        )


def test_dead_letters_land_whole_and_are_never_asked_again(tmp_path, monkeypatch):
    _pin_lane(monkeypatch)
    out = str(tmp_path)
    payload, selected = _fixture_slice(4)
    stage.run_pack_stage(out, selected)

    class AlwaysBad:
        def __init__(self):
            self.calls = 0

        def post(self, body):
            self.calls += 1
            return _reply("No numbers, no facts, no structure at all.")

    transport = AlwaysBad()
    report = run_render_stage(out, selected, Teacher(transport))
    assert report["dead_lettered"] == 4 and report["rendered"] == 0
    dead_file = write.path_for("dead_letter", "analysis", out)
    deads = write.read_jsonl(dead_file)
    assert len(deads) == 4
    assert all(
        "must_mention" in d["reason"] or "invented" in d["reason"] for d in deads
    )
    before = transport.calls
    replay = run_render_stage(out, selected, Teacher(transport))
    assert replay["existing"] == 4 and replay["dead_lettered"] == 0
    assert transport.calls == before, "a dead letter is a verdict, not a retry queue"


def test_the_two_absences_are_two_different_events(tmp_path, monkeypatch):
    """No pack *file* for a work type is a DAG ordering bug (fatal); a variant
    absent from a present file is the pack guard's verdict (recorded skip)."""
    _pin_lane(monkeypatch)
    out = str(tmp_path)
    payload, _ = _fixture_slice(4)
    jobs = select_prose_jobs(_plan_jobs(), types=("analysis",), limit=None)
    first = jobs[0]
    same_cell = [
        job
        for job in jobs
        if (job.work_type, job.family) == (first.work_type, first.family)
        and job.variant < 12
    ]

    def renderable(job) -> bool:
        try:
            compute_pack(job.work_type, job.family, job.variant)
        except PackError:
            return False
        return True

    packed = [job for job in same_cell if renderable(job)]
    assert len(packed) >= 2, "the cell must hold two renderable variants to stage this"
    good, absent = packed[0], packed[1]
    foreign = next(job for job in jobs if job.work_type != good.work_type)
    stage.run_pack_stage(out, [good])  # the file exists, holding exactly one variant
    transport = CountingFixture(entries=payload["entries"])
    report = run_render_stage(out, [good, absent, foreign], Teacher(transport))
    assert report["rendered"] == 1 and transport.calls == 1
    assert [e["variant"] for e in report["skipped_by_pack_gate"]] == [absent.variant]
    assert [e["work_type"] for e in report["missing_packs"]] == [foreign.work_type], (
        "the work type whose pack file was never written is the fatal one"
    )


def test_holdout_families_never_touch_the_training_shards(tmp_path, monkeypatch):
    _pin_lane(monkeypatch)
    out = str(tmp_path)
    holdouts = [
        job for job in _plan_jobs() if job.holdout and job.record_type == "analysis"
    ]
    assert holdouts, "the plan must carry holdout families for this to mean anything"
    payload, _ = _fixture_slice(2)
    transport = CountingFixture(entries=payload["entries"])
    report = run_render_stage(out, holdouts, Teacher(transport))
    assert report["jobs_seen"] == 0 and report["rendered"] == 0
    assert transport.calls == 0
    assert write.read_jsonl(write.path_for("sft", "analysis", out)) == []


def test_asking_for_a_non_prose_type_is_a_usage_error():
    with pytest.raises(ValueError, match="prose types only"):
        select_prose_jobs(_plan_jobs(), types=("analysis", "exam"))


def test_types_filter_outranks_limit():
    """``--types memo --limit N`` must never spend the budget on analysis jobs."""
    selected = select_prose_jobs(_plan_jobs(), types=("memo",), limit=3)
    assert [job.record_type for job in selected] == ["memo"] * 3, (
        "the limit counts jobs of the requested types, not of the whole plan"
    )
