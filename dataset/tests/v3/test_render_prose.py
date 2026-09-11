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
from pipelines.v3.teacher.client import (  # noqa: E402
    FixtureTransport,
    TeacherError,
)
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
    # §A: a prose row carries no transcript at all. Its prompt is the pack's
    # question and its target is the answer; the brief that produced it lives
    # in the teacher log, which is off by default.
    assert "messages" not in row
    assert row["question"] == pack["question"]
    assert row["answer"] == clean
    assert row["fact_pack"]["scenario_id"] == pack["scenario_id"]
    assert row["verified"] is True
    assert row["family"] == row["scenario_id"][len(row["work_type"]) + 1 :]
    assert row["holdout"] is False
    assert row["verification"]["invented_numbers"] == []
    assert row["verification"]["register_ok"] is True
    assert row["verification"]["attempts"] == 1
    assert row["verification"]["render"]["attempts"] == 1
    assert row["verification"]["teacher"]["model"] == "scripted-t"
    assert row["verification"]["teacher"]["think_present"] is True
    assert row["verification"]["pack_seed"] == pack_line["verification"]["pack_seed"]
    assert (
        row["verification"]["computed_by"] == pack_line["verification"]["computed_by"]
    )
    assert transport.calls[0]["temperature"] == config.PROSE_TEMPERATURES[0]
    # §B: the prose lanes do not think, so no thinking block rides the body.
    assert "thinking" not in transport.calls[0], "analysis no longer thinks"
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


def test_the_dead_letter_keeps_the_best_draft_not_merely_the_last():
    """A truncated last attempt must not erase the real draft in the middle.

    The live shape this reproduces: attempt 1 came back empty (the reasoning
    teacher spent its budget thinking), attempt 2 produced a full draft that
    failed on content, attempt 3 truncated again. Recording only the final
    round put "0 words" in the post-mortem and hid both the draft and the
    only violations anybody could act on.
    """
    _, pack_line, clean = _pinned()
    bad = clean + INVENTED
    transport = Scripted([_reply(""), _reply(bad), _reply("")])
    outcome = render_prose_row(Teacher(transport), pack_line, kind="analysis")
    dead = outcome["dead_letter"]
    assert dead["attempts"] == 3
    assert dead["best_attempt"] == 2, "the middle draft is the one worth reading"
    assert "invented numbers" in dead["reason"], "not a word-budget complaint"
    assert "8153.7729" in " | ".join(dead["violations"])
    assert [h["words"] for h in dead["history"]] == [0, len(bad.split()), 0]
    assert [h["attempt"] for h in dead["history"]] == [1, 2, 3]
    assert dead["history"][0]["violations"], "an empty draft still has a verdict"


def test_an_all_empty_dead_letter_says_truncated_not_gate():
    """Nothing was ever judged, so the reason may not read like a judgement."""
    _, pack_line, _ = _pinned()
    transport = Scripted([_reply("")] * config.PROSE_ATTEMPTS)
    dead = render_prose_row(Teacher(transport), pack_line, kind="analysis")[
        "dead_letter"
    ]
    assert dead["reason"].startswith("truncated:")
    assert dead["best_attempt"] is None
    assert all(h["words"] == 0 for h in dead["history"])


def test_every_attempt_pays_the_lanes_flat_cap_not_an_escalating_ladder():
    """§B replaces the escalating budget with a flat per-lane cap.

    The ladder existed for one failure and one only: a *reasoning* teacher on
    the prose lane spent its whole 16384-token budget thinking and returned
    `content: null`, so a later attempt -- carrying the failed draft and the
    violation list on top of the brief -- needed more room than the one before
    it. Across a five-lane live run every row that drafted cleanly on attempt 1
    then truncated on attempt 2 at the budget that had just worked.

    The amendment removes the cause rather than paying for the symptom: think
    is off on analysis, so there is no chain of thought to run out of, and 800
    tokens is twice the widest prose band's 400-word ceiling. Cooling the
    temperature stays, because the *other* failure the ladder answered -- a
    model padding its way past the contract -- is real and unchanged.
    """
    _, pack_line, clean = _pinned()
    transport = Scripted([_reply(clean + INVENTED)] * config.PROSE_ATTEMPTS)
    render_prose_row(Teacher(transport), pack_line, kind="analysis")
    budgets = [call["max_tokens"] for call in transport.calls]
    assert budgets == [config.MAX_TOKENS_THINK_OFF] * config.PROSE_ATTEMPTS
    # The ladder that stayed: temperature, one notch cooler each round.
    temps = [call["temperature"] for call in transport.calls]
    assert temps == list(config.PROSE_TEMPERATURES[: config.PROSE_ATTEMPTS])


def test_a_think_on_lane_gets_real_headroom_and_a_think_off_lane_does_not():
    """The two caps are a property of the flag, checked where it is spent."""
    _, pack_line, clean = _pinned()
    transport = Scripted([_reply(clean)])
    render_prose_row(Teacher(transport), pack_line, kind="analysis")
    assert transport.calls[0]["max_tokens"] == config.MAX_TOKENS_THINK_OFF
    assert config.MAX_TOKENS_THINK_ON > config.MAX_TOKENS_THINK_OFF


def test_an_outage_keeps_the_rows_already_paid_for(tmp_path, monkeypatch):
    """A timeout on job N must not throw away jobs 1..N-1.

    The stage buffered every row and wrote once at the end, so the first
    five-lane live run -- forty minutes of teacher time, several rows already
    clean through the gate -- put nothing on disk when a later call timed
    out. The resume contract exists precisely so an outage costs the run and
    not the bill; buffering defeated it at the only moment it mattered.
    """
    _pin_lane(monkeypatch)
    out = str(tmp_path)
    payload, selected = _fixture_slice(4)
    entries = payload["entries"]

    class DiesOnTheThird(CountingFixture):
        def post(self, body):
            if self.calls >= 2:
                raise TeacherError("teacher unreachable: timed out")
            return super().post(body)

    transport = DiesOnTheThird(entries)
    for job in selected:
        write.append_unique(
            write.path_for("fact_packs", job.work_type, out),
            [stage.pack_record(job.work_type, job.family, job.variant)],
        )
    with pytest.raises(TeacherError, match="timed out"):
        run_render_stage(out, selected, Teacher(transport), types=("analysis",))
    survived = write.existing_ids(write.path_for("sft", "analysis", out))
    assert len(survived) == 2, "the rows the teacher was already paid for"

    # ...and the retry does not buy them a second time.
    replay = CountingFixture(entries)
    report = run_render_stage(out, selected, Teacher(replay), types=("analysis",))
    assert report["existing"] == 2 and replay.calls == len(selected) - 2


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
