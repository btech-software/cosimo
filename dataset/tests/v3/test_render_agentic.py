"""The loop's policies under teachers whose behaviour the test chooses.

The gate (``test_verification_agentic``) decides *what* a legal trajectory is;
this file decides that the loop actually enforces it: the single strike on a
call the registry would refuse, the budget that ends an argument-with-itself
before the meter runs, the repair ladder that cools only the final turn while
the transcript's grammar stays fixed, and the resume contract -- ids decide
what is ever asked of the teacher, again. Scripted transports carry the
same replies the prose suite's do; the committed-fixture run underneath
replays the whole stage through the real ``Teacher``/``FixtureTransport``
pair, so what goes green here is what the offline cluster will run.
"""

from __future__ import annotations

import json
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_agentic_fixture as harness  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.oracle import faults, runtime  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.agentic import (  # noqa: E402
    render_agentic_row,
    row_id_from_coords,
    run_agentic_stage,
    select_agentic_jobs,
)
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verification.agentic import gate_violations  # noqa: E402

PINNED = ("valuation.equity.dcf", "mature_consumer", 0)
RANK_CLEAN, RANK_NO_CALL, RANK_RATE = 1, 0, 17
INVENTED = " The risk-free rate is 8153.7729 percent."


class Scripted:
    """A teacher whose every answer is chosen by the test, wire form included."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def post(self, body):
        self.calls.append(body)
        if not self._replies:
            raise AssertionError("the loop asked more times than the test scripted")
        return self._replies.pop(0)


class CountingFixture(FixtureTransport):
    """Replay, and count -- so "resume asked nobody" is an assertion, not a hope."""

    def __init__(self, entries):
        super().__init__(entries=entries)
        self.posts = 0

    def post(self, body):
        self.posts += 1
        return super().post(body)


def _reply(text="", calls=None, model="scripted-t", think=None):
    message = {"content": text, "role": "assistant"}
    if calls:
        message["tool_calls"] = [
            {
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for c in calls
        ]
    if think is not None:
        message["reasoning_content"] = think
    return {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"total_tokens": max(1, len(text.split()))},
    }


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


def _clean_script(pack, rank=RANK_CLEAN, think=None):
    """The three asks of a clean two-call trajectory, text chosen by the caller."""
    policy = harness.ScriptedTeacher(pack, rank, {})
    needed = list(runtime.tools_for(pack["work_type"]))
    taken = []
    replies = []
    for occurrence, name in enumerate(needed):
        spec = policy._call_for(name, occurrence, len(needed), taken)
        taken.append(name)
        replies.append(_reply(calls=[spec], think=think if occurrence == 0 else None))
    return replies, policy


def _pinned():
    pack = compute_pack(*PINNED).to_dict()
    pack_line = stage.pack_record(*PINNED)
    return pack, pack_line


def _fundamentals_metrics(pack):
    available = sorted(
        {*harness._scalars(pack["inputs"]), *harness._scalars(pack["computed"])}
    )
    return available[:4]


# ---------------------------------------------------------------- the happy path


def test_a_clean_trajectory_ships_the_transcript_and_the_ledger():
    pack, pack_line = _pinned()
    replies, policy = _clean_script(pack)
    closing = policy._final_text()
    replies.append(_reply(closing, think="compared against the pack"))
    outcome = render_agentic_row(Teacher(Scripted(replies)), pack_line, rank=RANK_CLEAN)
    row = outcome["row"]
    assert outcome["dead_letter"] is None
    roles = [m["role"] for m in row["messages"]]
    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["answer"] == closing == row["messages"][-1]["content"]
    assert row["tool_names"] == ["compute_metrics", "get_fundamentals"]
    assert {s["function"]["name"] for s in row["tool_schemas"]} == set(
        row["tool_names"]
    )
    render_field = row["verification"]["render"]
    assert render_field == {
        "kind": "agentic",
        "attempts": 3,
        "mode": "clean",
        "fault": None,
        "tool_calls": 2,
        "temperatures": [config.AGENTIC_TEMPERATURES[0]] * 3,
    }
    assert row["verification"]["teacher"]["model"] == "scripted-t"
    assert row["verification"]["teacher"]["think_present"] is True
    assert row["verification"]["pack_seed"] == pack_line["verification"]["pack_seed"]
    assert row["id"] == row_id_from_coords("agentic", *PINNED[:2], PINNED[2])


def test_no_call_mode_asks_once_and_executes_nothing():
    pack, pack_line = _pinned()
    policy = harness.ScriptedTeacher(pack, RANK_NO_CALL, {})
    outcome = render_agentic_row(policy, pack_line, rank=RANK_NO_CALL)
    row = outcome["row"]
    assert row["verification"]["render"] == {
        "kind": "agentic",
        "attempts": 1,
        "mode": "no_call",
        "fault": None,
        "tool_calls": 0,
        "temperatures": [config.AGENTIC_TEMPERATURES[0]],
    }
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["tool_names"] == [] and row["tool_schemas"] == []


def test_the_rate_limit_refusal_is_answered_by_an_identical_reissue():
    """The desk's own reflex, taught by the brief and proven by the ledger:
    three executions -- refusal, re-issue, the second ask -- and the answer
    that says so."""
    pack = compute_pack(*PINNED).to_dict()
    policy = harness.ScriptedTeacher(pack, RANK_RATE, {})
    outcome = render_agentic_row(
        policy, {**pack, "id": "t", "verification": {}}, rank=RANK_RATE
    )
    row = outcome["row"]
    render_field = row["verification"]["render"]
    assert render_field["mode"] == "faulted" and render_field["fault"] == "rate_limit"
    assert render_field["tool_calls"] == 3
    refused = json.loads(
        [m["content"] for m in row["messages"] if m["role"] == "tool"][0]
    )
    assert refused["__fault__"] == "rate_limit"
    assert "re-issued" in row["answer"]


# ---------------------------------------------------------------- single strike


def test_an_unregistered_tool_never_executes_and_kills_the_row_whole():
    pack, pack_line = _pinned()
    replies = [_reply(calls=[_call("get_weather", symbol="YCBK")])]
    transport = Scripted(replies)
    outcome = render_agentic_row(Teacher(transport), pack_line, rank=RANK_CLEAN)
    dead = outcome["dead_letter"]
    assert outcome["row"] is None
    assert dead["reason"].startswith("call:")
    assert "not in the registry" in dead["reason"]
    assert not any(m.get("role") == "tool" for m in dead["exchange"])  # nothing ran
    assert dead["attempts"] == 1


def test_arguments_the_schema_refuses_are_refused_before_the_oracle():
    pack, pack_line = _pinned()
    replies = [
        _reply(
            calls=[
                _call(
                    "get_fundamentals",
                    symbol="YCBK",
                    metrics=_fundamentals_metrics(pack),
                    junk=1,
                )
            ]
        )
    ]
    outcome = render_agentic_row(Teacher(Scripted(replies)), pack_line, rank=RANK_CLEAN)
    assert outcome["dead_letter"]["reason"].startswith("call:")
    assert "unknown argument" in outcome["dead_letter"]["reason"]
    assert not any(m.get("role") == "tool" for m in outcome["dead_letter"]["exchange"])


def test_an_argument_less_call_is_judged_by_the_schema_not_the_loop():
    """Missing optional arguments are the schema's business, not the loop's:
    the call is *accepted* (no ``call:`` death, nothing refused for being
    malformed) and then judged like any other transcript -- here by the
    message band, one call having made no loop."""
    pack, pack_line = _pinned()
    thin = _reply("Enough from the one block, yet too few exchanges.")
    replies = [_reply(calls=[_call("get_fundamentals", symbol="YCBK")])] + [thin] * 3
    outcome = render_agentic_row(Teacher(Scripted(replies)), pack_line, rank=RANK_CLEAN)
    dead = outcome["dead_letter"]
    assert dead is not None and dead["reason"].startswith("gate:")
    assert "outside the" in dead["reason"] and "band" in dead["reason"]
    assert "schema" not in dead["reason"] and "registry" not in dead["reason"]


# ---------------------------------------------------------------- budgets


def test_a_teacher_that_always_calls_meets_the_budget_before_the_seventh_execution():
    pack, pack_line = _pinned()
    metrics = _fundamentals_metrics(pack)
    transport = Scripted(
        [
            _reply(
                calls=[_call("get_fundamentals", symbol="YCBK", metrics=metrics[:1])]
            )
            for _ in range(9)
        ]
    )
    outcome = render_agentic_row(Teacher(transport), pack_line, rank=RANK_CLEAN)
    dead = outcome["dead_letter"]
    assert dead["reason"].startswith("budget:")
    assert "exceed the budget" in dead["reason"]
    assert dead["tool_calls"] == config.AGENTIC_MAX_TOOL_CALLS
    assert transport.calls and len(transport.calls) == config.AGENTIC_MAX_TOOL_CALLS + 1


def test_a_no_call_job_that_reaches_for_a_tool_dies_naming_the_waste():
    """Single strike, like the unregistered call: the shape of the row is
    fixed before any wording, and a repair cannot un-call what was asked."""
    pack, pack_line = _pinned()
    metrics = _fundamentals_metrics(pack)
    transport = Scripted(
        [_reply(calls=[_call("get_fundamentals", symbol="YCBK", metrics=metrics)])]
    )
    outcome = render_agentic_row(Teacher(transport), pack_line, rank=RANK_NO_CALL)
    dead = outcome["dead_letter"]
    assert dead["reason"].startswith("call:")
    assert "waste" in dead["reason"] and "no_call" in dead["reason"]
    assert dead["attempts"] == 1
    assert not any(m.get("role") == "tool" for m in dead["exchange"])


# ---------------------------------------------------------------- the ladder


def test_the_repair_ladder_cools_only_the_final_turn():
    pack, pack_line = _pinned()
    replies, policy = _clean_script(pack)
    replies.append(_reply("Bad: the risk-free rate is 8153.7729 percent."))
    replies.append(_reply(policy._final_text()))
    transport = Scripted(replies)
    outcome = render_agentic_row(Teacher(transport), pack_line, rank=RANK_CLEAN)
    row = outcome["row"]
    assert row["answer"] == policy._final_text()
    render_field = row["verification"]["render"]
    assert render_field["attempts"] == 4
    assert render_field["temperatures"] == [
        config.AGENTIC_TEMPERATURES[0],
        config.AGENTIC_TEMPERATURES[0],
        config.AGENTIC_TEMPERATURES[0],
        config.AGENTIC_TEMPERATURES[1],
    ]
    # Repair chatter is audit trail, not training data: no QA voice in the row.
    assert not any(
        "failed the contract" in m.get("content", "")
        for m in row["messages"]
        if m["role"] == "user"
    )
    assert (
        gate_violations(
            pack,
            row["messages"],
            mode="clean",
            fault=None,
        )
        == []
    )


def test_a_repair_that_reaches_for_a_tool_dies_at_the_repair_door():
    pack, pack_line = _pinned()
    replies, _policy = _clean_script(pack)
    replies.append(_reply("Bad: 8153.7729 percent invented."))
    replies.append(
        _reply(
            calls=[
                _call(
                    "get_fundamentals",
                    symbol="YCBK",
                    metrics=_fundamentals_metrics(pack),
                )
            ]
        )
    )
    outcome = render_agentic_row(Teacher(Scripted(replies)), pack_line, rank=RANK_CLEAN)
    dead = outcome["dead_letter"]
    assert dead["reason"].startswith("repair:")
    assert "only the final answer" in dead["reason"]


def test_three_bad_finals_and_the_row_is_dead_lettered_whole():
    pack, pack_line = _pinned()
    replies, _policy = _clean_script(pack)
    bad = _reply("Still bad: 8153.7729 percent.")
    replies += [bad, dict(bad), dict(bad)]
    outcome = render_agentic_row(Teacher(Scripted(replies)), pack_line, rank=RANK_CLEAN)
    dead = outcome["dead_letter"]
    assert dead["reason"].startswith("gate:")
    assert "8153.7729" in dead["reason"]
    assert dead["attempts"] == 5
    assert dead["temperatures"] == [
        config.AGENTIC_TEMPERATURES[0],
        config.AGENTIC_TEMPERATURES[0],
        config.AGENTIC_TEMPERATURES[0],
        config.AGENTIC_TEMPERATURES[1],
        config.AGENTIC_TEMPERATURES[2],
    ]
    # the exchange carries the whole audit trail the repair ladder left
    assert any(
        "failed the contract" in m.get("content", "")
        for m in dead["exchange"]
        if m["role"] == "user"
    )


# ---------------------------------------------------------------- the stage


def _stage_env(tmp_path, monkeypatch):
    out = str(tmp_path)
    monkeypatch.setenv(config.OUT_ENV, out)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, harness.DUMMY_MODEL)
    return out


def _seed_packs(out: str, jobs) -> int:
    """Materialise the fact-pack shard for *jobs* on the staged disk.

    The stage reads packs from disk like the pipeline does -- tests stage the
    disk, not the internals -- and the pack guard's own refusals (PackError)
    skip seeding exactly as the packs stage skipped them upstream: those
    coordinates never existed, and the render stage must see that absence the
    same way it would in production.
    """
    from pipelines.v3.packs import PackError

    by_work: dict[str, list[dict]] = {}
    for job in jobs:
        try:
            by_work.setdefault(job.work_type, []).append(
                stage.pack_record(job.work_type, job.family, job.variant)
            )
        except PackError:
            continue
    for work_type, lines in by_work.items():
        write.write_jsonl(write.path_for("fact_packs", work_type, out), lines)
    return sum(len(lines) for lines in by_work.values())


def test_the_stage_resumes_by_ids_and_asks_the_teacher_nobody(tmp_path, monkeypatch):
    payload = json.load(
        open(os.path.join(harness._HERE, harness.AGENTIC_FIXTURE_NAME), encoding="utf8")
    )
    out = _stage_env(tmp_path, monkeypatch)
    jobs = inventory.expand_jobs(inventory.load_plan(config.taxonomy_path()))
    seeded = _seed_packs(
        out, select_agentic_jobs(jobs, limit=payload["meta"]["ranks_walked"])
    )
    assert seeded > 0, "the stage seeded nothing; the fixture's slice is out of reach"
    counter = CountingFixture(payload["entries"])
    teacher = Teacher(counter)
    first = run_agentic_stage(out, jobs, teacher, limit=payload["meta"]["ranks_walked"])
    assert first["rendered"] == payload["meta"]["limit"]
    assert first["no_call"] == payload["meta"]["no_call"]
    assert first["faulted"] == payload["meta"]["faulted"]
    asked_after_first = counter.posts
    second = run_agentic_stage(
        out, jobs, teacher, limit=payload["meta"]["ranks_walked"]
    )
    assert second["rendered"] == 0
    assert second["existing"] == payload["meta"]["limit"]
    assert counter.posts == asked_after_first, "resume asked the teacher somebody"
    assert (
        len(write.read_jsonl(write.path_for("sft", "agentic", out)))
        == payload["meta"]["limit"]
    )


def test_the_missing_pack_file_is_fatal_and_the_refused_variant_is_data(
    tmp_path, monkeypatch
):
    """The two absences, kept apart: no file at all is the DAG out of order
    (fatal, ``missing_packs``); a variant the pack guard refused is that
    guard's own verdict (recorded, ``skipped_by_pack_gate``)."""
    out = _stage_env(tmp_path, monkeypatch)
    jobs = select_agentic_jobs(
        inventory.expand_jobs(inventory.load_plan(config.taxonomy_path()))
    )
    one_job = jobs[:1]
    teacher = Teacher(FixtureTransport(entries={}))
    missing = run_agentic_stage(out, one_job, teacher)
    assert missing["missing_packs"], "no pack file must be reported, not swallowed"
    assert not missing["skipped_by_pack_gate"]
    assert missing["missing_packs"][0]["reason"].startswith("no fact-pack file")
    # now the file exists; it simply does not carry this coordinate
    job = one_job[0]
    write.write_jsonl(
        write.path_for("fact_packs", job.work_type, out),
        [
            {
                "id": "unused-but-required-by-the-writer",
                "work_type": job.work_type,
                "scenario_id": "elsewhere",
                "variant": 9999,
            }
        ],
    )
    skipped = run_agentic_stage(out, one_job, teacher)
    assert not skipped["missing_packs"]
    assert skipped["skipped_by_pack_gate"]
    assert "pack guard" in skipped["skipped_by_pack_gate"][0]["reason"]


def test_the_selector_keeps_holdouts_out_and_the_mix_is_the_schedules():
    jobs = select_agentic_jobs(
        inventory.expand_jobs(inventory.load_plan(config.taxonomy_path()))
    )
    assert jobs and all(not job.holdout for job in jobs)
    modes = [faults.schedule_of(rank).mode for rank in range(len(jobs))]
    assert modes.count("no_call") == len(jobs) // config.AGENTIC_STRIDE
    assert modes.count("faulted") == len(jobs) // config.AGENTIC_STRIDE
    assert all(job.record_type == "agentic" for job in jobs)


def test_limit_counts_ranks_not_rendered_rows():
    """``--limit`` bounds the examination, so pack-guard refusals inside the
    window shrink the corpus honestly rather than silently extending it."""
    jobs = select_agentic_jobs(
        inventory.expand_jobs(inventory.load_plan(config.taxonomy_path()))
    )
    assert len(select_agentic_jobs(jobs, limit=3)) == 3
    assert select_agentic_jobs(jobs, limit=0) == []
