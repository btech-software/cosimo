"""The implementation contract, executable (analysis spec §5.9; arch spec §6 axis 11).

Five claims, each a test below, in the order the doctrine earns them:

* the generated suites **pass** -- the static references are faithful mirrors
  of the computers, to the printed digit (and the vacuity check proves the
  suites could have failed: no expected figure is baked into an instrument);
* the suites have **teeth**: a pin moved is a caught failure;
* a pack whose suite does not pass dead-letters **before the teacher is
  billed** -- a table defect is not a draft to repair;
* the board **reports tampered bytes, does not execute them** -- proven by a
  sentinel file the refused code would have touched;
* a replay **asks nobody** -- ids gate the bill, the second run is free.

The scripted transports are the prose tests' idiom: a policy (attempts, the
temperature ladder, the repair turn) is only testable with a teacher whose
behaviour the test chooses.
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
import pipelines.v3.render.implementation as render_impl  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.implementation import (  # noqa: E402
    IMPL_KIND,
    build_impl_row,
    run_impl_stage,
    select_impl_jobs,
)
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verification.implementation import (  # noqa: E402
    compose_impl_item,
    limitations_violations,
    run_sandboxed,
)
from pipelines.v3.verify_v3 import _check_row  # noqa: E402

PINNED = ("valuation.equity.dcf", "mature_consumer", 0)


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

    def __init__(self, path):
        super().__init__(path=path)
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


def _pack_and_line():
    pack = compute_pack(*PINNED).to_dict()
    return pack, stage.pack_record(*PINNED)


def _clean(pack):
    """A limitations passage this pack's gate accepts, from the harness itself."""
    return impl_harness.limitations_text(pack)


# ------------------------------------------------------------------ the contract


def test_the_generated_suite_passes_the_reference_on_the_pinned_pack():
    pack, _ = _pack_and_line()
    item = compose_impl_item(pack)
    passed, log = run_sandboxed(
        item["reference_code"],
        [*item["public_tests"], *item["hidden_tests"]],
        pack["inputs"],
    )
    assert passed, log


def test_a_moved_pin_is_a_caught_failure_not_a_shrug():
    """The suites have teeth: drift one expected figure and the sandbox says so."""
    pack, _ = _pack_and_line()
    item = compose_impl_item(pack)
    moved = [t.replace("want = ", "want = 1.5 * ", 1) for t in item["public_tests"]]
    passed, log = run_sandboxed(item["reference_code"], moved, pack["inputs"])
    assert not passed
    assert "enterprise_value_m" in log  # names the figure that drifted


def test_the_reference_carries_no_expected_answer():
    """No computed figure appears in the instrument that must reproduce it.

    Checked on the figures distinctive enough that no incidental digit of the
    source's own arithmetic could counterfeit them (values >= 100 in full).
    """
    pack, _ = _pack_and_line()
    item = compose_impl_item(pack)
    for key, value in (pack.get("computed") or {}).items():
        if isinstance(value, (int, float)) and abs(value) >= 100:
            spelling = repr(float(value))
            assert spelling not in item["reference_code"], (
                f"the reference embeds {key}={spelling}; the suite would then "
                "prove the instrument quotes itself, which proves nothing"
            )


def test_the_dirty_fixture_is_demanded_refused():
    pack, _ = _pack_and_line()
    item = compose_impl_item(pack)
    assert any("the dirty fixture was solved" in test for test in item["hidden_tests"])
    assert item["dirty_fixture"] != pack["inputs"], "the dirt must be drawn through"


def test_the_composition_is_provable_from_either_provenance():
    """The stored pack line and the recomputed pack compose identical bytes.

    The renderer composes from the shard the packs stage wrote; the board
    re-derives from the computers. One record, two readings -- if the
    recomposition were order-sensitive the board would flog honest rows.
    """
    pack, pack_line = _pack_and_line()
    from_disk = {k: v for k, v in pack_line.items() if k not in ("id", "verification")}
    assert compose_impl_item(from_disk) == compose_impl_item(pack)


# ------------------------------------------------------------------------ the stage


def test_clean_first_attempt_ships_the_whole_record():
    pack, pack_line = _pack_and_line()
    passage = _clean(pack)
    transport = Scripted([_reply(passage, think="compared against the pack")])
    outcome = build_impl_row(Teacher(transport), pack_line)
    row = outcome["row"]
    assert outcome["dead_letter"] is None
    assert row["record_type"] == IMPL_KIND
    assert row["answer"] == row["reference_code"]
    assert row["messages"][-1]["content"] == row["reference_code"]
    assert row["limitations"] == passage
    assert row["public_tests"] and row["hidden_tests"]
    assert row["dirty_fixture"] != pack["inputs"]
    assert row["verification"]["render"]["sandbox"] == "passed"
    assert row["verification"]["render"]["attempts"] == 1
    assert row["verification"]["teacher"]["model"] == "scripted-t"


def test_a_failing_suite_dead_letters_before_the_teacher_is_billed():
    """A pack whose suite fails its own instrument is a table defect.

    The stage must measure before it bills: no teacher call may be spent
    repairing limitations around an instrument that cannot pass its own
    tests, and the dead letter must say what broke.
    """
    pack_line = stage.pack_record(*PINNED)
    transport = Scripted([])  # any call at all is the failure this test names
    original = render_impl.run_sandboxed
    render_impl.run_sandboxed = lambda *_, **__: (False, "instruments do not rank")
    try:
        outcome = build_impl_row(Teacher(transport), pack_line)
    finally:
        render_impl.run_sandboxed = original
    assert outcome["row"] is None
    assert outcome["dead_letter"]["reason"].startswith("sandbox: instruments")
    assert transport.calls == [], "a table defect is not a draft; nobody was asked"


def test_a_shrug_of_limitations_costs_an_attempt_not_a_row():
    pack, pack_line = _pack_and_line()
    transport = Scripted(
        [
            _reply("It assumes things."),  # too short for the floor
            _reply(impl_harness.limitations_text(pack)),
        ]
    )
    outcome = build_impl_row(Teacher(transport), pack_line)
    assert outcome["dead_letter"] is None
    assert outcome["row"]["verification"]["render"]["attempts"] == 2
    assert len(transport.calls) == 2
    assert transport.calls[1]["temperature"] == config.IMPL_TEMPERATURES[1]


def test_a_shrug_of_limitations_is_returned_unshipped():
    pack, _ = _pack_and_line()
    problems = limitations_violations(pack, "No limits.")
    assert any("shrug" in problem for problem in problems)


def test_a_figure_out_of_thin_air_is_answered_for():
    """Long enough to clear the floor, the fact-lock still catches the coinage."""
    pack, _ = _pack_and_line()
    passage = (
        "The instrument reports 8677.13 as the value of the subject and "
        "states it to be exact under every reading taken together."
    )
    problems = limitations_violations(pack, passage)
    assert any("8677.13" in problem for problem in problems)


def test_exhausted_attempts_dead_letter_the_whole_exchange():
    pack, pack_line = _pack_and_line()
    strikes = [_reply("Too brief.") for _ in range(config.IMPL_ATTEMPTS)]
    transport = Scripted(strikes)
    outcome = build_impl_row(Teacher(transport), pack_line)
    dead = outcome["dead_letter"]
    assert outcome["row"] is None
    assert dead["reason"].startswith("gate:")
    assert dead["attempts"] == config.IMPL_ATTEMPTS
    assert dead["temperatures"] == list(
        config.IMPL_TEMPERATURES[: config.IMPL_ATTEMPTS]
    )
    assert [m["role"] for m in dead["exchange"][-2:]] == ["assistant", "user"]


# ------------------------------------------------------------------- the board


def test_tampered_bytes_are_reported_not_executed(tmp_path):
    """The board runs no code the corpus did not author -- proven, not hoped.

    The tampered reference reaches for the filesystem; if the board executed
    it before the recomposition check, the sentinel would exist after the
    audit. The audit is the experiment."""
    pack, pack_line = _pack_and_line()
    transport = Scripted([_reply(_clean(pack))])
    row = build_impl_row(Teacher(transport), pack_line)["row"]
    sentinel = os.path.join(str(tmp_path), "EXECUTED")
    row["reference_code"] = (
        f"import pathlib\npathlib.Path({sentinel!r}).touch()\n"
        "def solve(inputs):\n    return {}\n"
    )
    failures = _check_row(row, frozenset())
    assert any(p.startswith("impl recomposition:") for p in failures["schema"])
    assert any("not executed" in p for p in failures["hidden tests"])
    assert not os.path.lexists(sentinel), "the refused bytes were run anyway"


def test_a_clean_row_is_certified_by_the_board_too():
    pack, pack_line = _pack_and_line()
    transport = Scripted([_reply(_clean(pack))])
    row = build_impl_row(Teacher(transport), pack_line)["row"]
    failures = _check_row(row, frozenset())
    for axis, problems in failures.items():
        assert not problems, f"{axis}: {problems}"


# ------------------------------------------------------------------ the whole lane


def test_the_committed_lane_renders_once_and_replays_free(tmp_path, monkeypatch):
    """The committed train slice, end to end, through the real replay transport.

    Windowed to the fixture's own limit rather than the whole lane: with the
    amendment's WIP counts nothing truncates, so the lane plans 120
    implementation jobs against a table that commits the first 30, and
    `limit=None` would be asking the replay for bodies nobody captured.
    """
    committed = os.path.join(_HERE, "fixtures", impl_harness.IMPL_FIXTURE_NAME)
    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(inventory.load_plan())
    selected = select_impl_jobs(jobs, limit=impl_harness.DEFAULT_LIMIT)
    stage.run_pack_stage(out, selected)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, impl_harness.DUMMY_MODEL)
    monkeypatch.delenv(config.LIVE_ENV, raising=False)

    # The committed table's own size, not a literal: it is a plan number, and
    # the plan's WIP counts move it (see the docstring).
    expected = json.load(open(committed, encoding="utf8"))["meta"]["entries"]
    first = run_impl_stage(out, selected, Teacher(CountingFixture(committed)))
    assert first["rendered"] == expected and first["dead_lettered"] == 0
    rows = write.read_jsonl(write.path_for("sft", IMPL_KIND, out))
    assert len(rows) == expected == len({row["id"] for row in rows})
    assert all(row["verification"]["render"]["sandbox"] == "passed" for row in rows)

    replay = CountingFixture(committed)
    second = run_impl_stage(out, selected, Teacher(replay))
    assert second["rendered"] == 0 and second["existing"] == expected
    assert replay.calls == 0, "a row already on disk may not recall the teacher"
