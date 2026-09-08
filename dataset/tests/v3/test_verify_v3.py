"""The verification board must fail on every tamper it claims to catch (spec §6).

A gate nobody has watched open catches nothing: each test here renders a
small real corpus through the *production* path (fixture transport, real
stage, real writers), then breaks one property of one row on disk and demands
that exactly the owning axis light up -- not its neighbours, not none.

The distinction ``test_only_the_failing_axis_reports`` enforces is the one
that makes an exit-1 gate usable at 3am: "axis 2 failed" is a direction,
"verify failed" is a shrug.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.render.prose import run_render_stage, select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verification.invented_numbers import invented_numbers  # noqa: E402
from pipelines.v3.verify_v3 import AXES, verify_dir  # noqa: E402

N_ROWS = 6
LANE_ENVS = {
    config.TEACHER_PROSE_ENV: prose_harness.DUMMY_MODEL,
    config.TEACHER_REASONING_ENV: prose_harness.DUMMY_MODEL,
    config.LIVE_ENV: "0",
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """One rendered corpus, built once through the production path."""
    out = str(tmp_path_factory.mktemp("v3corpus"))
    payload = prose_harness.build_fixture(("analysis",), N_ROWS)
    jobs = inventory.expand_jobs(inventory.load_plan())
    selected = select_prose_jobs(jobs, types=("analysis",), limit=None)[
        : payload["meta"]["jobs_examined"]
    ]
    stage.run_pack_stage(out, selected)
    saved = {key: os.environ.get(key) for key in LANE_ENVS}
    os.environ.update(LANE_ENVS)
    try:
        run_render_stage(
            out, selected, Teacher(FixtureTransport(entries=payload["entries"]))
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return out


def _fresh(corpus_dir, tmp_path) -> str:
    copy = os.path.join(str(tmp_path), "corpus")
    shutil.copytree(corpus_dir, copy)
    return copy


def _rows(out: str) -> list[dict]:
    return write.read_jsonl(write.path_for("sft", "analysis", out))


def _rewrite(out: str, rows: list[dict]) -> None:
    write.write_jsonl(write.path_for("sft", "analysis", out), rows)


def _fail_axes(report: dict) -> list[str]:
    return [name for _, name in AXES if report["axes"][name]["failures"]]


def test_the_pristine_corpus_passes_and_counts_every_row(corpus):
    report = verify_dir(corpus)
    assert report["ok"] is True
    assert report["rows"] == N_ROWS
    for _, name in AXES:
        assert report["axes"][name]["checked"] == N_ROWS


def test_the_untampered_copy_fails_no_axis(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    report = verify_dir(out)
    assert _fail_axes(report) == [], "the fixture corpus itself must be the control"


def test_answer_drifted_from_the_assistant_turn_is_a_schema_incident(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    rows[0]["answer"] = rows[0]["answer"] + " "
    _rewrite(out, rows)
    report = verify_dir(out)
    assert not report["ok"]
    assert _fail_axes(report) == ["schema"]


def test_a_row_that_no_longer_matches_its_seed_fails_recompute(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    rows[0]["variant"] = rows[0]["variant"] + 1000
    _rewrite(out, rows)
    report = verify_dir(out)
    # axis 2 owns this tamper; axes that *read through* the pack (invented
    # numbers graded against the re-derived authority) may follow it -- the
    # contract is "the owner must light", not "exactly one axis may light"
    assert "pack recompute" in _fail_axes(report)
    problems = " | ".join(
        f["problem"] for f in report["axes"]["pack recompute"]["failures"]
    )
    assert (
        "drifted" in problems
        or "no longer computable" in problems
        or "hash" in problems
    )


def test_an_invented_number_in_the_answer_is_named_by_axis_three(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    row = rows[0]
    family = row["scenario_id"][len(row["work_type"]) + 1 :]
    pack = compute_pack(row["work_type"], family, row["variant"])
    assert "9751.6362" in invented_numbers(
        "The breakeven is 9751.6362.", pack.allowed_numbers
    ), "the chosen token must be unambiguously foreign to this pack"
    tampered = row["answer"] + " The breakeven is 9751.6362."
    row["answer"] = tampered
    row["messages"][-1]["content"] = tampered
    _rewrite(out, rows)
    report = verify_dir(out)
    assert _fail_axes(report) == ["invented numbers"]
    assert any(
        "9751.6362" in f["problem"]
        for f in report["axes"]["invented numbers"]["failures"]
    )


def test_a_silent_must_mention_fails_axis_four_only(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    replacement = "The desk has reviewed the file and feels comfortable."
    assert not invented_numbers(replacement, [0.5]), "the filler must be number-free"
    rows[0]["answer"] = replacement
    rows[0]["messages"][-1]["content"] = replacement
    _rewrite(out, rows)
    report = verify_dir(out)
    assert _fail_axes(report) == ["must_mention / forbidden_claims"]


def test_the_exam_tag_in_prose_fails_axis_five_only(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    tampered = rows[0]["answer"] + " FINAL ANSWER: cut the position."
    rows[0]["answer"] = tampered
    rows[0]["messages"][-1]["content"] = tampered
    _rewrite(out, rows)
    report = verify_dir(out)
    assert _fail_axes(report) == ["FINAL ANSWER is exam-only"]


def test_a_row_both_live_and_dead_lettered_is_a_lie_worth_failing_on(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    rows = _rows(out)
    write.append_unique(
        write.path_for("dead_letter", "analysis", out),
        [{"id": rows[1]["id"], "record_type": "analysis", "reason": "planted"}],
    )
    report = verify_dir(out)
    assert _fail_axes(report) == ["schema"]
    assert any(
        "simultaneously" in f["problem"] for f in report["axes"]["schema"]["failures"]
    )


def test_a_corrupt_line_fails_the_board_not_the_reader_silence(corpus, tmp_path):
    out = _fresh(corpus, tmp_path)
    path = write.path_for("sft", "analysis", out)
    with open(path, "a", encoding="utf8") as handle:
        handle.write("this is not json\n")
    report = verify_dir(out)
    assert not report["ok"]
    assert report["axes"]["schema"]["failures"], "the path and line must be named"
    assert any("not json" in f["problem"] for f in report["axes"]["schema"]["failures"])


def test_an_empty_tree_proves_nothing_and_passes_nothing(tmp_path):
    report = verify_dir(str(tmp_path / "never-rendered"))
    assert report["ok"] is True and report["rows"] == 0, (
        "absence of rows is not absence of defects: the CLI, not the board, "
        "decides whether an empty corpus is acceptable for a given run"
    )
