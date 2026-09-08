"""The render/verify CLI contract (spec §7): exit codes Airflow can schedule on.

``exit 0`` must mean "the bytes on disk are good" -- so these tests drive
``cli.main`` exactly as a DAG task would and assert the code, not the prose:
a stage that prints a happy table and exits 1 on a clean corpus, or exits 0
over a tampered one, breaks the pipeline in ways a unit test on the stage
cannot see.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import cli, config, inventory, stage, write  # noqa: E402


def _outerr(capsys) -> str:
    """pytest 9's capture API returns a result object; read it once per test."""
    captured = capsys.readouterr()
    return captured.out + captured.err


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """``plan.json`` + ``fact_packs/`` built once per module -- every CLI test
    that needs an upstream DAG copies this tree instead of paying the pack
    stage again (it is the slow honest part of the pipeline, not the part
    under test here)."""
    out = str(tmp_path_factory.mktemp("v3packed"))
    plan = os.path.join(out, "plan.json")
    assert cli.main(["inventory", "--out", plan]) == cli.EXIT_OK
    stage.run_pack_stage(out, inventory.expand_jobs(inventory.load_plan()))
    return out, plan


def _copy(packed_pair, tmp_path) -> tuple[str, str]:
    out = os.path.join(str(tmp_path), "corpus")
    shutil.copytree(packed_pair[0], out)
    return out, os.path.join(out, "plan.json")


def _bare_plan(tmp_path) -> tuple[str, str]:
    """A corpus root with a plan and deliberately *no* packs stage yet."""
    out = str(tmp_path / "fresh")
    plan = os.path.join(out, "plan.json")
    assert cli.main(["inventory", "--out", plan]) == cli.EXIT_OK
    return out, plan


def _fixture_file(tmp_path, payload) -> str:
    path = os.path.join(str(tmp_path), "prose_fixture.json")
    with open(path, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    return path


def _offline(monkeypatch, fixture_path: str) -> None:
    monkeypatch.delenv(config.LIVE_ENV, raising=False)
    monkeypatch.setenv(config.TEACHER_FIXTURE_ENV, fixture_path)
    monkeypatch.setenv(config.TEACHER_PROSE_ENV, prose_harness.DUMMY_MODEL)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, prose_harness.DUMMY_MODEL)


def _render(out: str, plan: str, payload, *extra: str) -> int:
    return cli.main(
        [
            "render",
            "--out",
            out,
            "--plan",
            plan,
            "--types",
            "analysis",
            "--limit",
            str(payload["meta"]["jobs_examined"]),
            *extra,
        ]
    )


def test_render_then_verify_exit_zero_over_a_clean_corpus(
    packed, tmp_path, monkeypatch, capsys
):
    out, plan = _copy(packed, tmp_path)
    payload = prose_harness.build_fixture(("analysis",), 6)
    _offline(monkeypatch, _fixture_file(tmp_path, payload))
    assert _render(out, plan, payload) == cli.EXIT_OK
    rendered = _outerr(capsys)
    assert "6 rows rendered" in rendered and "0 dead-lettered" in rendered
    assert cli.main(["verify", "--out", out]) == cli.EXIT_OK
    board = _outerr(capsys)
    assert "clean across 11 axes" in board
    assert len(write.read_jsonl(write.path_for("sft", "analysis", out))) == 6


def test_replay_reports_already_on_disk_and_bills_nobody(
    packed, tmp_path, monkeypatch, capsys
):
    out, plan = _copy(packed, tmp_path)
    payload = prose_harness.build_fixture(("analysis",), 6)
    _offline(monkeypatch, _fixture_file(tmp_path, payload))
    assert _render(out, plan, payload) == cli.EXIT_OK
    _outerr(capsys)  # drain the first run's board
    assert _render(out, plan, payload) == cli.EXIT_OK
    replayed = _outerr(capsys)
    assert "0 rows rendered" in replayed and "6 already on disk" in replayed


def test_render_without_packs_is_a_data_exit_naming_the_ordering_bug(
    tmp_path, monkeypatch, capsys
):
    out, plan = _bare_plan(tmp_path)
    payload = prose_harness.build_fixture(("analysis",), 2)
    _offline(monkeypatch, _fixture_file(tmp_path, payload))
    assert _render(out, plan, payload) == cli.EXIT_DATA
    assert "run the packs stage" in _outerr(capsys)


def test_unknown_type_is_a_usage_exit_not_a_data_one(tmp_path, monkeypatch, capsys):
    out, plan = _bare_plan(tmp_path)
    _offline(monkeypatch, _fixture_file(tmp_path, {"entries": {}}))
    rc = cli.main(["render", "--out", out, "--plan", plan, "--types", "preference"])
    assert rc == cli.EXIT_USAGE
    printed = _outerr(capsys)
    assert "the render stages cover" in printed
    assert "preference and eval slices arrive" in printed


def test_a_wildcard_cannot_smuggle_a_row_past_the_gate(
    packed, tmp_path, monkeypatch, capsys
):
    """The offline default must never *fabricate* compliance: a canned
    wildcard reply is gate-checked like any teacher's word -- "Understood."
    covers no must_mention point, so every row dies to the dead letter and
    the sft shard stays honestly empty."""
    out, plan = _copy(packed, tmp_path)
    canned = {
        "entries": {
            "*": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Understood.", "role": "assistant"},
                    }
                ],
                "model": prose_harness.DUMMY_MODEL,
                "usage": {"total_tokens": 1},
            }
        }
    }
    _offline(monkeypatch, _fixture_file(tmp_path, canned))
    assert (
        cli.main(
            [
                "render",
                "--out",
                out,
                "--plan",
                plan,
                "--types",
                "analysis",
                "--limit",
                "30",
            ]
        )
        == cli.EXIT_OK
    )
    deads = write.read_jsonl(write.path_for("dead_letter", "analysis", out))
    assert len(deads) >= 5, "the wildcard must answer, and the gate must judge"
    assert all("must_mention" in dead["reason"] for dead in deads)
    assert write.read_jsonl(write.path_for("sft", "analysis", out)) == []


def test_a_fixture_miss_is_loud_not_a_fallback(packed, tmp_path, monkeypatch, capsys):
    out, plan = _copy(packed, tmp_path)
    payload = prose_harness.build_fixture(("analysis",), 2)
    _offline(monkeypatch, _fixture_file(tmp_path, {"entries": {}}))
    assert _render(out, plan, payload) == cli.EXIT_DATA
    assert "fixture miss" in _outerr(capsys)


def test_verify_fails_the_exit_code_on_a_tampered_shard(
    packed, tmp_path, monkeypatch, capsys
):
    out, plan = _copy(packed, tmp_path)
    payload = prose_harness.build_fixture(("analysis",), 6)
    _offline(monkeypatch, _fixture_file(tmp_path, payload))
    assert _render(out, plan, payload) == cli.EXIT_OK
    _outerr(capsys)
    path = write.path_for("sft", "analysis", out)
    rows = write.read_jsonl(path)
    rows[0]["answer"] += " FINAL ANSWER: tampered."
    rows[0]["messages"][-1]["content"] = rows[0]["answer"]
    write.write_jsonl(path, rows)
    assert cli.main(["verify", "--out", out]) == cli.EXIT_DATA
    assert "VERIFY FAIL" in _outerr(capsys)


def test_live_flag_names_the_missing_env_instead_of_billing_it(
    tmp_path, monkeypatch, capsys
):
    out, plan = _bare_plan(tmp_path)
    monkeypatch.delenv(config.TEACHER_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(config.TEACHER_API_KEY_ENV, raising=False)
    monkeypatch.setenv(config.LIVE_ENV, "0")
    rc = cli.main(
        ["render", "--out", out, "--plan", plan, "--types", "analysis", "--live"]
    )
    assert rc == cli.EXIT_DATA
    err = _outerr(capsys)
    assert config.TEACHER_BASE_URL_ENV in err and config.TEACHER_API_KEY_ENV in err
