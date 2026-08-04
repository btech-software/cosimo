"""The shipped assistant prompt suites are hand-written assets, so they are
checked like assets: parseable, uniquely keyed, and structurally consistent with
what scripts/09_assistant_eval.py and cosimo_ft.assistant expect of them.
"""

from __future__ import annotations

import json

import pytest

from cosimo_ft import assistant
from cosimo_ft import config as config_mod

SUITE_NAMES = ("open_ended", "calibration", "agentic")


@pytest.fixture(scope="module")
def cfg():
    return config_mod.load_config(stage="assistant")


def load(cfg, name):
    path = config_mod.harness_path(config_mod.get(cfg, f"assistant.suites.{name}"))
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_suite_parses_and_is_non_empty(cfg, name):
    rows = load(cfg, name)
    assert rows, f"{name} is empty"
    assert all(row.get("id") and row.get("prompt") for row in rows)


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_suite_ids_are_unique(cfg, name):
    rows = load(cfg, name)
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", SUITE_NAMES)
def test_configured_suites_are_the_ones_that_run(cfg, name):
    assert name in config_mod.get(cfg, "assistant.run")


def test_calibration_prompts_all_declare_a_defect(cfg):
    """The defect kind is what makes a low abstention rate interpretable."""
    kinds = {"underspecified", "unanswerable", "false_premise"}
    assert all(row.get("defect") in kinds for row in load(cfg, "calibration"))


def test_agentic_scenarios_are_internally_consistent(cfg):
    for row in load(cfg, "agentic"):
        offered = {t["function"]["name"] for t in row["tools"]}
        assert len(offered) >= 2, f"{row['id']}: needs >1 tool to test selection"
        expected = set(row.get("expected_calls", []))
        assert expected <= offered, f"{row['id']}: expects an unoffered tool"
        if row.get("no_call"):
            assert not expected, f"{row['id']}: no_call scenario expects calls"
            continue
        assert expected, f"{row['id']}: call scenario expects no calls"
        # Every expected call needs a mock result, or the conversation stalls.
        assert expected <= set(row.get("results", {})), f"{row['id']}: missing results"
        assert row.get("expected_final"), f"{row['id']}: no result substrings to check"


def test_agentic_suite_covers_all_three_behaviours(cfg):
    """Single-call, multi-call and no-call must all be represented, or the
    aggregate hides the behaviour that actually regressed."""
    kinds = {row.get("kind") for row in load(cfg, "agentic")}
    assert {"single", "multi", "no_call"} <= kinds


def test_open_ended_prompts_do_not_ask_for_a_single_number(cfg):
    """A prompt that requests one figure invites exam format legitimately, which
    would make the exam_shape_rate uninterpretable."""
    for row in load(cfg, "open_ended"):
        assert "FINAL ANSWER" not in row["prompt"].upper()


def test_shipped_vocabulary_loads_and_covers_the_glossary(cfg):
    vocabulary = assistant.load_vocabulary(assistant.default_vocabulary_paths(cfg))
    assert len(vocabulary) > 200
    for term in ("sharpe ratio", "black scholes", "durbin watson", "nash equilibrium"):
        assert term in vocabulary


def test_the_motivating_failure_is_still_flagged(cfg):
    """'Durbin-Watson duration' must read as unknown even though 'Durbin-Watson'
    is a known term: the invention is the collocation, not the eponym."""
    vocabulary = assistant.load_vocabulary(assistant.default_vocabulary_paths(cfg))
    unknown = assistant.unknown_terms(
        "Measure the liability's Macaulay and Durbin-Watson durations.", vocabulary
    )
    assert any("Durbin-Watson" in term for term in unknown)
