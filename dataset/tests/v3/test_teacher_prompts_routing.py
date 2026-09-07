"""Routing matches spec §5.4 to the cell; briefs are packs' facts, not vibes.

The routing table is a *contract with the fleet*: an SGLang box answering the
reasoning lane and a vLLM box answering the prose lane must disagree only in
the model behind the name, never in which record types reach which lane, and
neither may ever let a rejected-side call keep the think flag on. Likewise
briefs: a teacher's prompt that drifted one field of the pack from the pack
itself is the drift that becomes the whole "two house styles and you called it
diversity" incident (spec §12, axis 13).
"""

from __future__ import annotations

import json

import pytest

from pipelines.v3.packs import compute_pack
from pipelines.v3.teacher import prompts, routing
from pipelines.v3.teacher.prompts import TEACHER_SYSTEM, render_brief, render_repair

ALL_TYPES = (
    "exam",
    "implementation",
    "agentic",
    "critique",
    "abstention",
    "analysis",
    "memo",
    "grounded",
)


def test_every_plan_routes_the_spec_table_exactly():
    expected = {
        "exam": ("reasoning", True),
        "implementation": ("reasoning", True),
        "agentic": ("reasoning", True),
        "critique": ("reasoning", True),
        "abstention": ("reasoning", False),
        "analysis": ("prose", True),
        "memo": ("prose", True),
        "grounded": ("prose", True),
    }
    for record_type, (lane, think) in expected.items():
        decision = routing.route(record_type)
        assert (decision.lane, decision.think) == (lane, think), record_type
        assert decision.record_type == record_type


def test_no_plan_routes_an_unknown_record_type(monkeypatch):
    monkeypatch.delenv("TEACHER_REASONING", raising=False)
    monkeypatch.delenv("TEACHER_PROSE", raising=False)
    with pytest.raises(routing.RoutingError, match="no teacher lane"):
        routing.route("hot_take")


def test_implementations_follow_the_environment_per_lane(monkeypatch):
    monkeypatch.setenv("TEACHER_REASONING", "custom-deep")
    monkeypatch.setenv("TEACHER_PROSE", "custom-shallow")
    assert routing.route("exam").model == "custom-deep"
    assert routing.route("memo").model == "custom-shallow"
    monkeypatch.delenv("TEACHER_REASONING", raising=False)
    monkeypatch.delenv("TEACHER_PROSE", raising=False)
    from pipelines.v3 import config

    assert routing.route("exam").model == config.TEACHER_MODEL_REASONING_DEFAULT
    assert routing.route("memo").model == config.TEACHER_MODEL_PROSE_DEFAULT


def test_rejected_sides_take_their_parents_model_and_lose_their_think():
    parent = routing.route("analysis")
    child = routing.route("analysis", rejected=True)
    assert child.model == parent.model
    assert child.think is False and parent.think is True
    abstention_child = routing.route("abstention", rejected=True)
    assert abstention_child.think is False


@pytest.mark.parametrize("kind", prompts._BRIEF_KINDS)
def test_briefs_render_every_pack_field_that_matters(kind):
    pack = compute_pack("valuation.equity.dcf", "mature_consumer", 0).to_dict()
    messages = render_brief(pack, kind=kind)
    assert [turn["role"] for turn in messages] == ["system", "user"]
    assert messages[0]["content"] == TEACHER_SYSTEM
    user = messages[1]["content"]
    assert pack["question"] in user
    assert f"'{pack['scenario_id']}'" in user or pack["scenario_id"] in user
    for key in ("must_mention", "forbidden_claims", "allowed_numbers", "fact_pack"):
        assert key in user
    low, high = prompts._WORD_BUDGETS[kind]
    assert f"{low}-{high} words" in user
    assert prompts._REGISTER_HINTS[pack["register"]] in user
    assert pack["as_of"] in user


def test_briefs_are_canonical_json_reversible_and_deterministic():
    pack = compute_pack("valuation.equity.dcf", "cyclical_industrial", 2).to_dict()
    first = render_brief(pack, kind="analysis")
    second = render_brief(pack, kind="analysis")
    assert first == second, "prompts must be a pure function of the pack"
    # the contract blob is the last json object in the user turn; it must
    # round-trip, or PR2's dead-letter post-mortems will be reading prose.
    user = first[1]["content"]
    blob = (
        user[user.index('{"as_of"') :]
        if '{"as_of"' in user
        else user[user.index("{") :]
    )
    decoded = json.loads(blob)
    assert decoded["task"] == "analysis"
    assert decoded["fact_pack"]["scenario_id"] == pack["scenario_id"]


def test_briefs_refuse_kinds_that_are_not_briefs():
    pack = compute_pack("valuation.equity.dcf", "mature_consumer", 0).to_dict()
    with pytest.raises(ValueError, match="no prose brief kind"):
        render_brief(pack, kind="memo_" + " ")
    with pytest.raises(ValueError, match="no prose brief kind"):
        render_brief(pack, kind="agentic")


def test_repairs_carry_the_violations_and_the_original_transcript():
    pack = compute_pack("valuation.equity.dcf", "mature_consumer", 0).to_dict()
    messages = render_brief(pack, kind="analysis")
    repaired = render_repair(
        messages, "my draft said 7% growth", ["invented 7%", "missed point"]
    )
    assert repaired[:2] == messages
    assert repaired[2]["role"] == "assistant" and "7% growth" in repaired[2]["content"]
    assert repaired[3]["role"] == "user"
    assert "invented 7%" in repaired[3]["content"]
    assert "missed point" in repaired[3]["content"]
    assert messages[0]["content"] == TEACHER_SYSTEM, "repair must not mutate the brief"
