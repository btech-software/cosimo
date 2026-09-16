"""The verifiable reward: what it pays for, and what it refuses to pay at all."""

from __future__ import annotations

import pytest

from cosimo_ft import reward

PACK = {
    "canonical": {"var95": 39.248, "es95": 48.957, "book": 1000.4},
    "conventions": [{"says": "the Basel multiplier starts at 3", "value": 3}],
    "must_mention": ["expected shortfall"],
    "as_of": "2026-03-31",
}


def row(kind="analysis", register="risk_committee", **over):
    base = {
        "record_type": kind,
        "register": register,
        "fact_pack": dict(PACK),
        "id": f"cosimov3_{kind}_0",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def test_an_invented_figure_zeroes_the_reward_however_good_the_rest():
    """The corpus's cardinal sin, priced as one.

    Gating rather than averaging is the whole decision here: a graded penalty
    would teach that a fabricated figure is survivable when the prose around
    it is strong, which is the failure this corpus exists to prevent.
    """
    good = "The expected shortfall is 48.957 against a 39.248 limit."
    assert reward.verifiable_reward(row(), good).total == 1.0
    bad = good + " Peers run at 61.3."
    scored = reward.verifiable_reward(row(), bad)
    assert scored.total == 0.0
    assert scored.gated and "invented_numbers" in scored.gates_tripped[0]


def test_the_pack_s_own_conventions_are_not_inventions():
    """A row may cite the constants its work type declared in advance.

    Reading ``canonical`` alone is not a stricter gate but a broken one: the
    evaluation records a live answer with the right arithmetic scored 1.000
    invented for citing its own conventions.
    """
    text = "The expected shortfall limit is 48.957; the Basel multiplier starts at 3."
    assert reward.verifiable_reward(row(), text).total == 1.0


def test_a_hyphen_is_not_a_minus_sign():
    """ "a 1-in-100 day" is two figures the pack allows, not an invented -100.

    The corpus dead-lettered a live VaR row three times over for exactly this,
    and reading the raw tokenizer here was this module's costliest drift.
    """
    text = "The expected shortfall limit is 48.957 on a 1-in-100 day."
    assert reward.verifiable_reward(row(), text).total == 1.0


def test_exam_liturgy_outside_an_exam_zeroes_the_reward():
    scored = reward.verifiable_reward(
        row(), "The expected shortfall limit is 48.957.\nFINAL ANSWER: 48.957"
    )
    assert scored.total == 0.0
    assert "exam_shape" in scored.gates_tripped


def test_a_collapsed_register_zeroes_the_reward():
    """The aggressive half of the policy, and recorded as such.

    ``register_violations`` is a shallow shape check by its own author's
    admission and RL is good at finding shallow checks, so this gate is the
    one to watch: ``gates_tripped`` is what makes it watchable.
    """
    scored = reward.verifiable_reward(
        row(), "The expected shortfall limit is 48.957, and we would buy the dip."
    )
    assert scored.total == 0.0
    assert any(g.startswith("register") for g in scored.gates_tripped)


# --------------------------------------------------------------------------
# per-type contracts
# --------------------------------------------------------------------------


def test_coverage_follows_the_record_type_not_the_pack():
    """An abstention owes the name of what is missing, not the question's points."""
    pack = dict(PACK, abstention_missing="the decision price")
    refusal = (
        "The decision price is missing, so the shortfall limit cannot be measured."
    )
    assert (
        reward.verifiable_reward(row("abstention", fact_pack=pack), refusal).total
        == 1.0
    )
    # Held to the analysis contract instead, the same words engage nothing.
    assert (
        reward.verifiable_reward(row("analysis", fact_pack=pack), refusal).total == 0.0
    )


def test_an_exam_row_is_graded_on_its_key_and_not_its_working():
    item = row(
        "exam",
        options=[
            {"label": "A", "value": 13.82, "text": "13.82"},
            {"label": "B", "value": 12.82, "text": "12.82"},
        ],
        answer_value=13.82,
        fact_pack={"canonical": {"cost": 13.82, "fee": 12.82}},
    )
    assert reward.verifiable_reward(item, "Working.\nFINAL ANSWER: A").total == 1.0
    # A distractor is the specific error the item was built to catch.
    assert reward.verifiable_reward(item, "Working.\nFINAL ANSWER: B").total < 1.0


def test_an_exam_row_may_close_on_its_tag_without_that_being_a_leak():
    """The liturgy is an exam item's contract; only elsewhere is it a crime."""
    item = row(
        "exam",
        options=[{"label": "A", "value": 1.0}],
        answer_value=1.0,
        fact_pack={"canonical": {"x": 1.0}},
    )
    assert not reward.verifiable_reward(item, "FINAL ANSWER: A").gated


def test_implementation_without_a_sandbox_is_unscoreable_not_failed():
    """A 0.0 here would be a signal invented by the absence of a tool.

    ``run_tests`` is injected rather than imported so that executing
    model-written code stays the caller's decision, made once with a sandbox
    it trusts -- and so that importing this module is never a code-execution
    risk.
    """
    impl = row(
        "implementation",
        register="desk_chat",
        hidden_tests=["assert f(1) == 1"],
        fact_pack={"canonical": {"x": 1.0}},
    )
    scored = reward.verifiable_reward(impl, "def f(x): return x")
    assert scored.gates_tripped == ["unscoreable"]

    scored = reward.verifiable_reward(
        impl, "def f(x): return x", run_tests=lambda text, tests: (1, 1)
    )
    assert scored.total == 1.0


def test_an_unknown_record_type_scores_nothing_rather_than_a_default():
    scored = reward.verifiable_reward(
        row("sonnet", register=""), "Shall I compare thee"
    )
    assert scored.total == 0.0
    assert scored.gates_tripped == ["unknown_record_type"]


@pytest.mark.parametrize("kind", sorted(reward.PROSE_KINDS))
def test_every_prose_kind_is_scoreable(kind):
    pack = dict(PACK, abstention_missing="expected shortfall")
    scored = reward.verifiable_reward(
        row(kind, fact_pack=pack), "The expected shortfall limit is 48.957."
    )
    assert scored.total == pytest.approx(1.0)
