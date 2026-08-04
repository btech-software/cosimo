"""Behavioural metrics: exam-shape leakage, abstention, terminology, trajectories."""

from __future__ import annotations

import json

import pytest

from cosimo_ft import assistant

# The actual served response that motivated this module: exam format and an
# invented term, on a question that asked for judgement.
COLLAPSED_ANSWER = """When you have a long-duration liability, your balance sheet
typically shows a fixed-income position.

Step 1. Measure the liability's Macaulay and Durbin-Watson durations and convexity.
Step 2. Compare with the holding's durations and convexity.
Step 3. The mismatch is the difference.
Step 4. To fix the mismatch, add a parallel offset.
"""

PROSE_ANSWER = """Convexity mismatch shows up when your liabilities respond to
rate moves with more curvature than your assets do. The practical fix is usually
a receiver swaption overlay rather than more cash bonds, because you are buying
curvature rather than level exposure. I would size it against the second-order
term of the liability, not the first.
"""


# --------------------------------------------------------------------------
# exam shape
# --------------------------------------------------------------------------


def test_collapsed_answer_is_flagged():
    markers = assistant.exam_shape_markers(COLLAPSED_ANSWER)
    assert "numbered_steps" in markers
    assert assistant.has_exam_shape(COLLAPSED_ANSWER)


def test_ordinary_prose_is_not_flagged():
    assert assistant.exam_shape_markers(PROSE_ANSWER) == []
    assert not assistant.has_exam_shape(PROSE_ANSWER)


def test_a_couple_of_numbered_steps_are_not_collapse():
    """One or two enumerated steps is normal writing, not the corpus template."""
    text = "Step 1. Compute the duration.\nStep 2. Compare to the liability."
    assert "numbered_steps" not in assistant.exam_shape_markers(text)


def test_assumptions_and_final_answer_are_flagged_on_their_own():
    assert assistant.exam_shape_markers("ASSUMPTIONS: annual compounding.") == [
        "assumptions_header"
    ]
    assert assistant.exam_shape_markers("FINAL ANSWER: 12.5") == ["final_answer_tag"]


def test_exam_shape_handles_empty_input():
    assert assistant.exam_shape_markers("") == []
    assert not assistant.has_exam_shape("")


# --------------------------------------------------------------------------
# abstention
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Could you clarify which portfolio you mean?",
        "I don't know the current exposure without the position file.",
        "I'd need to know the risk-free rate and the holding period.",
        "There is not enough information to compute this.",
        "The horizon is not specified, so the number would be arbitrary.",
    ],
)
def test_abstentions_are_detected(text):
    assert assistant.is_abstention(text)


@pytest.mark.parametrize(
    "text",
    [
        "The Sharpe ratio is 0.57.",
        "FINAL ANSWER: 12.5",
        PROSE_ANSWER,
    ],
)
def test_confident_answers_are_not_abstentions(text):
    assert not assistant.is_abstention(text)


def test_a_confident_answer_with_a_late_caveat_is_not_an_abstention():
    """Committing first and hedging afterwards is politeness, not calibration."""
    text = "The Sharpe ratio is 0.57. " + "x" * 800 + " Of course, I'd need to know your horizon."
    assert not assistant.is_abstention(text)


# --------------------------------------------------------------------------
# terminology
# --------------------------------------------------------------------------


def test_vocabulary_loads_from_glossary_and_json(tmp_path):
    glossary = tmp_path / "glossary.txt"
    glossary.write_text("# comment\nSharpe Ratio\nBlack-Scholes  # trailing\n\n")
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps({"programs": {"X": {"topics": [{"topic": "Fixed Income"}]}}})
    )
    vocabulary = assistant.load_vocabulary([glossary, taxonomy])
    assert "sharpe ratio" in vocabulary
    assert "black scholes" in vocabulary
    assert "fixed income" in vocabulary


def test_missing_vocabulary_file_raises(tmp_path):
    """An empty vocabulary would make every term look invented."""
    with pytest.raises(FileNotFoundError):
        assistant.load_vocabulary([tmp_path / "nope.txt"])


def test_hyphenated_and_multiword_terms_are_candidates():
    terms = assistant.candidate_terms(
        "We compared the Sharpe Ratio against the Durbin-Watson statistic."
    )
    assert "Sharpe Ratio" in terms
    assert any(t.startswith("Durbin-Watson") for t in terms)


def test_known_terms_are_not_reported():
    vocabulary = assistant.load_vocabulary([])
    vocabulary |= {assistant.normalize_term("Sharpe Ratio")}
    assert assistant.unknown_terms("The Sharpe Ratio is high.", vocabulary) == []


def test_sentence_starters_do_not_become_terms():
    """'The Sharpe' must not be reported as an unknown compound term."""
    terms = assistant.candidate_terms("The Sharpe number looked wrong.")
    assert not any(t.startswith("The ") for t in terms)


def test_normalize_term_folds_case_hyphens_and_spacing():
    assert (
        assistant.normalize_term("Black-Scholes")
        == assistant.normalize_term("black  scholes")
        == "black scholes"
    )


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------


def _scenario(**overrides):
    base = {
        "expected_calls": ["get_quote"],
        "offered_tools": ["get_quote", "get_yield_curve"],
        "expected_final": ["178.42"],
    }
    base.update(overrides)
    return base


def test_correct_single_call_trajectory():
    grade = assistant.grade_trajectory(
        _scenario(),
        [{"name": "get_quote", "arguments": {"symbol": "NVDA"}}],
        "NVDA is at 178.42.",
    )
    assert grade["correct"]
    assert grade["selected_expected"]
    assert grade["hallucinated_tools"] == []


def test_calls_without_using_the_result_is_not_correct():
    """Emitting the call but never reporting the number is an incomplete chain."""
    grade = assistant.grade_trajectory(
        _scenario(),
        [{"name": "get_quote", "arguments": {"symbol": "NVDA"}}],
        "I have looked it up.",
    )
    assert not grade["correct"]
    assert grade["completed_chain"] is False


def test_multi_step_is_order_insensitive():
    grade = assistant.grade_trajectory(
        _scenario(
            expected_calls=["get_fundamentals", "get_risk_metrics"],
            offered_tools=["get_fundamentals", "get_risk_metrics"],
            expected_final=["28.4", "0.22"],
        ),
        [
            {"name": "get_risk_metrics", "arguments": {"symbol": "MSFT"}},
            {"name": "get_fundamentals", "arguments": {"symbol": "MSFT"}},
        ],
        "P/E is 28.4 and realised vol is 0.22.",
    )
    assert grade["correct"]


def test_missing_second_call_fails_multi_step():
    grade = assistant.grade_trajectory(
        _scenario(
            expected_calls=["get_fundamentals", "get_risk_metrics"],
            offered_tools=["get_fundamentals", "get_risk_metrics"],
            expected_final=["28.4", "0.22"],
        ),
        [{"name": "get_fundamentals", "arguments": {"symbol": "MSFT"}}],
        "P/E is 28.4.",
    )
    assert not grade["correct"]
    assert grade["selected_expected"] is False


def test_no_call_scenario_is_correct_only_without_calls():
    scenario = _scenario(no_call=True, expected_calls=[], expected_final=[])
    assert assistant.grade_trajectory(scenario, [], "Duration measures...")["correct"]
    assert not assistant.grade_trajectory(
        scenario, [{"name": "get_quote", "arguments": {}}], ""
    )["correct"]


def test_unoffered_tool_is_reported_as_hallucinated():
    grade = assistant.grade_trajectory(
        _scenario(),
        [{"name": "get_moon_phase", "arguments": {}}],
        "",
    )
    assert grade["hallucinated_tools"] == ["get_moon_phase"]


def test_non_dict_arguments_are_invalid():
    grade = assistant.grade_trajectory(
        _scenario(),
        [{"name": "get_quote", "arguments": "NVDA"}],
        "NVDA is at 178.42.",
    )
    assert grade["arguments_valid"] is False
    assert not grade["correct"]


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_summarize_open_ended_counts_rates_and_ranks_terms():
    rows = [
        {
            "exam_shape": True,
            "exam_shape_markers": ["numbered_steps"],
            "abstention": False,
            "new_tokens": 120,
            "unknown_terms": ["Durbin-Watson duration"],
        },
        {
            "exam_shape": False,
            "exam_shape_markers": [],
            "abstention": True,
            "new_tokens": 400,
            "unknown_terms": ["Durbin-Watson duration", "Carino Smoothing Ratio"],
        },
    ]
    stats = assistant.summarize_open_ended(rows)
    assert stats["n"] == 2
    assert stats["exam_shape_rate"] == 0.5
    assert stats["abstention_rate"] == 0.5
    assert stats["mean_new_tokens"] == 260
    assert stats["unknown_term_rate"] == 1.0
    # Most frequent first, so a repeated invention outranks a one-off.
    assert list(stats["unknown_terms"])[0] == "Durbin-Watson duration"


def test_summarize_agentic_separates_multi_step_and_no_call():
    rows = [
        {"kind": "call", "correct": True, "arguments_valid": True, "n_expected": 1},
        {"kind": "call", "correct": False, "arguments_valid": True, "n_expected": 2},
        {"kind": "no_call", "correct": True, "n_expected": 0},
    ]
    stats = assistant.summarize_agentic(rows)
    assert stats["n"] == 3
    assert stats["accuracy"] == pytest.approx(2 / 3)
    assert stats["no_call_precision"] == 1.0
    assert stats["multi_step_accuracy"] == 0.0


def test_empty_suites_summarize_to_zero():
    assert assistant.summarize_open_ended([])["n"] == 0
    assert assistant.summarize_agentic([])["n"] == 0
