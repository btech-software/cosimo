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


# --------------------------------------------------------------------------
# fact grounding (spec §8.4)
# --------------------------------------------------------------------------

# The gr_001 contract: 9.4% return, 12.0% vol, 4.2% risk-free, 5.2 excess,
# Sharpe 0.43.
ALLOWED = [9.4, 12.0, 4.2, 5.2, 0.43, 0.433]


def test_numbers_the_prompt_supplied_are_not_inventions():
    text = "Excess return is 5.2 over 12.0 vol, so the Sharpe is 0.43."
    assert assistant.invented_numbers(text, ALLOWED) == []


def test_a_figure_no_input_explains_is_reported_verbatim():
    text = "The Sharpe is 0.43 against a peer median of 0.71."
    assert assistant.invented_numbers(text, ALLOWED) == ["0.71"]


def test_percent_and_ratio_renderings_of_the_same_value_both_pass():
    """0.43 written as 43% is the same number; 4.3 is not."""
    assert assistant.invented_numbers("a Sharpe of 43%", ALLOWED) == []
    assert assistant.invented_numbers("a Sharpe of 4.3", ALLOWED) == ["4.3"]


def test_the_tolerance_is_relative_not_absolute():
    """Packs mix 1e6 AUMs with 1e-4 spreads; a fixed epsilon serves neither.

    The same 0.5% band has to bind at both magnitudes: an absolute epsilon
    large enough to accept a rounded million would wave through any spread, and
    one tight enough for a spread would reject the pack's own AUM.
    """
    assert assistant.invented_numbers("9.43", [9.4]) == []  # 0.32%, inside
    assert assistant.invented_numbers("9.46", [9.4]) == ["9.46"]  # 0.64%, outside
    # Six orders of magnitude up, the same proportional band.
    assert assistant.invented_numbers("1234000", [1234567]) == []  # 0.05%
    assert assistant.invented_numbers("1200000", [1234567]) == ["1200000"]  # 2.8%


def test_thousands_separators_read_as_one_number():
    assert assistant.invented_numbers("1,430,000 of gross", [1430000]) == []


def test_the_whitelist_matches_the_written_token():
    text = "Half of the 4.0bp spread is 2.0bp."
    assert assistant.invented_numbers(text, [4.0]) == ["2.0"]
    assert assistant.invented_numbers(text, [4.0], whitelist=["2.0"]) == []


def test_each_offending_token_is_reported_once_in_order():
    text = "0.71 then 0.92 then 0.71 again"
    assert assistant.invented_numbers(text, ALLOWED) == ["0.71", "0.92"]


def test_no_allowed_set_means_the_metric_does_not_apply():
    """A suite row with no fact pack is unscored, not scored as all-invented."""
    assert assistant.invented_numbers("anything 1.23 goes", []) == []


def test_must_mention_splits_hit_from_missed():
    hit, missed = assistant.must_mention_hits(
        "The Sharpe uses excess return over volatility.",
        ["Sharpe", "excess return", "convexity"],
    )
    assert hit == ["Sharpe", "excess return"]
    assert missed == ["convexity"]


def test_must_mention_is_case_insensitive_and_matches_inflections():
    hit, missed = assistant.must_mention_hits(
        "the fade period is what drives it", ["fade", "FADE PERIOD"]
    )
    assert missed == []
    assert len(hit) == 2


# --------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------


def test_a_desk_reply_must_be_short_and_unstructured():
    assert assistant.register_match("Around 0.43. Thin sample, though.", "desk_chat")
    # The collapse this catches: a one-line desk question answered as a memo.
    memo_shaped = "## Summary\n\n" + "word " * 80
    assert not assistant.register_match(memo_shaped, "desk_chat")
    assert not assistant.register_match("word " * 300, "desk_chat")


def test_a_memo_must_be_long_and_sectioned():
    body = "## Thesis\n\n" + "word " * 300
    assert assistant.register_match(body, "ic_memo")
    # Long but shapeless, and short but sectioned, both fail.
    assert not assistant.register_match("word " * 300, "ic_memo")
    assert not assistant.register_match("## Thesis\n\nword word", "ic_memo")


def test_bullets_count_as_structure():
    assert assistant.register_match("- point one\n- point two\n" + "word " * 300, "ic_memo")


def test_an_unknown_register_is_unscored_rather_than_failed():
    """None keeps the row out of the denominator; False would read as a regression."""
    assert assistant.register_match("anything", "not_a_register") is None
    assert assistant.register_match("anything", "") is None
    assert assistant.register_match("anything", None) is None


# --------------------------------------------------------------------------
# aggregation of the three new metrics
# --------------------------------------------------------------------------


def test_unscored_rows_stay_out_of_each_denominator():
    rows = [
        # scored on all three
        {
            "invented_numbers": ["0.71"],
            "must_mention_hit": ["Sharpe"],
            "must_mention_missed": ["convexity"],
            "register_match": True,
        },
        {
            "invented_numbers": [],
            "must_mention_hit": ["Sharpe", "convexity"],
            "must_mention_missed": [],
            "register_match": False,
        },
        # an open_ended row: declares none of the three contracts
        {"invented_numbers": None, "must_mention_missed": None, "register_match": None},
    ]
    stats = assistant.summarize_open_ended(rows)
    assert stats["n"] == 3
    # Two scored rows, one of which invented a number.
    assert stats["invented_numbers_n"] == 2
    assert stats["invented_number_rate"] == pytest.approx(0.5)
    # A mean of per-answer coverage ratios: 1/2 and 2/2.
    assert stats["must_mention_n"] == 2
    assert stats["must_mention_hit_rate"] == pytest.approx(0.75)
    assert stats["register_match_n"] == 2
    assert stats["register_match_rate"] == pytest.approx(0.5)


def test_a_suite_declaring_no_contracts_reports_zero_denominators():
    rows = [{"exam_shape": False, "new_tokens": 10}]
    stats = assistant.summarize_open_ended(rows)
    for key in ("invented_numbers_n", "must_mention_n", "register_match_n"):
        assert stats[key] == 0
    for key in (
        "invented_number_rate",
        "must_mention_hit_rate",
        "register_match_rate",
    ):
        assert stats[key] == 0.0
