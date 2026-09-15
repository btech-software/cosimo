"""Behavioural metrics: exam-shape leakage, abstention, terminology, trajectories."""

from __future__ import annotations

import json
import pathlib

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
    text = (
        "The Sharpe ratio is 0.57. "
        + "x" * 800
        + " Of course, I'd need to know your horizon."
    )
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


def test_a_desk_reply_may_not_wear_memo_scaffolding():
    assert assistant.register_match("Around 0.43. Thin sample, though.", "desk_chat")
    # The collapse this catches: a one-line desk question answered as a memo.
    assert not assistant.register_match("Finding: it is 0.43.", "desk_chat")
    assert not assistant.register_match("Call: hold the position.", "desk_chat")


def test_a_desk_reply_has_a_sentence_ceiling_it_cannot_dodge():
    # Thirteen short sentences: under every word count, over the ceiling.
    assert not assistant.register_match("It moved. " * 13, "desk_chat")
    # And the ceiling cannot be met by writing fewer, longer sentences --
    # a live sample once scored clean at 48 words a sentence, which is a memo
    # wearing a desk reply's sentence count.
    assert not assistant.register_match(("word " * 60 + ". ") * 3, "desk_chat")


def test_an_ic_memo_must_reach_a_decision_however_short():
    """The inversion of the rule this replaced.

    The old reading demanded 250 words and a heading, which no row the corpus
    writes can reach: its own budgets top out at 220 words for an analysis row,
    and its gate calls headings permitted rather than required. So 0 of 36
    shipped ic_memo rows passed, and a base model rambling under markdown
    outscored a tuned one writing to contract. What a memo owes is a call.
    """
    assert assistant.register_match("Call: trim into the close.", "ic_memo")
    assert assistant.register_match("We would reduce the position.", "ic_memo")
    # Long and sectioned but deciding nothing is the failure, not the pass.
    assert not assistant.register_match("## Thesis\n\n" + "word " * 300, "ic_memo")


def test_the_call_exemption_follows_the_record_type():
    """A citation and a refusal are memos that legitimately decide nothing."""
    surveyed = "The peer median is 17.31 and the dispersion is wide."
    assert not assistant.register_match(surveyed, "ic_memo")
    for kind in sorted(assistant.CALL_EXEMPT_KINDS):
        assert assistant.register_match(surveyed, "ic_memo", kind)
    # An unlabelled row is held to the full contract rather than excused.
    assert not assistant.register_match(surveyed, "ic_memo", "")


def test_a_kind_stricter_than_its_register_sets_the_ceiling():
    """A citation is eight sentences and a refusal five, wherever written.

    The register's own ceiling is twelve, and reading only that made this
    module quietly more permissive than the gate for exactly the two kinds
    with the tightest real limits -- which are also two of the five the
    `--eval-shards` path exists to score honestly.
    """
    assert assistant.desk_chat_ceiling("grounded") == 8
    assert assistant.desk_chat_ceiling("abstention") == 5
    nine = "It moved. " * 9
    assert assistant.register_match(nine, "desk_chat", "analysis")
    assert not assistant.register_match(nine, "desk_chat", "grounded")
    assert not assistant.register_match("It moved. " * 6, "desk_chat", "abstention")


def test_the_ceiling_widens_where_twelve_is_unreachable():
    """A register belongs to the family and a word budget to the record type.

    A work type whose families all speak desk chat still has to write a memo
    at its own floor in the desk's voice, and a ceiling that refused every
    such memo forever would be a gate nobody could satisfy.
    """
    assert assistant.desk_chat_ceiling("memo") >= assistant.DESK_CHAT_MAX_SENTENCES
    assert assistant.desk_chat_ceiling("analysis") == assistant.DESK_CHAT_MAX_SENTENCES


def test_a_grounded_row_carries_no_scaffolding_in_any_register():
    """The rule belongs to the kind, not to the voice.

    `grounded` is call-exempt, so an ic_memo grounded row reaches no
    register rule at all -- and the corpus records the consequence: its first
    live grounded row on an ic_memo family closed on "Call: act on Financials
    allocation", obeying its register and breaking its kind.
    """
    headed = "Finding: FCFF is 465.\nEvidence: EBIT 620.\nCall: hold."
    for register in ("desk_chat", "ic_memo", "risk_committee"):
        assert not assistant.register_match(headed, register, "grounded")
    # A labelled call with no headings is its own violation.
    assert not assistant.register_match("Call: hold.", "ic_memo", "grounded")
    # And the same text is fine for a kind the rule does not govern.
    assert assistant.register_match(headed, "ic_memo", "memo")


def test_a_risk_paper_names_a_constraint_and_takes_no_position():
    assert assistant.register_match("The 1-day limit binds at 39.2.", "risk_committee")
    # Says nothing about a limit, horizon or assumption.
    assert not assistant.register_match("Vol was 2.33% yesterday.", "risk_committee")
    # The committee constrains; the desk takes the position.
    assert not assistant.register_match(
        "The limit binds, and we would buy the dip.", "risk_committee"
    )


def test_a_declared_but_unwritten_register_scores_clean_not_unscored():
    """``auditor``/``code_review`` are in the taxonomy and no family emits them.

    The corpus gate returns clean for both rather than inventing shape rules
    for prose nobody has read, and this mirrors it -- scoring them ``None``
    would quietly drop `critique.jsonl`'s auditor rows out of the denominator.
    """
    assert assistant.register_match("anything at all", "auditor") is True
    assert assistant.register_match("anything at all", "code_review") is True


#: The corpus's human-certified rows, and the one artifact that can prove this
#: module still agrees with the gate that shaped them. Committed (unlike
#: ``dataset/shards/``, which is generated and ignored), so the check is
#: hermetic; read as *data*, never imported, which is the boundary ``jobs``
#: keeps against ``dataset``.
GOLD_BAR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "dataset"
    / "goldbar"
    / "gold_bar_v3.jsonl"
)


@pytest.mark.skipif(not GOLD_BAR.exists(), reason="dataset/goldbar not checked out")
def test_the_eval_agrees_with_the_corpus_gate_on_certified_rows():
    """The drift this module exists to make impossible to ship unnoticed.

    Every gold-bar row carries the generation-time gate's own verdict,
    ``verification.register_ok``, beside the answer it judged. Scoring the
    answer here and comparing is the whole contract: two independent readings
    of one rule, checked against rows a human signed off.

    It is not hypothetical. Against the word-count reading this replaced, three
    of these seven rows -- every ic_memo, at 84, 157 and 178 words -- were
    scored as failures while the gate had passed them, because the eval's floor
    sat above the corpus's ceiling. That is the shape of the bug: not a metric
    that is merely wrong, but one that ranks a rambling base model above a
    model writing to the corpus's own contract.
    """
    rows = [
        json.loads(line)
        for line in GOLD_BAR.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    checked = 0
    registers = set()
    for row in rows:
        expected = (row.get("verification") or {}).get("register_ok")
        if row.get("register") is None or expected is None:
            continue
        actual = assistant.register_match(
            row["answer"], row["register"], row.get("record_type") or ""
        )
        assert actual == expected, (
            f"{row['id']} ({row['register']}/{row.get('record_type')}): the "
            f"corpus gate says register_ok={expected}, this module says {actual}"
        )
        registers.add(row["register"])
        checked += 1
    assert checked >= 5, f"only {checked} certified rows carried a verdict to check"
    # A bar that lost a voice would still pass row-by-row while testing less.
    assert registers == {"desk_chat", "ic_memo", "risk_committee"}, registers


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
