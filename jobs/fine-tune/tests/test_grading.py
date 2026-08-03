"""Grading: answer extraction, numeric parsing, MCQ, prose and distractors.

Formatting must never decide correctness, and a formatting habit must never move
a metric on its own. These tests pin both directions.
"""

from __future__ import annotations

import pytest

from cosimo_ft import grading

TAG = grading.DEFAULT_TAG


def gen(*lines: str) -> str:
    """A generation body, one line per argument."""
    return "\n".join(lines)


# --------------------------------------------------------------------------
# extract_final_answer
# --------------------------------------------------------------------------


def test_extract_takes_the_last_tag_occurrence():
    text = gen(
        "I will restate the protocol: finish with FINAL ANSWER: <value>",
        "Working: 3 * 4 = 12",
        "FINAL ANSWER: 12",
    )
    assert grading.extract_final_answer(text, TAG) == "12"


def test_extract_stops_at_the_end_of_the_line():
    text = gen("FINAL ANSWER: 42", "Let me double check that.")
    assert grading.extract_final_answer(text, TAG) == "42"


def test_extract_strips_markdown_decoration():
    assert (
        grading.extract_final_answer("FINAL ANSWER: **$1,234.00**", TAG) == "$1,234.00"
    )


def test_extract_is_case_insensitive():
    assert grading.extract_final_answer("Final Answer: 7", TAG) == "7"


def test_extract_returns_none_without_the_tag():
    assert grading.extract_final_answer("The answer is 12.", TAG) is None


def test_extract_returns_none_for_an_empty_value():
    assert grading.extract_final_answer("FINAL ANSWER:   ", TAG) is None


# --------------------------------------------------------------------------
# parse_number / numbers_in
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,805,579.56", 1805579.56),
        ("$174,425.08", 174425.08),
        ("12.5%", 12.5),
        ("-$1,234.00", -1234.0),
        ("(1,234)", -1234.0),
        ("+3.5", 3.5),
        ("1.2e-3", 0.0012),
        ("−0.75", -0.75),  # unicode minus
        ("€1 250", 1250.0),  # euro sign + non-breaking space
        ("42.", 42.0),
        (".5", 0.5),
    ],
)
def test_parse_number_tolerates_formatting(raw, expected):
    assert grading.parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", None, "n/a", "Increase", "$"])
def test_parse_number_rejects_non_numbers(raw):
    assert grading.parse_number(raw) is None


def test_hyphen_between_numbers_is_a_range_not_a_minus_sign():
    assert grading.numbers_in("Between 5-10 bp") == [5.0, 10.0]


def test_iso_date_is_not_read_as_negative_numbers():
    assert grading.numbers_in("2024-03-05") == [2024.0, 3.0, 5.0]


def test_leading_minus_is_still_a_minus_sign():
    assert grading.numbers_in("a drop of -12.5 bp") == [-12.5]


def test_numeric_close_has_an_absolute_floor_near_zero():
    assert grading.numeric_close(0.0, 1e-9)
    assert not grading.numeric_close(0.0, 1e-3)


def test_numeric_close_respects_relative_tolerance():
    assert grading.numeric_close(1805579.56, 1805580.0, rel_tol=1e-3)
    assert not grading.numeric_close(1805579.56, 1815579.56, rel_tol=1e-3)


# --------------------------------------------------------------------------
# grade_cosimo — numeric
# --------------------------------------------------------------------------


def numeric_record(answer: str = "1,805,579.56", **extra) -> dict:
    record = {
        "id": "rec-1",
        "question_type": "Calculation",
        "answer": answer,
        "distractors": [],
    }
    record.update(extra)
    return record


def test_numeric_formatting_never_decides_correctness():
    grade = grading.grade_cosimo(numeric_record("$1,234.00"), "FINAL ANSWER: 1234", TAG)
    assert grade.correct and grade.format_ok and grade.mode == "numeric"


def test_percent_sign_is_not_a_difference():
    assert grading.grade_cosimo(
        numeric_record("12.5%"), "FINAL ANSWER: 12.5", TAG
    ).correct
    assert grading.grade_cosimo(
        numeric_record("12.5"), "FINAL ANSWER: 12.5%", TAG
    ).correct


def test_numeric_outside_tolerance_is_wrong():
    grade = grading.grade_cosimo(numeric_record("100.0"), "FINAL ANSWER: 101.0", TAG)
    assert not grade.correct


def test_missing_tag_still_grades_the_last_line():
    grade = grading.grade_cosimo(
        numeric_record("12.5%"),
        gen("Step 1: compute the yield.", "The answer is 12.5%"),
        TAG,
    )
    assert grade.correct, "a right answer without the tag must not be scored wrong"
    assert not grade.format_ok, "format compliance must record the missing tag"


def test_missing_tag_and_missing_answer():
    grade = grading.grade_cosimo(numeric_record("12.5"), "", TAG)
    assert not grade.correct and not grade.format_ok and grade.pred is None


def test_trailing_prose_on_the_final_line_is_read_right_to_left():
    grade = grading.grade_cosimo(
        numeric_record("3"),
        "FINAL ANSWER: after selling 18 eggs she has 3",
        TAG,
    )
    assert grade.correct


# --------------------------------------------------------------------------
# grade_cosimo — MCQ
# --------------------------------------------------------------------------


def mcq_record(answer: str = "C. 43", distractors=("A. 12", "B. 20", "D. 51")) -> dict:
    return {
        "id": "mcq-1",
        "question_type": "MCQ",
        "answer": answer,
        "distractors": list(distractors),
    }


def test_mcq_letter_path():
    grade = grading.grade_cosimo(mcq_record(), "FINAL ANSWER: C. 43", TAG)
    assert grade.correct and grade.mode == "mcq"


def test_mcq_bare_letter_is_accepted():
    assert grading.grade_cosimo(mcq_record(), "FINAL ANSWER: C", TAG).correct


def test_mcq_letter_in_prose_is_accepted():
    assert grading.grade_cosimo(
        mcq_record(), "FINAL ANSWER: the correct option is C", TAG
    ).correct


def test_mcq_value_without_a_letter_is_accepted():
    assert grading.grade_cosimo(mcq_record(), "FINAL ANSWER: 43", TAG).correct


def test_mcq_duplicate_option_values_grade_on_the_value():
    # Both A and B carry the value 20 in the corpus; letter matching alone would
    # score an equally correct answer as wrong.
    record = mcq_record(answer="B. 20", distractors=["A. 20", "C. 35", "D. 51"])
    assert grading.grade_cosimo(record, "FINAL ANSWER: A. 20", TAG).correct


def test_mcq_wrong_letter_with_wrong_value_is_wrong():
    grade = grading.grade_cosimo(mcq_record(), "FINAL ANSWER: D. 51", TAG)
    assert not grade.correct


def test_mcq_lower_case_option_letter():
    assert grading.grade_cosimo(mcq_record(), "FINAL ANSWER: c) 43", TAG).correct


def test_english_article_a_is_not_read_as_option_a():
    record = mcq_record(answer="A. 12")
    grade = grading.grade_cosimo(record, "FINAL ANSWER: a portfolio worth 51", TAG)
    assert not grade.correct


# --------------------------------------------------------------------------
# distractor matching ("fell for the pitfall")
# --------------------------------------------------------------------------


def test_distractor_matching_on_value():
    grade = grading.grade_cosimo(
        numeric_record("100.0", distractors=["90.0", "110.0"]),
        "FINAL ANSWER: 110.0",
        TAG,
    )
    assert not grade.correct and grade.matched_distractor


def test_bare_letter_answer_counts_as_a_distractor_match():
    # "D" and "D. 51" are the same mistake; distractor_rate must not improve for
    # free just because the model changed how it phrases the final line.
    assert grading.grade_cosimo(mcq_record(), "FINAL ANSWER: D", TAG).matched_distractor
    assert grading.grade_cosimo(
        mcq_record(), "FINAL ANSWER: D. 51", TAG
    ).matched_distractor


def test_a_correct_answer_is_never_a_distractor_match():
    grade = grading.grade_cosimo(
        numeric_record("100.0", distractors=["100.0", "110.0"]),
        "FINAL ANSWER: 100.0",
        TAG,
    )
    assert grade.correct and not grade.matched_distractor


def test_an_unrelated_wrong_answer_is_not_a_distractor_match():
    grade = grading.grade_cosimo(
        numeric_record("100.0", distractors=["90.0", "110.0"]),
        "FINAL ANSWER: 7.25",
        TAG,
    )
    assert not grade.correct and not grade.matched_distractor


# --------------------------------------------------------------------------
# grade_cosimo — prose gold
# --------------------------------------------------------------------------


def prose_record(answer: str) -> dict:
    return {
        "id": "cr-1",
        "question_type": "Constructed Response",
        "answer": answer,
        "distractors": [],
    }


def test_prose_gold_requires_every_gold_number():
    record = prose_record("No; t-statistic 1.99 < 2.132")
    assert grading.grade_cosimo(
        record, "FINAL ANSWER: No; t-statistic 1.99 < 2.132", TAG
    ).correct
    assert not grading.grade_cosimo(
        record, "FINAL ANSWER: No; t-statistic 1.99", TAG
    ).correct


def test_prose_gold_yes_no_token_must_agree():
    record = prose_record("No; t-statistic 1.99 < 2.132")
    grade = grading.grade_cosimo(
        record, "FINAL ANSWER: Yes; t-statistic 1.99 < 2.132", TAG
    )
    assert not grade.correct and grade.mode == "prose"


def test_prose_gold_phrase_match():
    record = prose_record("Increase")
    assert grading.grade_cosimo(
        record, "FINAL ANSWER: The duration gap will increase", TAG
    ).correct


def test_negation_flips_a_prose_phrase_match():
    record = prose_record("Increase")
    grade = grading.grade_cosimo(
        record, "FINAL ANSWER: The duration gap will not increase", TAG
    )
    assert not grade.correct, "'will not increase' does not answer 'Increase'"


def test_prose_mode_is_reported_for_prose_gold():
    grade = grading.grade_cosimo(
        prose_record("Increase"), "FINAL ANSWER: increase", TAG
    )
    assert grade.mode == "prose" and grade.correct


# --------------------------------------------------------------------------
# grade_math
# --------------------------------------------------------------------------


def test_grade_math_numeric_gold():
    grade = grading.grade_math("18", "FINAL ANSWER: 18", TAG)
    assert grade.correct and grade.mode == "numeric" and grade.format_ok


def test_grade_math_accepts_a_restated_final_line():
    grade = grading.grade_math(
        "3", "FINAL ANSWER: she bakes 18 muffins and 3 are left", TAG
    )
    assert grade.correct


def test_grade_math_comma_formatting():
    assert grading.grade_math("72000", "FINAL ANSWER: $72,000", TAG).correct


def test_grade_math_string_gold_is_latex_normalised():
    assert grading.grade_math(
        "\\boxed{\\frac{3}{4}}", "FINAL ANSWER: \\frac{3}{4}", TAG
    ).correct
    grade = grading.grade_math("\\frac{3}{4}", "FINAL ANSWER: \\frac{1}{4}", TAG)
    assert not grade.correct and grade.mode == "string"


def test_grade_math_missing_tag_sets_format_flag_only():
    grade = grading.grade_math("18", "so the total is 18", TAG)
    assert grade.correct and not grade.format_ok


def test_grade_math_never_reports_a_distractor():
    assert not grading.grade_math("18", "FINAL ANSWER: 19", TAG).matched_distractor
