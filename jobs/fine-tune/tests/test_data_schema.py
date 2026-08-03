"""Row normalisation and the pinned JSONL field contract.

The dataset ships ``preference_pair`` as a struct on ~35% of rows and as nothing
at all on the rest, and the repo's own FORMAT.md documents a stale plain-string
shape. All three have to survive normalisation.
"""

from __future__ import annotations

import pytest

from cosimo_ft import chat, data_schema

TAG = "FINAL ANSWER:"
SYSTEM = "You are Cosimo."


def default_row(**overrides) -> dict:
    """A row of the dataset's ``default`` config."""
    row = {
        "id": "cfa_l1-000001",
        "program": "CFA_Level_I",
        "topic": "Fixed Income",
        "subtopic": "Duration",
        "difficulty": "medium",
        "question_type": "Calculation",
        "question": "What is the modified duration?",
        "answer": "7.42",
        "distractors": ["7.10", "8.00"],
        "reasoning_trace": "Macaulay duration / (1 + y/m).",
        "verified": True,
        "verification": {"template": "fi_modified_duration"},
        "metadata": {
            "generator": "fi_modified_duration",
            "pitfalls_addressed": ["forgot to divide by (1+y)"],
        },
        "preference_pair": None,
    }
    row.update(overrides)
    return row


PREFERENCE_STRUCT = {
    "chosen": {"answer": "7.42", "reasoning_trace": "Divide by (1 + y/m)."},
    "rejected": {"answer": "7.80", "reasoning_trace": "Forgot to divide by (1 + y/m)."},
    "pitfall": "forgot to divide by (1+y)",
}


# --------------------------------------------------------------------------
# stem_family
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("generator", "family"),
    [
        ("v_tvm_annuity_fv", "tvm_annuity_fv"),
        ("cr_eq_gordon", "eq_gordon"),
        ("m_corp_wacc", "corp_wacc"),
        ("m_credit_var", "credit_var"),
        ("tvm_annuity_fv", "tvm_annuity_fv"),
        ("mkt_cvar", "mkt_cvar"),
        ("credit_el", "credit_el"),
        ("", "unknown"),
    ],
)
def test_stem_family_strips_only_wrapper_prefixes(generator, family):
    assert data_schema.stem_family(generator) == family


def test_wrapper_and_base_stem_share_a_family():
    # This is what makes holding out by family (not by generator) meaningful.
    assert data_schema.stem_family("v_tvm_annuity_fv") == data_schema.stem_family(
        "tvm_annuity_fv"
    )


# --------------------------------------------------------------------------
# normalize_record
# --------------------------------------------------------------------------


def test_struct_preference_pair():
    rec = data_schema.normalize_record(default_row(preference_pair=PREFERENCE_STRUCT))
    assert data_schema.has_preference(rec)
    assert rec.chosen["answer"] == "7.42"
    assert rec.rejected["reasoning_trace"].startswith("Forgot")
    assert rec.pitfall == "forgot to divide by (1+y)"


def test_string_preference_pair_is_read_as_a_reasoning_trace():
    # The stale shape documented in dataset/FORMAT.md.
    rec = data_schema.normalize_record(
        default_row(
            preference_pair={"chosen": "Correct trace.", "rejected": "Wrong trace."}
        )
    )
    assert data_schema.has_preference(rec)
    assert rec.chosen == {"answer": "7.42", "reasoning_trace": "Correct trace."}
    assert rec.rejected == {"answer": "7.42", "reasoning_trace": "Wrong trace."}


def test_null_preference_pair():
    rec = data_schema.normalize_record(default_row(preference_pair=None))
    assert rec.chosen is None and rec.rejected is None
    assert not data_schema.has_preference(rec)


def test_identical_sides_are_not_a_usable_preference():
    rec = data_schema.normalize_record(
        default_row(preference_pair={"chosen": "Same.", "rejected": "Same."})
    )
    assert not data_schema.has_preference(rec)


def test_missing_generator_falls_back_to_the_verification_template():
    row = default_row(metadata={"pitfalls_addressed": []})
    assert data_schema.normalize_record(row).generator == "fi_modified_duration"


def test_missing_generator_and_template_is_unknown():
    row = default_row(metadata={}, verification={})
    rec = data_schema.normalize_record(row)
    assert rec.generator == "unknown"
    assert data_schema.to_eval_row(rec)["stem_family"] == "unknown"


def test_null_distractors_become_an_empty_tuple():
    rec = data_schema.normalize_record(default_row(distractors=None))
    assert rec.distractors == ()
    assert data_schema.to_eval_row(rec)["distractors"] == []


def test_mcq_record():
    rec = data_schema.normalize_record(
        default_row(
            question_type="MCQ",
            question="Value?\nA. 20\nB. 20\nC. 43\nD. 51",
            answer="C. 43",
            distractors=["A. 20", "B. 20", "D. 51"],
        )
    )
    assert rec.question_type == "MCQ"
    assert rec.answer == "C. 43"
    assert len(rec.distractors) == 3


def test_constructed_response_record_has_no_distractors():
    rec = data_schema.normalize_record(
        default_row(question_type="Constructed Response", distractors=[])
    )
    assert rec.question_type == "Constructed Response"
    assert rec.distractors == ()


def test_metadata_fills_in_missing_top_level_fields():
    row = default_row(topic=None, subtopic=None, difficulty=None, question_type=None)
    row["metadata"].update(
        {
            "topic": "Fixed Income",
            "subtopic": "Duration",
            "difficulty": "hard",
            "question_type": "Vignette",
        }
    )
    rec = data_schema.normalize_record(row)
    assert (rec.topic, rec.subtopic, rec.difficulty, rec.question_type) == (
        "Fixed Income",
        "Duration",
        "hard",
        "Vignette",
    )


# --------------------------------------------------------------------------
# normalize_pref_row (the `preference_pairs` config)
# --------------------------------------------------------------------------


def test_normalize_pref_row_uses_prompt_as_the_question():
    row = {
        "id": "cfa_l1-000001",
        "program": "CFA_Level_I",
        "topic": "Fixed Income",
        "subtopic": "Duration",
        "difficulty": "medium",
        "question_type": "Calculation",
        "prompt": "What is the modified duration?",
        "answer": "7.42",
        "generator": "fi_modified_duration",
        **PREFERENCE_STRUCT,
    }
    rec = data_schema.normalize_pref_row(row)
    assert rec.question == "What is the modified duration?"
    assert rec.id == "cfa_l1-000001"
    assert rec.generator == "fi_modified_duration"
    assert data_schema.has_preference(rec)


def test_normalize_pref_row_without_a_joined_generator_is_unknown():
    rec = data_schema.normalize_pref_row(
        {"id": "x-1", "prompt": "q", "answer": "1", **PREFERENCE_STRUCT}
    )
    assert rec.generator == "unknown"


# --------------------------------------------------------------------------
# row projections — the pinned JSONL field contract
# --------------------------------------------------------------------------


def test_eval_row_fields_are_exactly_the_contract():
    rec = data_schema.normalize_record(default_row())
    assert tuple(data_schema.to_eval_row(rec)) == data_schema.EVAL_FIELDS


def test_sft_row_is_the_eval_row_plus_the_rendered_text(fake_tokenizer):
    rec = data_schema.normalize_record(default_row())
    row = data_schema.to_sft_row(rec, fake_tokenizer, SYSTEM, TAG)
    assert set(row) == set(data_schema.EVAL_FIELDS) | {"prompt", "completion", "text"}
    assert row["text"] == row["prompt"] + row["completion"]
    assert row["completion"].endswith(
        chat.build_completion(rec.reasoning_trace, rec.answer, TAG)
        + "<|end|>"
        + fake_tokenizer.eos_token
    )
    assert SYSTEM in row["prompt"]


def test_pref_row_renders_both_sides_through_the_sft_path(fake_tokenizer):
    rec = data_schema.normalize_record(default_row(preference_pair=PREFERENCE_STRUCT))
    row = data_schema.to_pref_row(rec, fake_tokenizer, SYSTEM, TAG)
    assert set(row) == set(data_schema.EVAL_FIELDS) | {
        "pitfall",
        "prompt",
        "chosen",
        "rejected",
    }
    assert row["chosen"] != row["rejected"]
    assert TAG in row["chosen"] and TAG in row["rejected"]
    assert row["prompt"].endswith("<|assistant|>")
    # DPO must see exactly the SFT output distribution: the completion strings
    # are what SFT would have trained on for the same trace.
    sft = data_schema.to_sft_row(rec, fake_tokenizer, SYSTEM, TAG)
    assert row["prompt"] == sft["prompt"]


def test_pref_row_without_a_pair_is_refused(fake_tokenizer):
    rec = data_schema.normalize_record(default_row())
    with pytest.raises(ValueError):
        data_schema.to_pref_row(rec, fake_tokenizer, SYSTEM, TAG)
