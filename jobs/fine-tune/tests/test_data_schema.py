"""Row normalisation and the pinned JSONL field contract.

Two published corpora go through this module and they do not share a shape:

* v1 ships ``preference_pair`` as a struct on ~35% of rows and as nothing at all
  on the rest (and the repo's own FORMAT.md documents a stale plain-string
  shape); its ``metadata``/``verification`` are Arrow structs.
* v2 ships five record types, JSON-**encoded** nested columns, and standalone
  preference rows whose ids no supervised row shares.

All of it has to survive normalisation, and the ``FINAL ANSWER:`` contract has
to end up on exam rows and nowhere else.
"""

from __future__ import annotations

import json

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


# --------------------------------------------------------------------------
# v2: JSON-encoded nested columns
# --------------------------------------------------------------------------


def v2_row(**overrides) -> dict:
    """A row of v2's ``default`` config: nested columns are JSON strings."""
    row = {
        "id": "cosimo_CFA_Level_II_200966_3c152700b3126de4",
        "record_type": "analysis",
        "program": "CFA_Level_II",
        "topic": "Equity Valuation",
        "subtopic": "FCFF DCF",
        "difficulty": "CFA L2",
        "question_type": "Analysis",
        "question": "Walk through the modelling decisions for a mature firm.",
        "answer": "NOPAT = 208.53M. Capex = 122.69M. Terminal value dominates.",
        "distractors": [],
        "reasoning_trace": "",
        "code": "",
        "test_code": "",
        "conversation": "",
        "tool_schemas": "",
        "verified": True,
        "verification": "",
        "metadata": json.dumps(
            {"generator": "eq_fcff_dcf", "record_type": "analysis", "seed": 1182126958}
        ),
    }
    row.update(overrides)
    return row


def test_json_encoded_metadata_still_yields_the_generator():
    # The failure this guards is silent and total: an `isinstance(value, dict)`
    # guard reads a JSON string as {}, every generator becomes "unknown", the
    # split stratification collapses to one stratum and every configured
    # holdout family matches nothing.
    rec = data_schema.normalize_record(v2_row())
    assert rec.generator == "eq_fcff_dcf"
    assert data_schema.to_eval_row(rec)["stem_family"] == "eq_fcff_dcf"


def test_malformed_json_metadata_does_not_raise():
    rec = data_schema.normalize_record(v2_row(metadata="{not json", verification=""))
    assert rec.generator == "unknown"


def test_a_v1_row_without_record_type_is_an_exam_record():
    rec = data_schema.normalize_record(default_row())
    assert rec.record_type == data_schema.EXAM
    assert data_schema.is_exam(rec)


# --------------------------------------------------------------------------
# v2: the grading contract belongs to exam rows only
# --------------------------------------------------------------------------


def test_exam_target_ends_with_the_final_answer_line():
    rec = data_schema.normalize_record(default_row())
    completion = data_schema.build_supervised_completion(rec, TAG)
    assert completion.endswith(f"{TAG} 7.42")


@pytest.mark.parametrize("record_type", ["analysis", "abstention"])
def test_prose_targets_are_the_answer_verbatim_with_no_tag(record_type):
    # Appending the grading contract to a 900-token analysis is precisely the
    # uniformity that flattened the first run into a calculator.
    rec = data_schema.normalize_record(v2_row(record_type=record_type))
    completion = data_schema.build_supervised_completion(rec, TAG)
    assert completion == rec.answer
    assert TAG not in completion


def test_implementation_target_is_composed_from_code_and_tests():
    # `answer` alone is the bare result -- ~20 characters -- and the substance
    # is in `code`/`test_code`, so training on `answer` teaches nothing.
    rec = data_schema.normalize_record(
        v2_row(
            record_type="implementation",
            question_type="Implementation",
            answer="Value=$1,625,956,825",
            code="def dcf_value(fcff, wacc, g, nper):\n    return 1.0",
            test_code="assert dcf_value(100e6, 0.10, 0.02, 5) > 0",
        )
    )
    completion = data_schema.build_supervised_completion(rec, TAG)
    assert "```python\ndef dcf_value" in completion
    assert "assert dcf_value(100e6" in completion
    assert completion.endswith("Value=$1,625,956,825")
    assert TAG not in completion


def test_implementation_without_tests_still_renders_the_code():
    rec = data_schema.normalize_record(
        v2_row(record_type="implementation", answer="V=1", code="x = 1", test_code="")
    )
    assert data_schema.build_supervised_completion(rec, TAG) == "```python\nx = 1\n```\n\nV=1"


def test_unparseable_test_block_is_left_out_of_the_target():
    # 7,500 of v2's 13,000 implementation records ship this exact stray-indent
    # SyntaxError. Training a model that is meant to write idiomatic Python on
    # Python that does not parse is worse than training it on less Python.
    broken = "forwards = bootstrapped_yield([0.02])\n    assert len(forwards) == 4"
    assert not data_schema.is_valid_python(broken)
    rec = data_schema.normalize_record(
        v2_row(
            record_type="implementation", answer="V=1", code="x = 1", test_code=broken
        )
    )
    completion = data_schema.build_supervised_completion(rec, TAG)
    assert "assert len(forwards)" not in completion
    assert "```python\nx = 1\n```" in completion


# --------------------------------------------------------------------------
# v2: agentic records render as the whole tool conversation
# --------------------------------------------------------------------------

AGENTIC_CONVERSATION = [
    {"role": "user", "content": "Stock at $114, EPS $5.17. Three target prices?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_fundamentals",
                    "arguments": {"symbol": "AAPL", "metrics": ["target_price"]},
                },
            }
        ],
    },
    {
        "role": "tool",
        "name": "get_fundamentals",
        "content": '{"target_prices": [119.7, 124.26, 128.82]}',
    },
    {"role": "assistant", "content": "Avg target $124.26 vs $114.00 = 9.0% upside."},
]

AGENTIC_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get fundamental metrics",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def agentic_row(**overrides) -> dict:
    fields = {
        "record_type": "agentic",
        "question_type": "Agentic",
        "answer": "Avg target $124.26 9.0% upside",
        "conversation": json.dumps(AGENTIC_CONVERSATION),
        "tool_schemas": json.dumps(AGENTIC_SCHEMAS),
    }
    fields.update(overrides)
    return v2_row(**fields)


def test_agentic_conversation_and_schemas_survive_normalisation():
    rec = data_schema.normalize_record(agentic_row())
    assert len(rec.conversation) == 4
    assert rec.conversation[1]["tool_calls"][0]["function"]["name"] == (
        "get_fundamentals"
    )
    assert rec.tool_schemas == tuple(AGENTIC_SCHEMAS)


def test_agentic_sft_row_supervises_from_the_first_assistant_turn(jinja_tokenizer):
    rec = data_schema.normalize_record(agentic_row())
    row = data_schema.to_sft_row(rec, jinja_tokenizer, SYSTEM, TAG)

    assert row["text"] == row["prompt"] + row["completion"]
    # The schemas reach the system turn as the template's top-level `tools`
    # variable -- the vendor template read a per-message key nothing sets, and
    # silently dropped them.
    assert "<|tool|>" in row["prompt"]
    assert "get_fundamentals" in row["prompt"]
    # Everything from the first assistant turn on is supervised, including the
    # tool call and the final answer.
    assert "<tool_call>" in row["completion"]
    assert "9.0% upside" in row["completion"]
    assert "Three target prices?" not in row["completion"]
    # The interior tool result renders as a <|user|> turn, which is what
    # train_on_responses_only masks on -- without it the model would be trained
    # to generate its own tool results.
    assert "<|user|><tool_response>" in row["completion"]
    assert TAG not in row["completion"]


def test_agentic_record_without_an_assistant_turn_is_refused(jinja_tokenizer):
    rec = data_schema.normalize_record(
        agentic_row(conversation=json.dumps([{"role": "user", "content": "hi"}]))
    )
    with pytest.raises(ValueError):
        data_schema.to_sft_row(rec, jinja_tokenizer, SYSTEM, TAG)


# --------------------------------------------------------------------------
# v2: standalone preference rows
# --------------------------------------------------------------------------


def standalone_pref_row(**overrides) -> dict:
    row = {
        "id": "cosimopref_CFA_Level_I_300346_ff79f10bfc249123",
        "record_type": "preference",
        "program": "CFA_Level_I",
        "topic": "Portfolio Management",
        "subtopic": "Sharpe Ratio",
        "difficulty": "L1_Medium",
        "question_type": "Preference",
        "mode": "false_confidence",
        "prompt": "What's the Sharpe ratio on a book returning 14.40% with 22.49% vol?",
        "chosen": "I need the risk-free rate before I can answer that.",
        "rejected": "The Sharpe ratio is 0.51.",
        "pitfall": "answered an underspecified question by inventing the rate",
        "contains_intentional_fabrication": False,
        "verified": True,
        "verification": json.dumps({"template": "pref_sharpe_no_riskfree"}),
        "metadata": json.dumps(
            {"generator": "pref_sharpe_no_riskfree", "mode": "false_confidence"}
        ),
    }
    row.update(overrides)
    return row


def test_standalone_pref_row_carries_its_own_generator_and_mode():
    # These rows have no sibling in the `default` config to join against: the
    # `cosimopref_` id space is disjoint from every supervised id by
    # construction, which is what makes the DPO no-op impossible.
    rec = data_schema.normalize_standalone_pref_row(standalone_pref_row())
    assert rec.generator == "pref_sharpe_no_riskfree"
    assert rec.pref_mode == "false_confidence"
    assert rec.question.startswith("What's the Sharpe ratio")
    assert not data_schema.is_exam(rec)
    assert data_schema.has_preference(rec)


def test_standalone_pref_sides_are_rendered_verbatim(fake_tokenizer):
    rec = data_schema.normalize_standalone_pref_row(standalone_pref_row())
    row = data_schema.to_pref_row(rec, fake_tokenizer, SYSTEM, TAG)
    # No gold value exists and neither side is an exam answer, so bolting a
    # `FINAL ANSWER:` line onto them would train the exam shape onto the rows
    # whose whole purpose is teaching its absence.
    assert TAG not in row["chosen"] and TAG not in row["rejected"]
    assert row["chosen"].startswith("I need the risk-free rate")
    assert row["rejected"].startswith("The Sharpe ratio is 0.51.")
    assert row["chosen"] != row["rejected"]
    assert row["prompt"].endswith("<|assistant|>")


def test_standalone_pref_row_with_one_empty_side_is_not_usable():
    rec = data_schema.normalize_standalone_pref_row(standalone_pref_row(rejected=""))
    assert not data_schema.has_preference(rec)
