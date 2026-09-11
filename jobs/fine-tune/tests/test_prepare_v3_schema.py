"""The v3 corpus shape, as ``01_prepare_data.py`` has to read it.

v3 shares no taxonomy column with v1/v2 -- no ``program``, ``topic``,
``subtopic``, ``difficulty``, ``question_type``, ``metadata`` or ``verified`` --
and renames or restructures most of the rest. Every assertion here is one the
harness would otherwise discover as a silently empty corpus: a holdout that
matched nothing, a stratification collapsed to one bucket, an exam row with no
gradeable value, or an implementation row whose supervised target is the code
twice over.

``fixtures/v3_exam_row.json`` is a real row, produced by the real renderer::

    COSIMO_V3_OUT=... python -m dataset.pipelines.v3.cli inventory
    COSIMO_V3_OUT=... python -m dataset.pipelines.v3.cli packs
    COSIMO_V3_OUT=... python -m dataset.pipelines.v3.cli render --types exam

Exam is the one record type that renders with no teacher (it is composed from
the fact pack, and ``verification.teacher`` is null by construction), which is
what makes a genuine end-to-end sample affordable in an offline suite. The
other kinds are built here from the field lists ``verify_v3.py`` enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosimo_ft import chat, data_schema


def _prepare_module():
    """``01_prepare_data.py`` loaded by path -- its name is not an identifier."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_data",
        Path(__file__).resolve().parents[1] / "scripts" / "01_prepare_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _prepare_module()

FIXTURES = Path(__file__).resolve().parent / "fixtures"

SYSTEM = "You are Cosimo."
TAG = "FINAL ANSWER:"

WORK_TYPE = "risk.market.var_es"
FAMILY = "rates_book"
SCENARIO = f"{WORK_TYPE}.{FAMILY}"


@pytest.fixture(scope="module")
def exam_row() -> dict:
    return json.loads((FIXTURES / "v3_exam_row.json").read_text(encoding="utf-8"))


def v3_row(record_type: str, **overrides) -> dict:
    """A minimal v3 supervised row of ``record_type``.

    Only the fields ``verify_v3.ROW_REQUIRED`` puts on every kind, plus the
    per-kind extras. Anything the harness reads that is NOT here is a field the
    harness is inventing.
    """
    row = {
        "id": f"cosimov3_{record_type}_00000000deadbeef",
        "record_type": record_type,
        "work_type": WORK_TYPE,
        "scenario_id": SCENARIO,
        # The two columns the amendment added so the harness can refuse a leak
        # without re-deriving one: `family` was only ever recoverable by slicing
        # `scenario_id` at the length of a work type that itself contains dots,
        # and `holdout` was not on the row at all -- so a row did not know
        # whether it was allowed to train.
        "family": SCENARIO[len(WORK_TYPE) + 1 :],
        "holdout": False,
        "variant": 3,
        "question": "What is the one-day VaR?",
        "answer": "The one-day 99% VaR is 4.2m.",
        # No `messages`. A prose row's prompt is its question; the transcript it
        # used to carry was the teacher's brief, and that is exactly what the
        # harness must never train on. The kinds whose target really is a
        # transcript (agentic) or whose prompt really differs from the pack
        # question (exam) pass their own via **overrides.
        "register": "desk_chat",
        "fact_pack": {"scenario_id": SCENARIO, "work_type": WORK_TYPE},
        "verified": True,
        "verification": {
            "computed_by": "dataset.pipelines.v3.packs.risk_var",
            "pack_seed": "88405e883240d1ed",
            "teacher": {"model": "deepseek-v4-flash", "think_present": False},
            "attempts": 1,
            "invented_numbers": [],
            "missing_mentions": [],
            "register_ok": True,
            "render": {"kind": record_type, "attempts": 1},
        },
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------
# the corpus discriminator
# --------------------------------------------------------------------------


def test_the_id_namespace_identifies_the_corpus(exam_row):
    assert data_schema.is_v3_row(exam_row)
    assert data_schema.is_v3_row({"id": "cosimov3pref_00000000deadbeef"})
    # v1 and v2 ids must not be mistaken for v3: `cosimopref_` is a prefix of
    # nothing in the v3 namespace, and the two differ by three characters.
    assert not data_schema.is_v3_row({"id": "cosimo_exam_00123_ab12cd34"})
    assert not data_schema.is_v3_row({"id": "cosimopref_00123_ab12cd34"})
    assert not data_schema.is_v3_row({})


def test_scenario_family_slices_on_the_work_type_length():
    """`work_type` contains dots, so this can never be a `split(".")`."""
    assert data_schema.scenario_family(WORK_TYPE, SCENARIO) == FAMILY
    assert (
        data_schema.scenario_family(
            "valuation.equity.dcf", "valuation.equity.dcf.mature_consumer"
        )
        == "mature_consumer"
    )
    # A row whose two fields disagree keeps its whole id as the family rather
    # than silently joining another family's stratum.
    assert data_schema.scenario_family("other.type", SCENARIO) == SCENARIO
    assert data_schema.scenario_family("", SCENARIO) == SCENARIO


# --------------------------------------------------------------------------
# the axis mapping that lets splits.py stay corpus-blind
# --------------------------------------------------------------------------


def test_v3_axes_map_onto_the_split_and_holdout_keys(exam_row):
    rec = data_schema.normalize_v3_record(exam_row)
    row = data_schema.to_eval_row(rec)
    assert rec.work_type == "execution.tca.arrival"
    assert rec.scenario_id == "execution.tca.arrival.large_cap_intraday"
    # program/generator/stem_family are what splits.assign_splits reads. If any
    # of the three were blank the corpus would collapse into one stratum and
    # every configured holdout would match nothing -- silently.
    assert row["program"] == rec.work_type
    assert row["generator"] == rec.scenario_id
    assert row["stem_family"] == rec.scenario_id
    # The holdout key is the full scenario id, so two work types cannot collide
    # on a shared family name.
    assert data_schema.stem_family(row["stem_family"]) == row["stem_family"]


def test_v3_rows_carry_no_v1_taxonomy(exam_row):
    rec = data_schema.normalize_v3_record(exam_row)
    assert (rec.topic, rec.subtopic, rec.difficulty) == ("", "", "")
    assert rec.register == "desk_chat"


def test_eval_row_fields_are_the_same_shape_for_both_corpora(exam_row):
    v3 = data_schema.to_eval_row(data_schema.normalize_v3_record(exam_row))
    assert tuple(v3) == data_schema.EVAL_FIELDS


# --------------------------------------------------------------------------
# exam: the one kind with a grading contract
# --------------------------------------------------------------------------


def test_exam_question_is_the_item_text_not_the_bare_question_column(exam_row):
    """The options live in `question_text`, not in `question`.

    Preparing the bare `question` column would ask the model to pick a letter
    without ever showing it the letters -- and the row would still look valid.

    The column is a *named* one now rather than `messages[1]`. That index used
    to be the teacher's brief for every other record type, so reading it here
    was reading two different things through one accessor; an exam item's
    prompt is a real column, and the prose kinds have no message list at all.
    """
    rec = data_schema.normalize_v3_record(exam_row)
    assert "Options:" in rec.question
    assert "(A)" in rec.question and "(D)" in rec.question
    assert rec.question != exam_row["question"]
    assert rec.question == exam_row["question_text"]
    assert rec.question == exam_row["messages"][0]["content"]


def test_exam_target_round_trips_through_build_completion(exam_row):
    """trace + tag + value must reassemble the corpus's own assistant turn.

    Up to one thing: `chat.build_completion` puts a blank line before the
    contract, where v3 uses a single newline. That is deliberate and it is the
    harness's call to make -- exam targets get ONE layout across v1, v2 and v3,
    decided in one place, rather than inheriting three corpora's whitespace
    habits into the same training mix. The assertion is written to prove that
    the separator is the *only* difference, so any other drift still fails.
    """
    rec = data_schema.normalize_v3_record(exam_row)
    assert rec.reasoning_trace
    assert TAG not in rec.reasoning_trace
    rebuilt = chat.build_completion(rec.reasoning_trace, rec.answer, TAG)
    published = exam_row["messages"][-1]["content"].rstrip()
    assert rebuilt != published, "if these ever match, drop the normalisation below"
    assert rebuilt.replace(f"\n\n{TAG}", f"\n{TAG}") == published
    assert rebuilt.endswith(f"{TAG} {rec.answer}")


def test_exam_answer_is_gradeable_as_an_mcq(exam_row):
    from cosimo_ft import grading

    rec = data_schema.normalize_v3_record(exam_row)
    assert rec.question_type == "MCQ"
    assert grading.mcq_letter(rec.answer) is not None
    # Three distractors, none of them the gold option, all letter-prefixed so
    # grading's distractor helpers can read both letter and value off them.
    assert len(rec.distractors) == 3
    assert rec.answer not in rec.distractors
    letters = {grading.mcq_letter(d) for d in rec.distractors}
    assert None not in letters
    assert grading.mcq_letter(rec.answer) not in letters
    # The value the grader will read off the answer is the pack's computed
    # figure. Read through the public helper: the answer keeps the corpus's
    # `<letter> -- <value> <unit>` surface, which parse_number alone cannot
    # resolve and grade_cosimo does not need it to.
    assert exam_row["answer_value"] in grading.numbers_in(rec.answer)
    assert (
        rec.answer
        == exam_row["messages"][-1]["content"].splitlines()[-1].split(TAG)[-1].strip()
    )


def test_exam_row_grades_its_own_answer_correct(exam_row):
    """The corpus's own assistant turn must score correct against its own row.

    If this fails the two halves disagree about the answer format, and every
    exam number the harness ever reports is measuring that disagreement.
    """
    from cosimo_ft import grading

    rec = data_schema.normalize_v3_record(exam_row)
    grade = grading.grade_cosimo(
        data_schema.to_eval_row(rec), exam_row["messages"][-1]["content"], TAG
    )
    assert grade.correct and grade.format_ok


def test_split_final_answer_uses_the_last_tag():
    trace = f"We restate the contract: {TAG} <value>.\nThen we work it."
    text = f"{trace}\n\n{TAG} B -- 28.16 bp"
    assert data_schema.split_final_answer(text, TAG) == (trace, "B -- 28.16 bp")


def test_split_final_answer_without_a_tag_keeps_the_whole_trace():
    assert data_schema.split_final_answer("no contract here", TAG) == (
        "no contract here",
        "",
    )


# --------------------------------------------------------------------------
# the other record types
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record_type",
    [
        data_schema.ANALYSIS,
        data_schema.MEMO,
        data_schema.CRITIQUE,
        data_schema.GROUNDED,
        data_schema.ABSTENTION,
    ],
)
def test_prose_kinds_supervise_the_answer_verbatim(record_type, fake_tokenizer):
    rec = data_schema.normalize_v3_record(v3_row(record_type))
    row = data_schema.to_sft_row(rec, fake_tokenizer, SYSTEM, TAG)
    assert rec.record_type == record_type
    assert not data_schema.is_exam(rec)
    # The grading contract belongs to exam rows and nothing else: appending it
    # to a memo is how a model learns that being Cosimo means five formulaic
    # steps. `memo`/`critique`/`grounded` are new in v3 and would default into
    # the exam branch if the type dispatch missed them.
    assert TAG not in row["completion"]
    assert row["completion"].startswith(rec.answer)
    assert row["text"] == row["prompt"] + row["completion"]


def test_agentic_drops_the_corpus_system_turn(jinja_tokenizer):
    """Two system turns would put the renderer's instructions before the persona."""
    row_in = v3_row(
        data_schema.AGENTIC,
        messages=[
            {"role": "system", "content": "corpus system turn"},
            {"role": "user", "content": "What is the one-day VaR?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "compute_var", "arguments": {"days": 1}},
                    }
                ],
            },
            {"role": "tool", "name": "compute_var", "content": '{"var": 4.2}'},
            {"role": "assistant", "content": "The one-day 99% VaR is 4.2m."},
        ],
        tool_names=["compute_var"],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "compute_var",
                    "description": "Value at risk.",
                    "parameters": {
                        "type": "object",
                        "properties": {"days": {"type": "integer"}},
                        "required": ["days"],
                    },
                },
            }
        ],
    )
    rec = data_schema.normalize_v3_record(row_in)
    assert [m["role"] for m in rec.conversation] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "corpus system turn" not in json.dumps(rec.conversation)

    row = data_schema.to_sft_row(rec, jinja_tokenizer, SYSTEM, TAG)
    assert row["prompt"].count("<|im_start|>system") == 1
    assert SYSTEM in row["prompt"]
    assert "<|tool|>" in row["prompt"] and "compute_var" in row["prompt"]
    # Everything from the first assistant turn is supervised; the interior tool
    # result renders as a user turn so train_on_responses_only masks it.
    assert "<tool_call>" in row["completion"]
    assert "<|im_start|>user\n<tool_response>" in row["completion"]
    assert row["text"] == row["prompt"] + row["completion"]


def test_implementation_target_is_code_tests_and_limitations(fake_tokenizer):
    rec = data_schema.normalize_v3_record(
        v3_row(
            data_schema.IMPLEMENTATION,
            answer="def var(x):\n    return x",
            reference_code="def var(x):\n    return x",
            public_tests=["assert var(1) == 1", "assert var(2) == 2"],
            hidden_tests=["assert var(3) == 3"],
            dirty_fixture={"x": None},
            spec="Return the input.",
            limitations="Assumes a finite input and no missing readings.",
        )
    )
    # v3 renames v2's fields; the substance has to land in code/test_code or the
    # supervised target is a bare one-line answer.
    assert rec.code == "def var(x):\n    return x"
    assert "assert var(1) == 1" in rec.test_code
    assert "assert var(2) == 2" in rec.test_code
    # `answer` carries the one teacher-authored field, so the composed target is
    # code, then tests, then the prose -- not the reference code twice, which is
    # what taking v3's `answer` verbatim would produce.
    assert rec.answer.startswith("Assumes a finite input")
    completion = data_schema.build_supervised_completion(rec, TAG)
    assert completion.count("```python") == 2
    assert completion.endswith("Assumes a finite input and no missing readings.")
    assert "def var(x)" in completion
    assert TAG not in completion
    # The hidden tests stay out of the supervised target: they are the corpus's
    # own gate, and training on them would be training on the answer key.
    assert "assert var(3) == 3" not in completion


def test_no_v3_kind_is_blank_by_the_harness_reading(fake_tokenizer):
    """`is_blank_record` applies a different rule per record type.

    A kind it does not know falls through to "the answer must be non-empty",
    which is right for the three new prose types and wrong for the two that
    carry their substance elsewhere.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_data",
        Path(__file__).resolve().parents[1] / "scripts" / "01_prepare_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for record_type in (
        data_schema.ANALYSIS,
        data_schema.MEMO,
        data_schema.CRITIQUE,
        data_schema.GROUNDED,
        data_schema.ABSTENTION,
    ):
        rec = data_schema.normalize_v3_record(v3_row(record_type))
        assert not module.is_blank_record(rec), record_type


# --------------------------------------------------------------------------
# preference
# --------------------------------------------------------------------------


def v3_pref_row(**overrides) -> dict:
    row = {
        "id": "cosimov3pref_00000000cafef00d",
        "record_type": "preference",
        "source_sft_id": "cosimov3_analysis_00000000deadbeef",
        "work_type": WORK_TYPE,
        "scenario_id": SCENARIO,
        "variant": 3,
        "parent_kind": "analysis",
        "pitfall": "false_precision",
        "question": "What is the one-day VaR?",
        "prompt": [
            {"role": "system", "content": "You are sitting a Cosimo item."},
            {"role": "user", "content": "Desk brief. What is the one-day VaR?"},
        ],
        "chosen": "Around 4.2m on a one-day 99% horizon, with the usual tail caveat.",
        "rejected": "Exactly 4.2137m, which fully characterises the tail.",
        "register": "desk_chat",
        "verification": {"computed_by": "x", "pack_seed": "y", "teacher": {}},
    }
    row.update(overrides)
    return row


def test_v3_preference_prompt_is_read_from_the_message_list(fake_tokenizer):
    """v3's `prompt` is a message LIST, where v2's is a string.

    Reading it as text would stringify the list into the question, and the
    failure is invisible: the pair still renders, still differs, still trains.
    """
    rec = data_schema.normalize_v3_pref_row(v3_pref_row())
    assert rec.question == "Desk brief. What is the one-day VaR?"
    assert "role" not in rec.question and "{" not in rec.question
    assert data_schema.has_preference(rec)

    row = data_schema.to_pref_row(rec, fake_tokenizer, SYSTEM, TAG)
    assert row["chosen"] != row["rejected"]
    # No gold value and neither side is an exam answer, so bolting the grading
    # contract onto them would train the exam shape onto rows about judgement.
    assert TAG not in row["chosen"] and TAG not in row["rejected"]
    assert row["prompt"].endswith("<|im_start|>assistant\n")


def test_v3_preference_ids_are_disjoint_from_supervised_ids():
    """The overlap that made the first DPO run a zero-gradient no-op."""
    pref = data_schema.normalize_v3_pref_row(v3_pref_row())
    sft = data_schema.normalize_v3_record(v3_row(data_schema.ANALYSIS))
    assert pref.id != sft.id
    assert pref.id.startswith(data_schema.V3_PREFERENCE_ID_PREFIX)
    assert sft.id.startswith(data_schema.V3_ID_PREFIX)
    assert not pref.id.startswith(data_schema.V3_ID_PREFIX)


def test_v3_preference_carries_the_pitfall_as_its_mode():
    rec = data_schema.normalize_v3_pref_row(v3_pref_row())
    assert rec.pitfall == "false_precision"
    assert rec.pref_mode == "false_precision"
    assert rec.record_type == data_schema.PREFERENCE


# --------------------------------------------------------------------------
# amendment §A / §G: the two leak gates, on this side of the join
# --------------------------------------------------------------------------


def test_a_row_carrying_the_teacher_brief_is_refused():
    """The fingerprint gate, which the corpus side also runs.

    Two gates on one rule is deliberate. `dataset` and `jobs` join on the Hub
    and neither can see the other's tree, so "the corpus would never ship that"
    is a claim this side cannot verify. An old shard tree, a hand-edited file
    or a `local_dir` pointed at last month's render all produce exactly this
    row -- and the failure it causes downstream is a model that answers a desk
    question by reciting a JSON contract, which nobody would diagnose from a
    loss curve.
    """
    row = v3_row("analysis")
    assert not data_schema.carries_teacher_brief(row)
    leaked = v3_row(
        "analysis",
        messages=[
            {"role": "system", "content": "You are the Cosimo v3 teacher, writing..."},
            {"role": "user", "content": "Answer strictly from the fact pack below."},
            {"role": "assistant", "content": "The one-day 99% VaR is 4.2m."},
        ],
    )
    assert data_schema.carries_teacher_brief(leaked)
    # It leaks the same however it is spelled: the check reads the whole row,
    # not a message list, because a brief pasted into `question` trains just as
    # hard as one left in `messages`.
    in_question = v3_row(
        "analysis", question="You are the Cosimo v3 teacher. What is the VaR?"
    )
    assert data_schema.carries_teacher_brief(in_question)


def test_a_holdout_row_is_refused_even_though_the_corpus_should_not_emit_one():
    """Defence in depth for the one leak no downstream number can un-certify."""
    assert not data_schema.is_holdout(v3_row("analysis"))
    assert data_schema.is_holdout(v3_row("analysis", holdout=True))


def test_verified_is_the_rows_own_claim_not_a_proxy_for_a_stamp():
    """§G.2. The old reading was "carries a verification stamp", which a row
    that failed its gate and was written anyway satisfies just as well."""
    assert prepare.is_verified(v3_row("analysis")) is True
    assert prepare.is_verified(v3_row("analysis", verified=False)) is False
    unclaimed = v3_row("analysis")
    del unclaimed["verified"]
    assert prepare.is_verified(unclaimed) is False
    # The stamp is still required alongside the claim: asserting `verified`
    # with no provenance behind it is asserting something nobody can check.
    assert prepare.is_verified(v3_row("analysis", verification={})) is False


def test_the_agentic_transcript_loses_the_factory_turns_not_the_tool_turns():
    """§G.1: ignore `messages` except for agentic, and even then strip it."""
    rec = data_schema.normalize_v3_record(
        v3_row(
            "agentic",
            messages=[
                {"role": "system", "content": "You are the Cosimo v3 teacher..."},
                {"role": "user", "content": "Work the goal with the tools."},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": '{"var": 4.2}'},
                {"role": "assistant", "content": "The one-day 99% VaR is 4.2m."},
            ],
        )
    )
    roles = [m["role"] for m in rec.conversation]
    assert "system" not in roles
    assert roles == ["user", "assistant", "tool", "assistant"], roles


def test_the_exam_share_cap_thins_an_exam_heavy_tree():
    """§G.4: the prepared mix is a claim only this side can make good on."""
    records = [
        data_schema.normalize_v3_record(
            v3_row("exam", id=f"cosimov3_exam_{i:016x}", question_text="Q?")
        )
        for i in range(40)
    ] + [
        data_schema.normalize_v3_record(
            v3_row("analysis", id=f"cosimov3_analysis_{i:016x}")
        )
        for i in range(60)
    ]
    kept, dropped = prepare.cap_exam_share(records, 0.18, seed=3407)
    exam = [r for r in kept if data_schema.is_exam(r)]
    assert dropped == 40 - len(exam)
    assert len(exam) / len(kept) <= 0.18 + 1e-9
    # A mix already inside the band is left alone -- the cap is a ceiling, not
    # a target, and thinning a compliant corpus would be discarding data.
    assert prepare.cap_exam_share(kept, 0.18, seed=3407) == (kept, 0)


def test_the_fact_pack_reaches_the_eval_sidecar_and_not_a_chat_turn():
    """§G.5. Scoring an invented-number rate needs the row's own contract; a
    prompt that carried it would be the labelling protocol under a new name."""
    row = v3_row("analysis", fact_pack={"canonical": {"var95": 4.2}, "question": "Q?"})
    rec = data_schema.normalize_v3_record(row)
    assert rec.fact_pack == {"canonical": {"var95": 4.2}, "question": "Q?"}
    eval_row = data_schema.to_eval_row(rec)
    assert eval_row["fact_pack"]["canonical"] == {"var95": 4.2}
    assert tuple(eval_row) == data_schema.EVAL_FIELDS
    assert "canonical" not in rec.question
