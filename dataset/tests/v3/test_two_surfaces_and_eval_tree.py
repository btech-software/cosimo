"""The two surfaces (§A) and the eval tree (§E), exercised through the stage.

These are the amendment's two structural changes and neither is visible in a
single row: §A is about what a render *writes* -- one trainable object and one
debug object, in different places, under different rules -- and §E is about
*where* a row lands. Both are properties of the stage, so both are tested by
running it.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config, inventory, row as rowlib, stage, write  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.render.prose import run_render_stage, select_prose_jobs  # noqa: E402
from pipelines.v3.teacher import Teacher  # noqa: E402

KIND = "analysis"


class Scripted:
    """One prepared answer per pack, looked up by the question in the brief."""

    def __init__(self, by_question: dict[str, str]):
        self._by_question = by_question
        self.asked = 0

    def post(self, body: dict) -> dict:
        self.asked += 1
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        text = next(
            (a for q, a in self._by_question.items() if q in user),
            "",
        )
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "model": "scripted",
            "usage": {"total_tokens": max(1, len(text.split()))},
        }


def _teacher_for(jobs) -> Teacher:
    """One prepared answer per *renderable* job.

    Coordinates the pack guard refuses are skipped rather than raising: the
    renderer never asks about them either (the packs stage recorded the skip),
    so a harness that insisted on them would be preparing answers to questions
    nobody poses.
    """
    answers = {}
    for job in jobs:
        try:
            pack = compute_pack(job.work_type, job.family, job.variant).to_dict()
        except PackError:
            continue
        answers[pack["question"]] = prose_harness.compliant_text(pack, KIND)
    return Teacher(Scripted(answers))


@pytest.fixture
def plan():
    return inventory.load_plan(config.taxonomy_path())


def _render(out, jobs, *, holdout):
    stage.run_pack_stage(out, jobs)
    return run_render_stage(
        out, jobs, _teacher_for(jobs), types=(KIND,), limit=None, holdout=holdout
    )


# --------------------------------------------------------------------------
# §E: the holdout families render, and they render somewhere else
# --------------------------------------------------------------------------


def test_holdout_families_render_into_eval_and_never_into_sft(plan, tmp_path):
    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(plan)
    # A wide window on purpose: the first holdout coordinates belong to
    # `execution.tca.arrival/etf_rebalance`, whose guard legitimately refuses a
    # run of them (an ETF rebalance drawn at 58% of ADV is not an aggressive
    # fill, it is an impossible premise), and a four-job window would measure
    # the guard rather than the routing.
    holdout_jobs = select_prose_jobs(jobs, types=(KIND,), limit=None, holdout=True)
    assert holdout_jobs, "the plan declares holdout families; they must be renderable"

    report = _render(out, holdout_jobs, holdout=True)
    assert report["bucket"] == "eval"
    assert report["rendered"] > 0

    eval_rows = write.read_jsonl(write.path_for("eval", KIND, out))
    assert len(eval_rows) == report["rendered"]
    assert all(row["holdout"] is True for row in eval_rows)
    # Not a flag on a row in the training tree: a different directory. A glob
    # cannot confuse two paths the way a boolean gets forgotten.
    assert not os.path.exists(write.path_for("sft", KIND, out))


def test_train_families_still_render_into_sft(plan, tmp_path):
    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(plan)
    train_jobs = select_prose_jobs(jobs, types=(KIND,), limit=4, holdout=False)

    report = _render(out, train_jobs, holdout=False)
    assert report["bucket"] == "sft"
    rows = write.read_jsonl(write.path_for("sft", KIND, out))
    assert rows and all(row["holdout"] is False for row in rows)
    assert not os.path.exists(write.path_for("eval", KIND, out))


def test_the_two_cohorts_are_disjoint_sets_of_coordinates(plan):
    """The selector partitions, it does not filter.

    Before §E the holdout half was dropped outright, which is why the harness
    had to hold out *shipped* families instead -- there was nothing downstream
    to hold out, so the generalisation number was taken on families the model
    had trained on.
    """
    jobs = inventory.expand_jobs(plan)
    train = select_prose_jobs(jobs, types=(KIND,), limit=None, holdout=False)
    held = select_prose_jobs(jobs, types=(KIND,), limit=None, holdout=True)
    assert train and held
    assert not {(j.work_type, j.family) for j in train} & {
        (j.work_type, j.family) for j in held
    }
    assert len(train) + len(held) == sum(1 for j in jobs if j.record_type == KIND)


# --------------------------------------------------------------------------
# §A: two objects, one trainable
# --------------------------------------------------------------------------


def test_the_teacher_log_is_off_by_default(plan, tmp_path, monkeypatch):
    monkeypatch.delenv(config.KEEP_TEACHER_MESSAGES_ENV, raising=False)
    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(plan)
    selected = select_prose_jobs(jobs, types=(KIND,), limit=3)
    report = _render(out, selected, holdout=False)
    assert report["rendered"] > 0
    assert report["teacher_logs"] == 0
    assert not os.path.isdir(os.path.join(out, "teacher_logs"))


def test_the_teacher_log_carries_what_the_row_refuses(plan, tmp_path, monkeypatch):
    """The debug surface exists so the post-mortem does not need the row."""
    monkeypatch.setenv(config.KEEP_TEACHER_MESSAGES_ENV, "1")
    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(plan)
    selected = select_prose_jobs(jobs, types=(KIND,), limit=3)
    report = _render(out, selected, holdout=False)
    assert report["teacher_logs"] == report["rendered"]

    rows = write.read_jsonl(write.path_for("sft", KIND, out))
    for row in rows:
        log_path = write.teacher_log_path(KIND, row["id"], out)
        assert os.path.isfile(log_path), row["id"]
        with open(log_path, encoding="utf8") as handle:
            log = json.load(handle)
        # The log has the brief; the row does not. That is the whole change.
        assert any(m["role"] == "system" for m in log["messages"])
        assert config.TEACHER_FINGERPRINT in json.dumps(log)
        assert not rowlib.carries_teacher_brief(row)
        assert "messages" not in row
        # And it is outside the shard layout, so nothing that walks the corpus
        # tree can pick it up: `teacher_logs` is not one of write.KINDS.
        assert "teacher_logs" not in write.KINDS
        assert log["id"] == row["id"]


def test_a_student_row_states_everything_a_reader_needs(plan, tmp_path):
    out = str(tmp_path / "corpus")
    jobs = inventory.expand_jobs(plan)
    selected = select_prose_jobs(jobs, types=(KIND,), limit=2)
    _render(out, selected, holdout=False)
    rows = write.read_jsonl(write.path_for("sft", KIND, out))
    assert rows
    for row in rows:
        pack = compute_pack(row["work_type"], row["family"], row["variant"]).to_dict()
        # `family` is stated rather than recovered by slicing a scenario id at
        # the length of a work type that itself contains dots.
        assert row["family"] == rowlib.family_of(row)
        assert row["question"] == pack["question"]
        # `fact_pack` is an object, not a string, and it is the pack itself.
        assert isinstance(row["fact_pack"], dict)
        assert row["fact_pack"]["seed"] == pack["seed"]
        assert row["fact_pack"]["canonical"] == pack["canonical"]
        assert row["verified"] is True
        v = row["verification"]
        assert v["invented_numbers"] == [] and v["register_ok"] is True
        assert v["teacher"]["model"]


def test_the_answer_is_stripped_of_think_and_leading_blanks():
    """The committed examples all used to open on two newlines (§A).

    A target that begins with a blank line teaches the model to begin with a
    blank line, and the reasoning channel leaking into `content` teaches
    something considerably worse.
    """
    assert rowlib.visible_answer("\n\nSell 430,567 shares.") == "Sell 430,567 shares."
    assert (
        rowlib.visible_answer("<think>weighing it up</think>\n\nThe call is sell.")
        == "The call is sell."
    )
    # An unclosed opener is a truncated chain of thought; taking it to the end
    # of the text is the only reading that does not ship the whole trace.
    assert rowlib.visible_answer("<think>never finished") == ""
    assert rowlib.visible_answer(None) == ""
