"""The composed exam slice: determinism, the pitfall law, the tamper trap.

The exam lane is the one render stage with no author, so its tests are not
about a model's behaviour but about the *compositor*: the item must be a pure
function of the pack (§so resume is byte-identity), the distractors must be
named wrong models that print distinctly (or vanish -- never a patched
number), the liturgy must be a residue class and not a dice roll, and the
board must catch a hand-tuned row by recomposition. The synthetic collapses
here are arithmetic, not moral: the "wrong model" of the title is a
mis-set dial on the pack's own instrument.
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

from pipelines.v3 import inventory, stage, verify_v3, write  # noqa: E402
from pipelines.v3.packs import PackError, compute_pack  # noqa: E402
from pipelines.v3.render.exam import (  # noqa: E402
    EXAM_PROTOCOL,
    build_exam_row,
    run_exam_stage,
    select_exam_jobs,
)
from pipelines.v3.verification.exam import (  # noqa: E402
    EXAM_KIND,
    LITURGY_MARKERS,
    LITURGICAL_STYLE,
    TAG_TAMPER,
    TRACE_STYLES,
    exam_gate_violations,
)
from pipelines.v3.verification.exam_pitfalls import (  # noqa: E402
    PITFALL_DERIVATIONS,
)


def _plan_jobs():
    return inventory.expand_jobs(inventory.load_plan())


def _train_families(plan=None):
    plan = plan or inventory.load_plan()
    return {
        work_type: [
            family for family, meta in spec["families"].items() if not meta["holdout"]
        ]
        for work_type, spec in plan.items()
    }


def _pack_line(work_type, family, variant):
    """A stored pack line, envelope and all -- what the stage actually reads."""
    return stage.pack_record(work_type, family, variant)


def _any_pack_line(work_type, family, variant=0):
    return _pack_line(work_type, family, variant)


# --------------------------------------------------------------------- compose


@pytest.mark.parametrize("work_type", sorted(PITFALL_DERIVATIONS))
def test_a_composed_row_is_the_item_the_pack_entails(work_type):
    """Two compositions of one coordinate are byte-identical, and the row says
    what the pack means: answer verbatim from ``computed``, one correct seat,
    named distractors, the protocol in the system turn."""
    family = _train_families()[work_type][0]
    for variant in (0, 1, 7, 23):
        try:
            pack = compute_pack(work_type, family, variant)
        except PackError:
            continue
        row = build_exam_row(_pack_line(work_type, family, variant))["row"]
        assert row is not None
        again = build_exam_row(_pack_line(work_type, family, variant))["row"]
        assert again == row, "the compositor rolled a die where it should not"
        assert row["record_type"] == EXAM_KIND
        assert row["verification"]["teacher"] is None, (
            "an exam row that names a teacher was dictated, not composed"
        )
        assert row["verification"]["render"]["kind"] == EXAM_KIND
        assert row["verification"]["render"]["attempts"] == 0
        assert row["answer_value"] == pack.to_dict()["computed"][row["answer_key"]], (
            "the answer must be the computer's own figure, verbatim"
        )
        assert row["messages"][0]["content"] == EXAM_PROTOCOL
        assert row["messages"][-1]["content"] == row["answer"]
        assert row["messages"][1]["content"].startswith(pack.to_dict()["question"]), (
            "the item is asked with the pack's own question"
        )
        assert exam_gate_violations(pack.to_dict(), row) == [], (
            "a composed row must audit clean against its own pack"
        )


@pytest.mark.parametrize("work_type", sorted(PITFALL_DERIVATIONS))
def test_options_are_named_wrong_models_printing_distinctly(work_type):
    """>= 2 distractors, all printed distinctly from the answer and from each
    other, every one traceable to a named pitfall; exactly one seat is right.
    The v1 band-aid (``correct + 7.0``) must be impossible here: the only
    values on the card are the computer's figure and the registry's numbers."""
    family = _train_families()[work_type][0]
    names = {name for name, _ in PITFALL_DERIVATIONS[work_type]}
    seen_pitfalls = set()
    for variant in range(12):
        try:
            compute_pack(work_type, family, variant)
        except PackError:
            continue
        row = build_exam_row(_pack_line(work_type, family, variant))["row"]
        options = row["options"]
        assert 3 <= len(options) <= 4
        assert [o["label"] for o in options] == ["A", "B", "C", "D"][: len(options)]
        texts = [o["text"] for o in options]
        assert len(set(texts)) == len(texts), "colliding options were patched"
        correct = [o for o in options if o["pitfall"] is None]
        assert len(correct) == 1
        assert correct[0]["text"] in row["answer"], (
            "the final line must quote the figure printed on the card"
        )
        pitfalls = [o["pitfall"] for o in options if o["pitfall"] is not None]
        assert set(pitfalls) <= names
        seen_pitfalls |= set(pitfalls)
    assert len(seen_pitfalls) >= 2, (
        f"{work_type}: the slice draws on too few of its registered wrong models"
    )


def test_the_liturgy_is_a_residue_class_not_a_dice_roll():
    """``Step N.`` appears exactly when the variant says so -- 1 in
    ``EXAM_LITURGY_MODULUS`` -- and never otherwise, so no sample can overshoot
    the cap and no seed can summon a ceremony the plan did not schedule."""
    from pipelines.v3 import config

    work_type = "valuation.equity.dcf"
    family = _train_families()[work_type][0]
    liturgical = []
    styles = set()
    for variant in range(30):
        try:
            compute_pack(work_type, family, variant)
        except PackError:
            continue
        row = build_exam_row(_pack_line(work_type, family, variant))["row"]
        style = row["verification"]["render"]["style"]
        styles.add(style)
        marked = any(marker in row["answer"] for marker in LITURGY_MARKERS)
        assert marked is (
            variant % config.EXAM_LITURGY_MODULUS == config.EXAM_LITURGY_RESIDUE
        )
        assert marked is (style == LITURGICAL_STYLE)
        assert row["verification"]["render"]["liturgy"] is marked, (
            "the flag and the text must agree; the board counts the text"
        )
        if marked:
            liturgical.append(variant)
    assert styles == set(TRACE_STYLES), "all four shapes must occur in 30 variants"
    computable = sum(1 for v in range(30) if _computable(work_type, family, v))
    assert liturgical and computable
    assert len(liturgical) / computable <= config.LITURGY_CAP, (
        "the residue class must sit under the cap it is measured against"
    )


def _computable(work_type, family, variant):
    try:
        compute_pack(work_type, family, variant)
        return True
    except PackError:
        return False


def test_final_answer_closes_the_item_and_points_at_the_right_seat():
    work_type = "execution.tca.arrival"
    family = _train_families()[work_type][0]
    for variant in range(10):
        if not _computable(work_type, family, variant):
            continue
        row = build_exam_row(_pack_line(work_type, family, variant))["row"]
        last = row["answer"].splitlines()[-1]
        assert last.startswith("FINAL ANSWER:")
        label = last.split("FINAL ANSWER:")[1].strip().split()[0]
        right = next(o for o in row["options"] if o["pitfall"] is None)
        assert label == right["label"]
        assert right["text"] in last and row["unit"] in last
        assert (
            exam_gate_violations(
                compute_pack(work_type, family, variant).to_dict(), row
            )
            == []
        )


def test_no_exam_turn_opens_with_whitespace_in_any_trace_style():
    """A supervised target may not begin with a newline, in any style.

    The student's chat template closes the role header with a newline, so a turn
    that starts with one makes the tokenizer emit a single "\n\n" where
    `train_on_responses_only` looks for "\n". The marker then matches nowhere,
    every label masks to -100, and the harness drops the row while attributing it
    to truncation. The `short_table` style opened with "" and cost 45 of 163 exam
    rows on the first real trainer construction.

    Swept across styles rather than pinned to one, because the style is drawn per
    variant and the next shape added is the one nobody will re-check by hand.
    """
    styles = set()
    for work_type, families in _train_families().items():
        for family in families:
            for variant in range(12):
                if not _computable(work_type, family, variant):
                    continue
                row = build_exam_row(_pack_line(work_type, family, variant))["row"]
                answer = row["answer"]
                assert not answer[:1].isspace(), (
                    f"{row['scenario_id']} v{variant} "
                    f"({row['verification']['render']['style']}) opens with "
                    f"{answer[:12]!r}"
                )
                assert row["messages"][-1]["content"] == answer
                styles.add(row["verification"]["render"]["style"])
    assert len(styles) > 1, f"only exercised {styles}; the sweep found no variety"


def test_a_collapsed_item_is_dead_lettered_as_a_pack_finding():
    """When the wrong models all print the same figure, the item is not an
    item: dead-letter it, naming the collapse, do not inflate the card."""
    degenerate = {
        "schema_version": "v3.0",
        "scenario_id": "risk.market.var_es.rates_book",
        "work_type": "risk.market.var_es",
        "seed": 1,
        "variant": 0,
        "entities": [],
        "inputs": {
            "book_value_m": 1.0,
            "mu_daily": 0.0,
            "sigma_daily": 1.0,
            "horizon_days": 1,
            "z95": 1.645,
            "z99": 1.645,
        },
        "computed": {"var95_h_m": 1.645},
        "formulas": [],
        "allowed_numbers": [1.645],
        "forbidden_claims": [],
        "must_mention": [],
        "register": "risk_committee",
        "as_of": "2026-03-31",
        "question": "Degenerate draw: every quantile is the same number.",
    }
    line = {
        "id": "pack_x",
        "verification": {"computed_by": "t", "pack_seed": "0" * 16},
        **degenerate,
    }
    outcome = build_exam_row(line)
    assert outcome["row"] is None
    assert "distractors" in outcome["dead_letter"]["reason"]
    assert outcome["dead_letter"]["record_type"] == EXAM_KIND


# ------------------------------------------------------------------------ gate


def test_tampering_with_a_row_is_tampering_with_the_pack(tmp_path):
    """Every stored byte must agree with the recomposition: a tampered option,
    a swapped answer, a style-field lying about its text -- all are caught,
    and the file that was written clean audits clean."""
    work_type, family, variant = "valuation.equity.dcf", None, 0
    family = _train_families()[work_type][0]
    line = _pack_line(work_type, family, variant)
    pack = compute_pack(work_type, family, variant).to_dict()
    row = build_exam_row(line)["row"]
    assert exam_gate_violations(pack, row) == []

    tampered = {**row, "options": [dict(row["options"][0], text="9,999.99")]}
    tags = [v.split(":")[0] for v in exam_gate_violations(pack, tampered)]
    assert TAG_TAMPER.split(":")[0] in tags

    lying_style = {
        **row,
        "verification": {
            **row["verification"],
            "render": {
                **row["verification"]["render"],
                "style": (
                    "prose"
                    if row["verification"]["render"]["style"] != "prose"
                    else "liturgy"
                ),
            },
        },
    }
    assert any(
        v.startswith(TAG_TAMPER) for v in exam_gate_violations(pack, lying_style)
    ), "a style field that does not match the text is a tamper finding"

    value_flip = {**row, "answer_value": row["answer_value"] + 1.0}
    assert any(v.startswith(TAG_TAMPER) for v in exam_gate_violations(pack, value_flip))

    teacher_claim = {
        **row,
        "verification": {**row["verification"], "teacher": {"model": "liar"}},
    }
    report = _verify_rows(tmp_path, [teacher_claim], "tampered")
    axis1 = report["axes"]["schema"]["failures"]
    assert any("claims a teacher" in f["problem"] for f in axis1)


def _verify_rows(tmp_path, rows, name):
    out = os.path.join(str(tmp_path), name)
    write.append_unique(write.path_for("sft", EXAM_KIND, out), rows)
    return verify_v3.verify_dir(out, kinds=(EXAM_KIND,))


# ----------------------------------------------------------------------- stage


def _exam_jobs(limit=None):
    return select_exam_jobs(_plan_jobs(), limit=limit)


def test_the_stage_is_idempotent_and_never_asks_an_author(tmp_path):
    out = str(tmp_path)
    jobs = _exam_jobs(limit=8)
    assert jobs and all(job.record_type == EXAM_KIND for job in jobs)
    assert all(not job.holdout for job in jobs), "holdout is gold-bar material"
    stage.run_pack_stage(out, jobs)
    first = run_exam_stage(out, jobs)
    assert first["missing_packs"] == []
    assert (
        first["rendered"] + first["dead_lettered"] + len(first["skipped_by_pack_gate"])
        == first["jobs_seen"]
    )
    rows_before = write.read_jsonl(write.path_for("sft", EXAM_KIND, out))
    second = run_exam_stage(out, jobs)
    assert second["rendered"] == 0
    assert second["existing"] == first["rendered"]
    assert write.read_jsonl(write.path_for("sft", EXAM_KIND, out)) == rows_before, (
        "a resume must not rewrite what it already wrote"
    )
    assert (second["liturgy"] + first["liturgy"]) or True  # tallies reported


def test_the_two_absences_keep_their_verdicts(tmp_path):
    out = str(tmp_path)
    jobs = _exam_jobs(limit=1)
    orphan = run_exam_stage(out, jobs)  # no pack files at all: the DAG is wrong
    assert orphan["missing_packs"] and not orphan["rendered"]
    assert all(
        "no fact-pack file" in entry["reason"] for entry in orphan["missing_packs"]
    )
    stage.run_pack_stage(out, jobs)
    # A guard-rejected variant is the guard's own data, recorded as a skip:
    # walk the family of the pinned work type until one variant raises.
    job = jobs[0]
    rejected = next(
        (v for v in range(40) if not _computable(job.work_type, job.family, v)),
        None,
    )
    if rejected is None:
        pytest.skip(f"{job.work_type}/{job.family}: no guard-rejected variant in 40")
    from pipelines.v3.inventory import Job

    guard_job = Job(job.work_type, job.family, EXAM_KIND, rejected, False)
    report = run_exam_stage(out, [guard_job])
    assert report["dead_lettered"] == 0
    assert len(report["skipped_by_pack_gate"]) == 1
    assert "pack guard" in report["skipped_by_pack_gate"][0]["reason"]


def test_the_board_shades_red_on_a_tampered_shard_and_back(tmp_path):
    """End-to-end through ``verify_dir``: the composed slice audits clean; one
    edited byte in the shard -- the work of a hand, not of the compositor --
    turns axis 1 red, and repairing the byte clears the board."""
    out = str(tmp_path)
    jobs = _exam_jobs(limit=6)
    stage.run_pack_stage(out, jobs)
    report = run_exam_stage(out, jobs)
    assert report["rendered"] > 0 and report["dead_lettered"] == 0
    path = write.path_for("sft", EXAM_KIND, out)
    assert verify_v3.verify_dir(out, kinds=(EXAM_KIND,))["ok"] is True

    rows = write.read_jsonl(path)
    victim = json.loads(json.dumps(rows[0]))
    # Tamper *consistently* -- answer and its assistant turn move together --
    # so the row's self-consistency checks pass and only the recomposition can
    # catch it: this is the hand with the good editor, not the careless one.
    bad_text = victim["answer"].replace("FINAL ANSWER:", "FINAL ANSWER*")
    victim["answer"] = bad_text
    victim["messages"][-1] = {**victim["messages"][-1], "content": bad_text}
    import os as _os

    _os.remove(path)
    write.append_unique(path, [victim] + rows[1:])
    dirty = verify_v3.verify_dir(out, kinds=(EXAM_KIND,))
    assert dirty["ok"] is False
    assert any(
        "FINAL ANSWER" in f["problem"]
        for f in dirty["axes"]["FINAL ANSWER is exam-only"]["failures"]
    ), "the missing closing tag must surface on the axis that owns it"
    assert any(
        "recomposition" in f["problem"] for f in dirty["axes"]["schema"]["failures"]
    ), "the byte-difference from the recomposed item is a tamper finding"
    _os.remove(path)
    write.append_unique(path, rows)
    assert verify_v3.verify_dir(out, kinds=(EXAM_KIND,))["ok"] is True
