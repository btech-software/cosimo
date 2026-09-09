"""The preference lane: the draw, the two ladders, the copy gate, the board (spec §5.7).

Five claims, each a test below:

* the **draw is the dice of the seed, not of the hour**: which rows pair and
  what crime each carries is recomputible from coordinates alone, and two
  parents of one coordinate are two pairs (the salt);
* a **pair ships whole or not at all**: through the committed fixture, every
  written pair clears its own contract when re-graded from the recomputed
  pack, and its recorded shingle overlaps sit under the set threshold;
* the **ladders climb**: a chosen side that is the target byte-for-byte draws
  a repair turn naming the offence and a cooler temperature;
* a **rejected side that does not commit its crime is not training signal**:
  it dies to the dead letter with the detector's sentence in the reason, and
  a rejected side that is a near copy of the chosen dies the same way --
  v1's loss-0.0 pair, refused at the door;
* the **board lights for exactly those failures**: tamper the stored bytes
  and axis 12 names the tampered clause, and nothing else burns.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_preference_fixture as pref_harness  # noqa: E402
import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import config, inventory, stage, write  # noqa: E402
from pipelines.v3.packs import compute_pack  # noqa: E402
from pipelines.v3.prefer import run_prefer_stage, select_pair_jobs  # noqa: E402
from pipelines.v3.render.prose import run_render_stage  # noqa: E402
from pipelines.v3.teacher import Teacher  # noqa: E402
from pipelines.v3.teacher.client import FixtureTransport  # noqa: E402
from pipelines.v3.verification.preference import (  # noqa: E402
    PREFER_KIND,
    TAG_COPY,
    TAG_PITFALL,
    pair_draw,
    pair_gate_violations,
    pair_id,
    shingle_overlap,
)
from pipelines.v3.verify_v3 import _check_pair  # noqa: E402

COMMITTED = os.path.join(_HERE, "fixtures", pref_harness.PREF_FIXTURE_NAME)
PROSE_COMMITTED = os.path.join(_HERE, "fixtures", prose_harness.PROSE_FIXTURE_NAME)


class Scripted:
    """A teacher whose every answer is chosen by the test."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def post(self, body):
        self.calls.append(body)
        if not self._replies:
            raise AssertionError("stage asked the teacher more times than scripted")
        return self._replies.pop(0)


class CountingFixture(FixtureTransport):
    """Replay, and count -- so 'resume asked nobody' is an assertion, not a hope."""

    def __init__(self, path):
        super().__init__(path=path)
        self.calls = 0

    def post(self, body):
        self.calls += 1
        return super().post(body)


def _reply(text, model="fixture", think=None):
    message = {"content": text, "role": "assistant"}
    if think is not None:
        message["reasoning_content"] = think
    return {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"total_tokens": max(1, len(text.split()))},
    }


def _plan():
    return inventory.load_plan(config.taxonomy_path())


def _pin_prose(monkeypatch):
    """Offline replay keys on the model name; the prose fixture was captured
    under the prose dummy, so the render stage must ask under that name."""
    monkeypatch.setenv(config.TEACHER_PROSE_ENV, prose_harness.DUMMY_MODEL)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, prose_harness.DUMMY_MODEL)
    monkeypatch.delenv(config.LIVE_ENV, raising=False)


def _render_rows(
    out: str, monkeypatch, *, limit: int = 6, kind: str = "analysis"
) -> int:
    """Ship *limit* rows of *kind* through the real prose replay, offline."""
    _pin_prose(monkeypatch)
    selected = [
        job for job in inventory.expand_jobs(_plan()) if job.record_type == kind
    ]
    stage.run_pack_stage(out, selected)
    rendered = run_render_stage(
        out,
        selected,
        Teacher(FixtureTransport(PROSE_COMMITTED)),
        types=(kind,),
        limit=limit,
    )
    return rendered["rendered"]


def _shipped_rows(out: str, kind: str = "analysis"):
    return write.read_jsonl(write.path_for("sft", kind, out))


def _pin_pre(monkeypatch):
    monkeypatch.setenv(config.TEACHER_PROSE_ENV, pref_harness.DUMMY_MODEL)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, pref_harness.DUMMY_MODEL)
    monkeypatch.delenv(config.LIVE_ENV, raising=False)


# ------------------------------------------------------------------ the draw


def test_the_draw_is_the_dice_of_the_seed_never_of_the_hour():
    licensed = ["false_precision", "wrong_assumption"]
    first = [
        pair_draw("valuation.equity.dcf", "mature_consumer", "analysis", v, licensed)
        for v in range(40)
    ]
    again = [
        pair_draw("valuation.equity.dcf", "mature_consumer", "analysis", v, licensed)
        for v in range(40)
    ]
    assert first == again, "the same coordinates drew a different pair set"
    assert any(paired for paired, _ in first), "nobody paired across forty variants"
    assert not all(paired for paired, _ in first), (
        "everybody paired: the probability is a throttle, not a rail"
    )


def test_two_parents_of_one_coordinate_are_two_pairs():
    a = pair_id("valuation.equity.dcf", "mature_consumer", "analysis", 3)
    m = pair_id("valuation.equity.dcf", "mature_consumer", "memo", 3)
    assert a != m, "unsalted ids collide: the second pair would never be written"
    assert a.startswith(config.PREFERENCE_ID_PREFIX + "_")
    assert m.startswith(config.PREFERENCE_ID_PREFIX + "_")


# ------------------------------------------------------------------- shipping


def test_a_pair_ships_whole_or_not_at_all(tmp_path, monkeypatch):
    out = str(tmp_path / "corpus")
    rows = _render_rows(out, monkeypatch, limit=6)
    assert rows == 6
    _pin_pre(monkeypatch)
    report = run_prefer_stage(
        out, Teacher(FixtureTransport(COMMITTED)), types=("analysis",), limit=5
    )
    assert report["rendered"] >= 1
    assert report["dead_lettered"] == 0
    pairs = write.read_jsonl(write.path_for("preference", "pairs", out))
    assert len(pairs) == report["rendered"]
    shipped = {row["id"]: row for row in _shipped_rows(out)}
    for pair in pairs:
        parent = shipped[pair["source_sft_id"]]
        assert pair["source_sft_id"] != pair["id"]
        assert pair["record_type"] == PREFER_KIND
        pack = compute_pack(
            pair["work_type"],
            pair["scenario_id"][len(pair["work_type"]) + 1 :],
            pair["variant"],
        ).to_dict()
        assert pair_gate_violations(pack, parent, pair) == [], pair["id"]
        rendered = pair["verification"]["render"]
        assert rendered["shingle_overlap"]["target"] <= config.PREF_MAX_SHINGLE_OVERLAP
        assert rendered["shingle_overlap"]["pair"] <= config.PREF_MAX_SHINGLE_OVERLAP


def test_ids_never_share_a_namespace_with_the_supervised_corpus(tmp_path, monkeypatch):
    out = str(tmp_path / "corpus")
    _render_rows(out, monkeypatch, limit=6)
    _pin_pre(monkeypatch)
    run_prefer_stage(
        out, Teacher(FixtureTransport(COMMITTED)), types=("analysis",), limit=5
    )
    pair_ids = {
        row["id"]
        for row in write.read_jsonl(write.path_for("preference", "pairs", out))
    }
    row_ids = {
        row["id"]
        for kind in config.PREF_PROBABILITIES
        for row in _shipped_rows(out, kind)
    }
    assert pair_ids and pair_ids.isdisjoint(row_ids), (
        "v1 nested pairs on row ids; v2 ended that"
    )


def test_resume_asks_the_teacher_nobody(tmp_path, monkeypatch):
    out = str(tmp_path / "corpus")
    _render_rows(out, monkeypatch, limit=6)
    _pin_pre(monkeypatch)
    first = CountingFixture(COMMITTED)
    report = run_prefer_stage(out, Teacher(first), types=("analysis",), limit=5)
    assert report["rendered"] >= 1
    second = CountingFixture(COMMITTED)
    again = run_prefer_stage(out, Teacher(second), types=("analysis",), limit=5)
    assert again["rendered"] == 0 and again["existing"] == report["rendered"]
    assert second.calls == 0, "a pair already on the shard recalled the teacher"


# ------------------------------------------------------------------ the ladders


def _scripted_pair_setup(tmp_path, monkeypatch):
    """One drawn pair's material, rendered for the ladders to climb."""
    out = str(tmp_path / "corpus")
    _render_rows(out, monkeypatch, limit=6)
    jobs = inventory.expand_jobs(_plan())
    selected = [job for job in jobs if job.record_type == "analysis"]
    stage.run_pack_stage(out, selected)
    drawn = select_pair_jobs(out, _plan(), types=("analysis",), limit=1)
    assert drawn, "the draw paired nobody from six shipped rows"
    kind, row, pitfall = drawn[0]
    pack = compute_pack(
        row["work_type"],
        row["scenario_id"][len(row["work_type"]) + 1 :],
        row["variant"],
    ).to_dict()
    answer = row["messages"][2]["content"]
    return out, kind, row, pitfall, pack, answer


def test_a_chosen_that_is_the_target_draws_a_repair_and_a_cooler(tmp_path, monkeypatch):
    out, kind, row, pitfall, pack, answer = _scripted_pair_setup(tmp_path, monkeypatch)
    good = pref_harness.compose_chosen(pack, kind)
    transport = Scripted(
        [
            _reply(answer),  # the v1 disease, offered as the telling
            _reply(good),  # the correction, after the repair turn
            _reply(
                pref_harness.compose_rejected(
                    pack, row["work_type"], pitfall, answer, good
                )
            ),
        ]
    )
    from pipelines.v3.prefer import build_pair_row  # late: the unit under test

    outcome = build_pair_row(Teacher(transport), kind, row, pitfall)
    assert outcome["row"] is not None
    attempts = outcome["row"]["verification"]["render"]["attempts"]
    assert attempts["chosen"] == 2, (
        "the byte-identical restatement should have been sent back"
    )
    assert len(transport.calls) == 3
    repair = transport.calls[1]["messages"][-1]["content"]
    assert "target" in repair and "objections" in repair.lower()
    assert transport.calls[1]["temperature"] == config.PREF_TEMPERATURES[1]
    assert transport.calls[2]["temperature"] == config.PREF_TEMPERATURES[0]
    assert transport.calls[2]["messages"][-1]["role"] == "user"
    assert "defect" in transport.calls[2]["messages"][-1]["content"].casefold()


def test_a_rejected_that_commits_no_crime_is_dead_letter_not_training(
    tmp_path, monkeypatch
):
    out, kind, row, pitfall, pack, answer = _scripted_pair_setup(tmp_path, monkeypatch)
    good = pref_harness.compose_chosen(pack, kind)
    innocent = prose_harness.compliant_text(pack, kind)  # fluent, on-pack, crimeless
    transport = Scripted([_reply(good), _reply(innocent), _reply(innocent)])
    from pipelines.v3.prefer import build_pair_row  # late, as above

    outcome = build_pair_row(Teacher(transport), kind, row, pitfall)
    assert outcome["row"] is None
    dead = outcome["dead_letter"]
    assert dead["reason"].startswith("gate:")
    assert TAG_PITFALL in dead["reason"] or "not fluent" in dead["reason"]
    assert dead["attempts"]["rejected"] == config.PREF_ATTEMPTS
    assert (
        dead["exchange"][-2]["role"] == "assistant"
        and dead["exchange"][-1]["role"] == "user"
    )


def test_a_rejected_near_copy_of_the_chosen_dies_the_same_way(tmp_path, monkeypatch):
    _out, kind, row, pitfall, pack, answer = _scripted_pair_setup(tmp_path, monkeypatch)
    good = pref_harness.compose_chosen(pack, kind)
    transport = Scripted([_reply(good), _reply(good), _reply(good)])
    from pipelines.v3.prefer import build_pair_row  # late, as above

    outcome = build_pair_row(Teacher(transport), kind, row, pitfall)
    assert outcome["row"] is None
    assert "near copy of the chosen side" in outcome["dead_letter"]["reason"]
    assert shingle_overlap(good, good) == 1.0 > config.PREF_MAX_SHINGLE_OVERLAP


# ----------------------------------------------------------------- the board


def _pair_corpus(tmp_path, monkeypatch):
    """A shipped corpus with one real pair, returned with its parts."""
    out = os.path.join(str(tmp_path), "pair-corpus")
    _render_rows(out, monkeypatch, limit=6)
    jobs = inventory.expand_jobs(_plan())
    stage.run_pack_stage(out, [job for job in jobs if job.record_type == "analysis"])
    kind, row, pitfall = select_pair_jobs(out, _plan(), types=("analysis",), limit=1)[0]
    pack = compute_pack(
        row["work_type"],
        row["scenario_id"][len(row["work_type"]) + 1 :],
        row["variant"],
    ).to_dict()
    good = pref_harness.compose_chosen(pack, kind)
    answer = row["messages"][2]["content"]
    rejected = pref_harness.compose_rejected(
        pack, row["work_type"], pitfall, answer, good
    )
    pair = {
        "id": pair_id(
            row["work_type"],
            row["scenario_id"][len(row["work_type"]) + 1 :],
            kind,
            row["variant"],
        ),
        "record_type": PREFER_KIND,
        "source_sft_id": row["id"],
        "work_type": row["work_type"],
        "scenario_id": row["scenario_id"],
        "variant": row["variant"],
        "parent_kind": kind,
        "pitfall": pitfall,
        "question": pack["question"],
        "prompt": row["messages"][:2],
        "chosen": good,
        "rejected": rejected,
        "register": pack["register"],
        "verification": {
            "pack_seed": f"{pack['seed']:016x}",
            "render": {"kind": PREFER_KIND},
        },
    }
    return out, pair, pack, row


def _loader_for(out, kind):
    rows = {r.get("id"): r for r in write.read_jsonl(write.path_for("sft", kind, out))}
    return lambda wanted, source_id: rows.get(source_id) if wanted == kind else None


def test_the_pristine_pair_leaves_axis_12_quiet(tmp_path, monkeypatch):
    out, pair, pack, row = _pair_corpus(tmp_path, monkeypatch)
    failures = _check_pair(pair, frozenset(), _loader_for(out, pair["parent_kind"]))
    assert failures["preference disjointness"] == []
    for axis, problems in failures.items():
        assert not problems, f"{axis}: {problems}"


def test_a_chosen_that_is_the_target_lights_the_copy_axis(tmp_path, monkeypatch):
    out, pair, _pack, row = _pair_corpus(tmp_path, monkeypatch)
    tampered = dict(pair, chosen=row["messages"][2]["content"])
    failures = _check_pair(tampered, frozenset(), _loader_for(out, pair["parent_kind"]))
    assert any(TAG_COPY in p for p in failures["preference disjointness"])
    assert any("SFT target" in p for p in failures["preference disjointness"])


def test_an_innocent_rejected_lights_the_copy_axis_too(tmp_path, monkeypatch):
    out, pair, pack, row = _pair_corpus(tmp_path, monkeypatch)
    innocent = prose_harness.compliant_text(pack, pair["parent_kind"])
    tampered = dict(pair, rejected=innocent)
    failures = _check_pair(tampered, frozenset(), _loader_for(out, pair["parent_kind"]))
    lit = failures["preference disjointness"]
    assert lit, "a rejected side that commits nothing slipped past the board"
    assert any(TAG_PITFALL in problem for problem in lit)


def test_a_foreign_id_lights_the_copy_axis_and_the_hash_axis(tmp_path, monkeypatch):
    out, pair, _pack, _row = _pair_corpus(tmp_path, monkeypatch)
    tampered = dict(pair, id=config.supervised_id("analysis", 0))
    failures = _check_pair(tampered, frozenset(), _loader_for(out, pair["parent_kind"]))
    assert any(TAG_COPY in p for p in failures["preference disjointness"])
    assert failures["pack recompute"], "a pasted id must fail the hash axis as well"


def test_a_vanished_parent_is_a_finding_not_a_forgiveness(tmp_path, monkeypatch):
    out, pair, _pack, _row = _pair_corpus(tmp_path, monkeypatch)
    failures = _check_pair(pair, frozenset(), lambda kind, source_id: None)
    assert any(
        "outlived its own parent" in p for p in failures["preference disjointness"]
    )
