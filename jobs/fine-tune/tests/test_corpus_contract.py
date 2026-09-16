"""Every harness mirror, against the corpus gate it mirrors.

``jobs`` does not import ``dataset``: the two trees are joined by published
artifacts, not by Python, and an eval that imported the generator's gate would
be checking the generator against itself. What that buys is a real second
reading. What it costs is a policy that can drift, and three of them had --
each found only when something downstream produced an absurd number:

* ``register_match`` demanded 250 words of an ``ic_memo``; the corpus writes
  at most 220, so none of its 36 shipped memo rows could pass, and the metric
  ranked a rambling base model above a tuned one.
* ``register_violations`` held an exam row to its pack's register. The corpus
  grades an exam item on the exam contract whatever voice its family speaks,
  and 84 of 188 shipped exam rows disagreed.
* ``must_mention_hits`` matched verbatim substrings after the corpus had moved
  to stems, scoring 51.8% where the gate scored 98.2% -- a correct answer that
  made every point in its own words was marked as having made none.

Each was a single function, obvious once looked at, and invisible for as long
as nobody looked. So this file stops relying on anybody looking: it reads the
corpus's certified rows as *data* -- the boundary rule is about imports, not
about facts -- and asserts that each mirror returns what the row was actually
certified against. A fourth drift is caught by the test that would have caught
the first three, rather than by the next absurd number.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from cosimo_ft import assistant, reward

REPO = pathlib.Path(__file__).resolve().parents[3]
GOLD_BAR = REPO / "dataset" / "goldbar" / "gold_bar_v3.jsonl"
SHARDS = REPO / "dataset" / "shards" / "v3" / "sft"


#: The kinds the corpus holds to a register contract. ``implementation`` and
#: ``agentic`` answer to their own gates -- the board returns on them before it
#: reaches a word of prose -- so asserting a voice on them would invent a rule
#: the generator never enforced.
_SHAPED_KINDS = frozenset(reward.PROSE_KINDS | {"exam"})


def _rows(path: pathlib.Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def certified() -> list[dict]:
    """Rows the corpus certified, newest evidence first.

    The gold bar is committed and always present. The shard tree is generated
    and gitignored, so it is used when a developer has one and skipped in CI --
    it is a hundredfold more rows, and the three drifts above were each found
    in it rather than in the seven-row bar.
    """
    rows = _rows(GOLD_BAR) if GOLD_BAR.is_file() else []
    if SHARDS.is_dir():
        for path in sorted(SHARDS.glob("*.jsonl")):
            rows.extend(_rows(path))
    if not rows:
        pytest.skip("no certified corpus rows on disk")
    return rows


def test_no_certified_row_is_refused_by_the_register_mirror(certified):
    """A row the generator's register gate passed must pass this one.

    Includes exam rows on purpose: the corpus grades those on the exam
    contract whatever register their pack names, and reading the register
    instead is exactly the drift that put 84 rows on the wrong side.
    """
    refused = [
        (row["id"], row.get("register"), row.get("record_type"), violations)
        for row in certified
        if row.get("record_type") in _SHAPED_KINDS
        and (
            violations := assistant.register_violations(
                str(row.get("register") or ""),
                row.get("answer") or "",
                kind=str(row.get("record_type") or ""),
            )
        )
    ]
    assert not refused, f"{len(refused)} certified rows refused, e.g. {refused[:3]}"


def test_no_certified_row_is_refused_by_the_number_mirror(certified):
    """Every figure a certified answer quotes is one its pack entails.

    Graded against the pack's canonical values *and* its conventions, with the
    pack's own whitelist -- reading canonical alone scored a live answer with
    the right arithmetic 1.000 invented, which is how a gate becomes noise.
    """
    offenders = [
        (row["id"], invented)
        for row in certified
        if row.get("record_type") in _SHAPED_KINDS
        and (allowed := reward.allowed_numbers(row))
        and (
            invented := assistant.invented_numbers(
                row.get("answer") or "", allowed, reward.number_whitelist(row)
            )
        )
    ]
    assert not offenders, (
        f"{len(offenders)} certified rows quote figures this mirror calls "
        f"invented, e.g. {offenders[:3]}"
    )


def test_certified_rows_engage_the_points_their_kind_owes(certified):
    """Coverage, on the contract the record type actually carries.

    An abstention owes the name of what is missing, not the points of the
    question it declined; scoring it against ``must_mention`` marks a correct
    refusal as having engaged nothing.
    """
    thin = []
    for row in certified:
        if row.get("record_type") not in reward.PROSE_KINDS:
            continue
        points = assistant.mention_points(
            row.get("fact_pack") or {}, row.get("record_type") or ""
        )
        if not points:
            continue
        hit, _ = assistant.must_mention_hits(row.get("answer") or "", points)
        if len(hit) / len(points) < 0.8:
            thin.append((row["id"], row.get("record_type"), len(hit), len(points)))
    assert not thin, (
        f"{len(thin)} certified rows score under 0.8 coverage, e.g. {thin[:3]}"
    )


def test_the_certified_corpus_scores_near_the_reward_ceiling(certified):
    """The whole point, stated as one number.

    A reward that does not pay the corpus's own certified prose is a reward
    optimising away from it, and every drift above showed up here first: the
    first cut of this module scored the gold rows 0.29-0.70.
    """
    prose = [r for r in certified if r.get("record_type") in reward.PROSE_KINDS]
    if not prose:
        pytest.skip("no certified prose rows on disk")
    scored = [reward.verifiable_reward(r, r.get("answer") or "") for r in prose]
    gated = [(s.kind, s.gates_tripped) for s in scored if s.gated]
    mean = sum(s.total for s in scored) / len(scored)
    assert not gated, f"{len(gated)} certified prose rows gated, e.g. {gated[:3]}"
    assert mean >= 0.95, f"certified prose scores {mean:.3f}, under the 0.95 floor"
