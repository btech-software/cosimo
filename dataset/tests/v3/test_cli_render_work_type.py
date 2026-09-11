"""``render --work-type`` -- the only way a slice covers more than one voice.

A register is a property of the *family* (``work_types.yaml``'s ``registers:``
table), not of the record type, and the three registers this corpus publishes
are spread across four work types. ``--limit`` applies after the job sort, and
the sort groups by work type -- so ``render --types analysis --limit 20`` takes
the first twenty jobs, all forty of which belong to ``execution.tca.arrival``,
and every row comes back ``desk_chat``.

That is not a hypothetical. Four consecutive live samples came back
single-register and were read as evidence of register collapse; they were
evidence of an unfiltered selector. ``packs`` had this flag from the start
because the DAG fans out on it. ``render`` did not, so no slice could be asked
for a second voice.

The middle test below is the one that matters: it fails on the old CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "fixtures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_prose_fixture as prose_harness  # noqa: E402
from pipelines.v3 import cli, config, inventory, stage, write  # noqa: E402

#: Deep enough to reach three work types in the analysis walk (they arrive in
#: blocks of forty), which is what makes a *register* assertion possible at all.
FIXTURE_DEPTH = 120


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("v3wt"))
    plan = os.path.join(out, "plan.json")
    assert cli.main(["inventory", "--out", plan]) == cli.EXIT_OK
    stage.run_pack_stage(out, inventory.expand_jobs(inventory.load_plan()))
    payload = prose_harness.build_fixture(("analysis",), FIXTURE_DEPTH)
    fixture = os.path.join(out, "prose_fixture.json")
    with open(fixture, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    return out, plan, fixture


@pytest.fixture
def corpus(packed, tmp_path, monkeypatch):
    out = os.path.join(str(tmp_path), "corpus")
    shutil.copytree(packed[0], out)
    monkeypatch.delenv(config.LIVE_ENV, raising=False)
    monkeypatch.setenv(config.TEACHER_FIXTURE_ENV, packed[2])
    monkeypatch.setenv(config.TEACHER_PROSE_ENV, prose_harness.DUMMY_MODEL)
    monkeypatch.setenv(config.TEACHER_REASONING_ENV, prose_harness.DUMMY_MODEL)
    return out, os.path.join(out, "plan.json")


def _render(corpus, work_type: str | None, limit: int) -> int:
    out, plan = corpus
    args = [
        "render",
        "--out",
        out,
        "--plan",
        plan,
        "--types",
        "analysis",
        "--limit",
        str(limit),
    ]
    if work_type is not None:
        args += ["--work-type", work_type]
    return cli.main(args)


def _rows(corpus) -> list[dict]:
    path = write.path_for("sft", "analysis", corpus[0])
    return write.read_jsonl(path) if os.path.isfile(path) else []


#: The three work types the analysis walk reaches inside ``FIXTURE_DEPTH``.
SCOPES = (
    "execution.tca.arrival",
    "portfolio.attribution.brinson_carino",
    "risk.market.var_es",
)


def test_a_scoped_render_writes_only_the_requested_work_type(corpus):
    assert _render(corpus, SCOPES[1], 3) == cli.EXIT_OK
    rows = _rows(corpus)
    assert rows, "a scoped render must still render something"
    assert {r["work_type"] for r in rows} == {SCOPES[1]}


def test_scoped_renders_reach_registers_one_limited_render_cannot(corpus):
    """The failure the flag exists to fix, as an assertion on the same budget.

    Nine rows taken in one ``--limit 9`` pass are nine rows of one work type
    and therefore one voice. The same nine, asked for by scope, span three.
    """
    assert _render(corpus, None, 9) == cli.EXIT_OK
    unscoped = {r.get("register") for r in _rows(corpus)}
    assert len(unscoped) == 1, (
        "the premise of this test is that an unscoped limited render is "
        f"single-register; it returned {unscoped}"
    )

    for work_type in SCOPES:
        assert _render(corpus, work_type, 3) == cli.EXIT_OK
    by_scope = {}
    for row in _rows(corpus):
        by_scope.setdefault(row["work_type"], set()).add(row.get("register"))

    assert set(by_scope) >= set(SCOPES)
    reached = {r for regs in by_scope.values() for r in regs if r}
    assert len(reached) >= 2, (
        f"scoped renders still reached only {reached}; the point of the flag "
        "is that a slice can hold more than one voice"
    )


def test_an_unknown_work_type_is_a_usage_error_naming_what_the_plan_has(corpus, capsys):
    assert _render(corpus, "no.such.work", 3) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "no.such.work" in err
    assert SCOPES[0] in err, "an error that does not say what is valid teaches nothing"
