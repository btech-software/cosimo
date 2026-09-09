"""``packs --work-type`` -- the scoping seam the DAG's mapped packs wave leans on.

``packs`` fans out one task per work type, so each task must be able to compute
just its own ``fact_packs/<work_type>.jsonl`` and leave every other shard
untouched -- a shard has exactly one writer, or concurrent appends clobber (AGENTS
§5.3). And the whole must still equal the sum of its parts: fanning out is a
scheduling optimisation, not a change to what gets produced. These tests pin
exactly that -- the disjointness, the union-equals-whole equivalence, and that a
single-work-type run does not trip the coverage check with the work types it was
never asked to cover (that check is scoped to the jobs handed to it).
"""

from __future__ import annotations

import os

import pytest

from pipelines.v3 import cli, config, inventory, write


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    out = tmp_path / "corpus"
    monkeypatch.setenv(config.OUT_ENV, str(out))
    return str(out)


def _shards(out: str) -> list[str]:
    d = os.path.join(out, "fact_packs")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _ids(out: str, work_type: str) -> set[str]:
    return set(write.existing_ids(write.path_for("fact_packs", work_type, out)))


def _planned_work_types(plan_path: str) -> list[str]:
    return sorted({job.work_type for job in inventory.read_plan(plan_path)[1]})


def test_a_single_work_type_run_writes_only_its_own_shard(corpus):
    assert cli.main(["inventory"]) == cli.EXIT_OK
    _, jobs = inventory.read_plan(os.path.join(corpus, "plan.json"))
    work_types = sorted({job.work_type for job in jobs})
    assert len(work_types) >= 2, "need a multi-work-type plan to prove disjointness"

    chosen = work_types[0]
    assert cli.main(["packs", "--work-type", chosen]) == cli.EXIT_OK

    assert _shards(corpus) == [f"{chosen}.jsonl"], (
        "a per-work-type task must create no shard but its own -- the whole "
        "point of mapping packs without a clobber-prone shared write"
    )
    assert _ids(corpus, chosen), "the one shard it owns must actually hold packs"


def test_fanning_out_equals_one_whole_corpus_run(tmp_path):
    plan = str(tmp_path / "plan.json")
    assert cli.main(["inventory", "--out", plan]) == cli.EXIT_OK
    work_types = _planned_work_types(plan)

    whole = str(tmp_path / "whole")
    assert cli.main(["packs", "--plan", plan, "--out", whole]) == cli.EXIT_OK

    sliced = str(tmp_path / "sliced")
    for work_type in work_types:
        assert (
            cli.main(
                ["packs", "--plan", plan, "--out", sliced, "--work-type", work_type]
            )
            == cli.EXIT_OK
        ), f"{work_type}: a scoped run must not fail the coverage check"

    assert _shards(whole) == _shards(sliced), (
        "the set of shards must be identical however the plan is partitioned"
    )
    for work_type in work_types:
        assert _ids(whole, work_type) == _ids(sliced, work_type), work_type
    assert _ids(whole, work_types[0]) == _ids(sliced, work_types[0])


def test_the_filter_combines_with_limit_and_stays_scoped(tmp_path):
    plan = str(tmp_path / "plan.json")
    assert cli.main(["inventory", "--out", plan]) == cli.EXIT_OK
    work_type = _planned_work_types(plan)[0]

    out = str(tmp_path / "limited")
    assert (
        cli.main(
            [
                "packs",
                "--plan",
                plan,
                "--out",
                out,
                "--work-type",
                work_type,
                "--limit",
                "1",
            ]
        )
        == cli.EXIT_OK
    )
    # limit is a head-truncation of the (already work-type-scoped) job list; it
    # may thin the shard but must not reach into a work type the task did not ask
    # for, and an unmet family must not make a scoped run cry about its siblings.
    assert _shards(out) == [f"{work_type}.jsonl"]


def test_an_unknown_work_type_is_a_usage_error_not_a_silent_empty(corpus, capsys):
    assert cli.main(["inventory"]) == cli.EXIT_OK
    assert cli.main(["packs", "--work-type", "no.such.work.type"]) == cli.EXIT_USAGE, (
        "a typo the operator would otherwise read as 'nothing to do' must be exit 2"
    )
    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert "unknown --work-type" in printed
    assert "no.such.work.type" in printed
    assert _shards(corpus) == [], "a rejected invocation must write nothing at all"
