"""The CLI: exit codes are the contract Airflow schedules against (spec §7).

Every invocation here runs the real ``main`` with real argv in a scratch
``COSIMO_V3_OUT``, so the tests exercise the same argv the DAG cells will --
the flags the spec §7 examples use are asserted to *parse*, not merely to
exist, because an Airflow command line that argparse rejects is a task that
dies at midnight with an error no developer ever heard.
"""

from __future__ import annotations

import os
import re

import pytest

from pipelines.v3 import cli, config, inventory, write
from pipelines.v3.packs import compute_pack


def _outerr(capsys) -> str:
    """stdout+stderr of the captured run as one string (readouterr is a tuple)."""
    captured = capsys.readouterr()
    return captured.out + captured.err


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    out = tmp_path / "corpus"
    monkeypatch.setenv(config.OUT_ENV, str(out))
    return str(out)


def test_inventory_writes_the_plan_the_downstream_stages_consume(corpus):
    assert cli.main(["inventory"]) == cli.EXIT_OK
    plan_path = os.path.join(corpus, "plan.json")
    assert os.path.isfile(plan_path)
    manifest, jobs = inventory.read_plan(plan_path)
    assert manifest["jobs"] == len(jobs) > 0
    assert manifest["schema_version"] == config.SCHEMA_VERSION
    # the on-disk plan is byte-stable: a re-run is a no-op, not a reshuffle
    with open(plan_path, "rb") as handle:
        first = handle.read()
    assert cli.main(["inventory"]) == cli.EXIT_OK
    with open(plan_path, "rb") as handle:
        assert handle.read() == first


def test_inventory_honours_explicit_plan_and_out_paths(tmp_path, corpus):
    plan_file = tmp_path / "plan.json"
    assert (
        cli.main(
            ["inventory", "--plan", config.taxonomy_path(), "--out", str(plan_file)]
        )
        == cli.EXIT_OK
    )
    assert os.path.isfile(str(plan_file))
    assert not os.path.exists(os.path.join(corpus, "plan.json"))


def test_inventory_a_broken_plan_exits_data_not_traceback(tmp_path, corpus):
    bad = tmp_path / "bad.yaml"
    bad.write_text("valuation.equity.dcf: {computer: nope}\n")
    assert cli.main(["inventory", "--plan", str(bad)]) == cli.EXIT_DATA


def test_packs_stage_is_a_replayable_no_op(corpus, capsys):
    import re

    assert cli.main(["inventory", "--smoke"]) == cli.EXIT_OK
    assert cli.main(["packs"]) == cli.EXIT_OK
    assert cli.main(["packs"]) == cli.EXIT_OK
    printed = _outerr(capsys)
    computed = [int(n) for n in re.findall(r"computed (\d+)", printed)]
    already = [int(n) for n in re.findall(r"already there (\d+)", printed)]
    # the smoke plan is 15 renderable packs; the first run computes them and
    # the second computes zero while finding them all already there -- the
    # 3am resume is that second line, printed a hundred times, changing
    # nothing (idempotence is not a feature here, it is the resume contract)
    assert computed == [15, 0], printed
    assert already == [0, 15], printed


def test_packs_without_a_plan_points_at_inventory(corpus, capsys):
    assert cli.main(["packs"]) == cli.EXIT_DATA
    assert "inventory" in _outerr(capsys)


def test_packs_types_filter_and_limit_precede_the_write(corpus):
    assert cli.main(["inventory"]) == cli.EXIT_OK
    assert (
        cli.main(["packs", "--types", "analysis,memo", "--limit", "16"]) == cli.EXIT_OK
    )
    packed = os.path.join(corpus, "fact_packs")
    assert os.path.isdir(packed)
    assert all(name.endswith(".jsonl") for name in os.listdir(packed))
    for name in os.listdir(packed):
        assert len(write.read_jsonl(os.path.join(packed, name))) <= 16


def test_smoke_packs_the_board_and_the_files_both_say_ok(corpus, capsys):
    assert cli.main(["smoke"]) == cli.EXIT_OK
    printed = _outerr(capsys)
    assert "families" in printed and "SMOKE FAIL" not in printed
    smoke_dir = os.path.join(corpus, "_smoke")
    families = set()
    for name in os.listdir(os.path.join(smoke_dir, "fact_packs")):
        for line in write.read_jsonl(os.path.join(smoke_dir, "fact_packs", name)):
            # the pack names its own family through scenario_id; the plan
            # (jobs) and the product (packs) agree on one spelling
            assert line["scenario_id"].startswith(line["work_type"] + ".")
            families.add(line["scenario_id"])
    assert len(families) == 15, "one renderable cell per family"
    # and the smoke run left no .tmp corpses
    for _root, _dirs, files in os.walk(smoke_dir):
        assert not [f for f in files if f.endswith(".tmp")]


def test_smoke_detects_a_pack_lying_about_its_numbers(corpus, capsys):
    from pipelines.v3.verification.invented_numbers import invented_numbers

    assert cli.main(["smoke"]) == cli.EXIT_OK
    target = os.path.join(corpus, "_smoke", "fact_packs", "valuation.equity.dcf.jsonl")
    records = write.read_jsonl(target)
    tampered = None
    for record in records:
        for token in ("999.75", "616.47", "87.31", "455.02"):
            candidate = f"{record['question']} under a {token}% discount"
            if invented_numbers(
                candidate, [float(a) for a in record["allowed_numbers"]]
            ):
                record["question"] = candidate
                tampered = record["id"]
                break
        if tampered is not None:
            break
    assert tampered is not None, (
        "no candidate token escaped the contract -- widen the list"
    )
    write.write_jsonl(target, records)
    # resume must not paper over the lie: append_unique sees the id already
    # committed and leaves the tampered bytes exactly where they are, so the
    # *only* thing standing between a corrupt shard and a published corpus
    # is this reading -- which is why the gate reads back, never forward
    assert cli.main(["smoke"]) == cli.EXIT_DATA
    err = _outerr(capsys)
    assert "SMOKE FAIL" in err
    assert tampered in err and "does not authorise" in err


def test_smoke_fails_loudly_on_a_shattered_computer(corpus, capsys, monkeypatch):
    from pipelines.v3 import packs as packs_registry
    from pipelines.v3.packs import PackError

    def broken(family, variant):
        raise PackError("nothing can be drawn here")

    monkeypatch.setattr(
        packs_registry, "COMPUTERS", {wt: broken for wt in packs_registry.COMPUTERS}
    )
    assert cli.main(["smoke"]) == cli.EXIT_DATA
    assert "SMOKE FAIL" in _outerr(capsys)


def test_the_dag_command_lines_parse_today_exit_tomorrow(corpus, capsys):
    """Spec §7's Airflow cells, verbatim: the whole line now runs for real."""
    lines = [
        ["inventory", "--out", os.path.join(corpus, "plan.json")],
        ["packs", "--plan", os.path.join(corpus, "plan.json")],
        ["render", "--types", "analysis,memo", "--limit", "200"],
        ["verify"],
        ["prefer"],
        ["publish", "--dry-run"],
        ["smoke"],
    ]
    assert cli.main(lines[0]) == cli.EXIT_OK
    assert cli.main(lines[1]) == cli.EXIT_OK
    assert cli.main(lines[6]) == cli.EXIT_OK
    # PR2 wired render and verify, PR4 wired prefer and publish: every one of
    # the spec's lines now runs for real, so none exits 2 to mark a PR ahead of
    # it. Offline render speaks through the committed echo fixture, whose
    # wildcard answer cannot pass the prose gate -- every row is honestly
    # dead-lettered -- so the pair stage draws nobody and publish has nothing to
    # certify: publish is expected to *refuse* this corpus (no rows, no gold-bar
    # fence), which is the gate doing its job, not a stub standing in for it.
    assert cli.main(lines[2]) == cli.EXIT_OK
    assert cli.main(lines[3]) == cli.EXIT_OK
    assert cli.main(lines[4]) == cli.EXIT_OK
    assert cli.main(lines[5]) == cli.EXIT_DATA
    printed = _outerr(capsys)
    assert re.search(
        r"0 rows rendered, 0 already on disk, [1-9]\d* dead-lettered", printed
    ), "with no real teacher in the room, every asked row must die at the gate"
    assert "prefer:" in printed and "pairs written" in printed, (
        "the pair stage must report its pass, not merely exit green"
    )
    assert "PUBLISH REFUSED" in printed, "publish must refuse an empty, unfenced corpus"


def test_an_unknown_command_is_usage_not_a_crash(corpus, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["frobnicate"])
    assert excinfo.value.code == cli.EXIT_USAGE


def test_render_reports_not_recomputes_what_inventory_decided(corpus, monkeypatch):
    """The frozen pair: inventory and smoke must agree on one job list, twice."""
    assert (
        cli.main(["inventory", "--out", os.path.join(corpus, "plan.json")])
        == cli.EXIT_OK
    )
    _, from_disk = inventory.read_plan(os.path.join(corpus, "plan.json"))
    plan = inventory.load_plan()
    assert from_disk == inventory.expand_jobs(plan)
    # and every smoke job the CLI scheduled is recomputable through the
    # registry -- the verify axis 2 rehearsal, in miniature, in CI
    assert cli.main(["smoke"]) == cli.EXIT_OK
    smoke_dir = os.path.join(corpus, "_smoke", "fact_packs")
    for name in os.listdir(smoke_dir):
        for line in write.read_jsonl(os.path.join(smoke_dir, name)):
            family = line["scenario_id"][len(line["work_type"]) + 1 :]
            pack = compute_pack(line["work_type"], family, line["variant"])
            assert pack.to_dict() == {
                k: v for k, v in line.items() if k not in ("id", "verification")
            }
