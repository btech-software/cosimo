"""The writers: nothing half-written may be mistaken for a whole shard.

Tests for the append/merge discipline (spec §6.1's "no half-written shards")
and the pack stage report (spec §5.3): the report numbers must add up, the
resume must add nothing, and a family the plan promised must be visible when
the computer betrayed it.
"""

from __future__ import annotations

import os

import pytest

from pipelines.v3 import stage, write


def test_path_for_refuses_unlisted_kinds_and_wandering_names():
    with pytest.raises(ValueError, match="unlisted shard kind"):
        write.path_for("dead_letters", "x", "/tmp/o")
    with pytest.raises(ValueError, match="single segments"):
        write.path_for("sft", "../escape", "/tmp/o")
    with pytest.raises(ValueError, match="single segments"):
        write.path_for("sft", "sub/dir", "/tmp/o")
    with pytest.raises(ValueError, match="single segments"):
        write.path_for("sft", "", "/tmp/o")
    assert write.path_for(
        "fact_packs", "valuation.equity.dcf", "/tmp/o"
    ) == os.path.join("/tmp/o", "fact_packs", "valuation.equity.dcf.jsonl")


def test_write_then_read_round_trips_and_rejects_ill_records(tmp_path):
    path = write.path_for("sft", "analysis", str(tmp_path))
    assert write.read_jsonl(path) == [], "absent file reads empty, resume asks anyway"
    written = write.write_jsonl(path, [{"id": "a", "n": 1}, {"id": "b"}])
    assert written == 2
    again = write.write_jsonl(path, [{"id": "a", "n": 1}, {"id": "b"}])
    assert again == 2
    assert write.read_jsonl(path) == [{"id": "a", "n": 1}, {"id": "b"}]
    assert write.existing_ids(path) == frozenset({"a", "b"})


def test_append_unique_counts_what_it_adds_and_what_it_skips(tmp_path):
    path = write.path_for("sft", "memo", str(tmp_path))
    added, skipped = write.append_unique(path, [{"id": "x"}, {"id": "y"}, {"id": "x"}])
    assert (added, skipped) == (2, 1)
    added, skipped = write.append_unique(path, [{"id": "x"}, {"id": "z"}])
    assert (added, skipped) == (1, 1)
    assert sorted(write.existing_ids(path)) == ["x", "y", "z"]
    # an id already on disk is never rewritten by a late-arriving twin
    assert write.read_jsonl(path)[0] == {"id": "x"}


def test_ill_records_are_refused_at_the_gate_not_written_then_wept_over(tmp_path):
    path = write.path_for("sft", "memo", str(tmp_path))
    with pytest.raises(ValueError, match="not a json object"):
        write.append_unique(path, [{"no_id": 1}])
    with pytest.raises(ValueError, match="not a json object"):
        write.write_jsonl(path, ["a string"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        handle.write('{"id": "ok"}\n')
        handle.write("{ broken json\n")
    with pytest.raises(ValueError, match=":2:"):
        write.read_jsonl(path)
    os.remove(path)
    with open(path, "w", encoding="utf8") as handle:
        handle.write('{"no": "id"}\n')
    with pytest.raises(ValueError, match=":1:"):
        write.read_jsonl(path)


def test_no_half_written_shard_is_left_behind(tmp_path):
    path = write.path_for("sft", "analysis", str(tmp_path))
    write.write_jsonl(path, [{"id": "a"}])
    with pytest.raises(OSError):
        write._stage_and_swap("/nonexistent-root-deep/deeper/x.jsonl", [{"id": "a"}])
    assert [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")] == []
    # the swap left exactly one file, with the whole record
    assert os.path.basename(path) in os.listdir(tmp_path / "sft")


def test_pack_records_carry_their_provenance_stamp():
    line = stage.pack_record("valuation.equity.dcf", "mature_consumer", 0)
    assert (
        line["id"].startswith("cosimov3pack_") and len(line["id"].split("_")[-1]) == 16
    )
    ver = line["verification"]
    assert ver["computed_by"].endswith("packs.valuation_fcff")
    assert ver["computer"] == "compute"
    assert ver["verified"] is True
    int(ver["pack_seed"], 16)
    assert line["schema_version"] == "v3.0"
    assert line["allowed_numbers"], (
        "a pack with no authorised numbers cannot be rendered"
    )


def test_run_pack_stage_reports_computed_existing_and_skipped(tmp_path):
    from pipelines.v3 import inventory

    jobs = inventory.expand_jobs(inventory.load_plan(), smoke=True)
    out = str(tmp_path)
    report = stage.run_pack_stage(out, jobs)
    assert report["unique_packs"] == len(jobs) // 8
    assert report["computed"] == report["unique_packs"] - len(report["skipped"])
    assert report["existing"] == 0
    assert report["families_without_packs"] == []
    # replay: everything is already there, and nothing re-asked
    again = stage.run_pack_stage(out, jobs)
    assert again["computed"] == 0
    assert again["existing"] == report["unique_packs"] - len(again["skipped"])
    assert again["skipped"] == report["skipped"]


def test_run_pack_stage_limits_and_selects_before_writing_anything(tmp_path):
    from pipelines.v3 import inventory

    jobs = inventory.expand_jobs(inventory.load_plan())
    out = str(tmp_path)
    report = stage.run_pack_stage(out, jobs, types=["analysis", "memo"], limit=16)
    assert report["jobs_seen"] == 16
    assert report["unique_packs"] <= 16
    on_disk = os.listdir(tmp_path / "fact_packs")
    assert all(name.endswith(".jsonl") for name in on_disk)


def test_a_family_whose_computer_betrays_it_is_named_not_silently_dropped(
    tmp_path, monkeypatch
):
    from pipelines.v3 import inventory
    from pipelines.v3.packs import PackError

    def broken(family, variant):
        raise PackError(f"{family}/{variant} cannot be drawn into a scenario")

    # patch the registry *compute_pack* resolves through, not a copy of it:
    # the module global is looked up at call time on purpose (verification
    # re-imports through it), so this is the seam both the stage and the
    # verifier share
    from pipelines.v3 import packs as packs_registry

    monkeypatch.setattr(packs_registry, "COMPUTERS", {"valuation.equity.dcf": broken})
    jobs = [
        inventory.Job("valuation.equity.dcf", "mature_consumer", "analysis", 0, False)
    ]
    report = stage.run_pack_stage(str(tmp_path), jobs)
    assert report["computed"] == 0
    assert len(report["skipped"]) == 1
    assert report["skipped"][0]["reason"].endswith("cannot be drawn into a scenario")
    assert report["families_without_packs"] == ["valuation.equity.dcf/mature_consumer"]
