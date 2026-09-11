"""``inventory``: the deterministic, capped, resumable expansion of the plan.

The four properties the corpus's whole lifecycle depends on, tested at their
source: order independence (two runs, one file), idempotence (re-expanding
repeats nothing, resume skips what the disk already holds), the family cap
(bitten *before* the teacher is billed), and validation that fails at load
rather than twelve million jobs in.
"""

from __future__ import annotations

import json
import os

import pytest
import yaml

from pipelines.v3 import config, inventory, stage, write
from pipelines.v3.packs import compute_pack

PLAN_PATH = os.path.join(config.BASE_DIR, "taxonomy", "work_types.yaml")


def _synthetic_plan():
    return {
        "valuation.equity.dcf": {
            "computer": "valuation_fcff",
            "families": {
                "fat": {"holdout": False},
                "thin_a": {"holdout": False},
                "thin_b": {"holdout": False},
                "holdout_c": {"holdout": True},
            },
            "record_types": ["exam"],
            "variants_per_family": {"exam": 500},
            "max_share": 0.03,
            "pitfalls": [],
        }
    }


def _plan_path(tmp_path, plan):
    path = os.path.join(str(tmp_path), "work_types.yaml")
    with open(path, "w", encoding="utf8") as handle:
        yaml.safe_dump(plan, handle, sort_keys=False)
    return path


def test_the_committed_plan_loads_with_every_key_the_code_demands():
    plan = inventory.load_plan(PLAN_PATH)
    assert len(plan) == 5
    for work_type, spec in plan.items():
        assert spec["computer"]
        assert spec["families"], work_type
        assert any(meta["holdout"] for meta in spec["families"].values()), (
            f"{work_type}: a plan without a holdout family cannot measure "
            "unseen_scenario_family and is not a v3 plan"
        )
        assert spec["record_types"]
        assert all(v >= 1 for v in spec["variants_per_family"].values())
        assert 0 < spec["max_share"] <= 1


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda p: p["valuation.equity.dcf"].pop("max_share"), "missing keys"),
        (
            lambda p: p.__setitem__(
                "valuation.equities.dcf", p.pop("valuation.equity.dcf")
            ),
            "no computer registered",
        ),
        (
            lambda p: p["valuation.equity.dcf"].__setitem__(
                "computer", "valuation_multiples"
            ),
            "registry resolves it",
        ),
        (
            lambda p: p["valuation.equity.dcf"]["families"].__setitem__(
                "mature_consumer", {"holdout": "yes"}
            ),
            "boolean",
        ),
        (
            lambda p: p["valuation.equity.dcf"].__setitem__(
                "variants_per_family", {"exam": 0}
            ),
            "int >= 1",
        ),
        (
            lambda p: p["valuation.equity.dcf"].__setitem__("max_share", 1.5),
            "max_share must be",
        ),
        (
            lambda p: p["valuation.equity.dcf"].__setitem__(
                "record_types", ["exam", "analysis"]
            ),
            "lacks",
        ),
        (lambda p: p["valuation.equity.dcf"].__setitem__("families", {}), "non-empty"),
    ],
)
def test_a_malformed_plan_dies_at_load_not_twelve_million_jobs_in(
    tmp_path, mutate, needle
):
    plan = _synthetic_plan()
    mutate(plan)
    path = _plan_path(tmp_path, plan)
    with pytest.raises(inventory.PlanError, match=needle):
        inventory.load_plan(path)


def test_a_missing_or_unparsable_plan_is_named_not_invented(tmp_path):
    with pytest.raises(inventory.PlanError, match="not found"):
        inventory.load_plan(os.path.join(str(tmp_path), "nowhere.yaml"))
    bad = os.path.join(str(tmp_path), "bad.yaml")
    with open(bad, "w", encoding="utf8") as handle:
        handle.write("just: [a, string, list, no")
    with pytest.raises(inventory.PlanError, match="not valid yaml"):
        inventory.load_plan(bad)


def test_expansion_is_the_same_file_twice_rained():
    plan = inventory.load_plan(PLAN_PATH)
    first = [job.as_record() for job in inventory.expand_jobs(plan)]
    second = [job.as_record() for job in inventory.expand_jobs(plan)]
    assert first == second
    assert first == sorted(
        first,
        key=lambda row: (
            row["work_type"],
            [str(f) for f in plan[row["work_type"]]["families"]].index(row["family"]),
            plan[row["work_type"]]["record_types"].index(row["record_type"]),
            row["variant"],
        ),
    )


def test_the_cap_truncates_every_overreaching_family_never_the_holdout():
    # The plan's schema gives every family of a work type the same budget, so
    # a 3% cap with ten train families (the shipped plan: 372 demanded each,
    # 3720 planned) necessarily bites every family equally -- down to
    # floor(0.03*3720)=111 -- while the holdout, which trains nothing, rides
    # uncapped. That equality is the point: the cap measures share of the
    # *pool*, and a run that reads it as "trim the fat and let the rest flow"
    # has misread which denominator the 3% story counts.
    jobs = inventory.expand_jobs(_synthetic_plan())
    rows: dict[str, int] = {}
    for job in jobs:
        rows[job.family] = rows.get(job.family, 0) + 1
    planned = 500 * 3
    plan = _synthetic_plan()
    # The effective share is the tighter of the plan's own ceiling and the
    # derived anti-dominance cap -- three train families make the derived one
    # 1.25/3, so the plan's 0.03 still binds and the arithmetic is unchanged.
    effective = min(
        plan["valuation.equity.dcf"]["max_share"],
        config.family_max_share(inventory.train_family_count(plan)),
    )
    cap = int(effective * planned)  # floor(45.0) == 45
    assert cap == 45
    assert rows["fat"] == cap
    assert rows["thin_a"] == cap
    assert rows["thin_b"] == cap
    assert rows["holdout_c"] == 500  # overreaches wildly, and nothing forbids it
    manifest = inventory.plan_manifest(jobs, _synthetic_plan(), smoke=False)
    for cell in manifest["families"].values():
        assert cell["cap"] == cap
        assert cell["rows"] <= cell["cap"]


def test_holdouts_are_never_capped_and_never_packed_into_the_denominator():
    plan = _synthetic_plan()
    plan["valuation.equity.dcf"]["families"]["holdout_c"] = {"holdout": True}
    jobs = inventory.expand_jobs(plan)
    holdouts = [j for j in jobs if j.holdout]
    assert all(j.family == "holdout_c" for j in holdouts)
    assert len(holdouts) == 500  # uncapped
    supervised = [j for j in jobs if not j.holdout]
    manifest = inventory.plan_manifest(jobs, plan, smoke=False)
    assert manifest["supervised_rows"] == len(supervised)
    assert manifest["eval_rows"] == len(holdouts)


def test_the_real_plan_honours_its_own_caps_under_the_emitted_lens():
    plan = inventory.load_plan(PLAN_PATH)
    jobs = inventory.expand_jobs(plan)
    manifest = inventory.plan_manifest(jobs, plan, smoke=False)
    assert manifest["jobs"] == len(jobs)
    for _name, cell in manifest["families"].items():
        assert cell["rows"] <= cell["cap"]
        assert cell["holdout"] is False
        assert abs(cell["share"] - cell["rows"] / manifest["supervised_rows"]) <= 1e-6


def test_the_real_plan_authors_the_exam_slice_into_its_band():
    """The plan must *author* the exam slice into [0.12, 0.18], not hope for it.

    ``verify`` measures the exam share on the shipped rows (axis 8), where the
    preference stage's pairs inflate the denominator -- so the board is a
    downstream check, not the guarantee. The guarantee is authored here, in the
    one file that decides how much of anything exists: a plan whose exam slice
    already sits above the band can only reach it by luck of how many pairs
    happen to land, which is the exam-overweight style-collapse pathology the
    whole v2->v3 change exists to prevent, left one silent number away from
    returning.
    """
    jobs = inventory.expand_jobs(inventory.load_plan(PLAN_PATH))
    total = len(jobs)
    exam = sum(1 for job in jobs if job.record_type == "exam")
    share = exam / total
    low, high = config.EXAM_SHARE_BAND
    assert low <= share <= high, (
        f"exam slice is {exam}/{total} = {share:.3f}, outside the "
        f"[{low}, {high}] band the acceptance gate certifies"
    )


def test_smoke_collapses_to_one_renderable_variant_per_family():
    plan = inventory.load_plan(PLAN_PATH)
    jobs = inventory.expand_jobs(plan, smoke=True)
    seen = {}
    for job in jobs:
        key = (job.work_type, job.family)
        seen.setdefault(key, set()).add(job.variant)
    assert all(len(variants) == 1 for variants in seen.values())
    assert len(seen) == 15, "five work types, three families each"
    for (work_type, family), variants in seen.items():
        (variant,) = variants
        compute_pack(work_type, family, variant)  # smoke may not schedule a broken cell


def test_plan_files_round_trip_through_their_own_readers(tmp_path):
    plan = inventory.load_plan(PLAN_PATH)
    jobs = inventory.expand_jobs(plan, smoke=True)
    manifest = inventory.plan_manifest(jobs, plan, smoke=True)
    out = os.path.join(str(tmp_path), "plan.json")
    inventory.write_plan(out, manifest, jobs)
    read_manifest, read_jobs = inventory.read_plan(out)
    assert read_jobs == jobs
    assert read_manifest == manifest
    with open(out, encoding="utf8") as handle:
        payload = json.load(handle)
    assert list(payload) == ["jobs", "manifest"]


def test_resume_consults_the_disk_before_the_teacher(tmp_path):
    """The 3am story: half the packs are on disk; the stage must add only the rest."""
    out = str(tmp_path)
    jobs = inventory.expand_jobs(inventory.load_plan(PLAN_PATH), smoke=True)
    report_a = stage.run_pack_stage(out, jobs)
    assert report_a["computed"] > 0
    path = write.path_for("fact_packs", "valuation.equity.dcf", out)
    before = write.existing_ids(path)
    report_b = stage.run_pack_stage(out, jobs)
    assert report_b["computed"] == 0
    assert report_b["existing"] == report_a["computed"]
    assert write.existing_ids(path) == before
