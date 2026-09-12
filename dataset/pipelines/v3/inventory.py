"""The plan, not "run every template 1000 times" (spec §5.1).

``work_types.yaml`` is law: which families exist, which of them are holdout,
how many variants each record type gets, and what fraction of the supervised
pool any single family may take. This module turns that file into the
deterministic job list ``[(work_type, family, record_type, variant)]`` the
stages consume, and it enforces the family cap **here, at planning time** --
a family that would exceed its share is not generated, rather than generated
and filtered later when the token budget has already been spent on it.

Two properties make the corpus resumable, and both are properties of this
module rather than of the stages that call it:

1. *order independence*: the expansion is a pure function of the yaml file
   -- sorted work types, file-order families, file-order record types, variant
   ascending -- so two runs (or a Spark executor and a laptop) build the same
   list, and ``render_seed`` keys off the tuple, never off position or clock;
2. *idempotence*: every stage checks its own namespace with
   :func:`write.existing_ids` before asking the teacher for anything, so a
   re-run appends the missing tail and touches nothing else -- the resume
   story the operator needs after an API-key expiry at 3am mid-80k-calls.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(_HERE))
for _p in (_DATASET, os.path.dirname(_DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import asdict, dataclass  # noqa: E402
import yaml  # noqa: E402  -- corpus group (spec §9)

from . import config  # noqa: E402
from .packs import COMPUTERS, PackError  # noqa: E402

REQUIRED_KEYS = (
    "computer",
    "families",
    "record_types",
    "variants_per_family",
    "max_share",
    "pitfalls",
)


#: The registers a ``memo`` may be written in (amendment §C).
#:
#: A memo is a document with headings and a labelled call; desk chat is neither,
#: and the gate refuses both moves for that register. The pair was legal in the
#: plan anyway, so ``execution.tca.arrival`` -- desk-only in all three families
#: -- was asked for forty memos a run, and each one was a row that could satisfy
#: its kind or its register and not both. ``desk_chat_ceiling`` exists entirely
#: to keep those rows alive, and a live sample shows what survived: a memo in
#: desk voice with the headings filed off.
#:
#: One pack serves every record type of a variant, so the register is drawn
#: before the record type is known. That is why the rule is a *plan* rule and
#: not a render-time one: a family that may speak desk chat may not also be
#: asked for a memo, because nothing downstream gets to choose.
MEMO_REGISTERS = frozenset({"ic_memo", "risk_committee"})


class PlanError(ValueError):
    """The yaml plan is malformed or contradicts the code. Fail at load."""


def illegal_triple(spec: dict, family: str, record_type: str) -> str:
    """Why this ``(family, record_type)`` may not be emitted, or ``""``.

    Public and shared: :func:`_validate_work` refuses a plan with it and the
    inventory test walks :func:`expand_jobs` through it, so "illegal pair" has
    one definition rather than one in the loader and one in the assertion.
    """
    if record_type != "memo":
        return ""
    declared = tuple((spec.get("registers") or {}).get(family) or ())
    if not declared:
        return (
            f"family {family!r} emits 'memo' but declares no registers; a memo "
            f"needs a family whose register is one of {sorted(MEMO_REGISTERS)}"
        )
    wrong = [r for r in declared if r not in MEMO_REGISTERS]
    if wrong:
        return (
            f"family {family!r} may be written in {', '.join(map(repr, wrong))} "
            f"and also emits 'memo'; a memo needs a family whose registers are "
            f"all in {sorted(MEMO_REGISTERS)} -- one pack serves every record "
            "type of a variant, so the register is drawn before the record "
            "type is known"
        )
    return ""


@dataclass(frozen=True)
class Job:
    """One unit of generation work: exactly what ``render_seed`` keys off."""

    work_type: str
    family: str
    record_type: str
    variant: int
    holdout: bool

    def as_record(self) -> dict:
        return asdict(self)


def load_plan(path: str | None = None) -> dict:
    """Parse and *validate* the work-type plan. Structural errors raise here.

    Loading, not reading: every inconsistency between the file and the code it
    drives (an unregistered computer, a work type with no families, a zero
    variant count) is rejected at this edge, because the alternative is an
    ``inventory`` run that dies twelve million jobs in, mid-teacher-billing.
    """
    path = os.path.abspath(path or config.taxonomy_path())
    try:
        with open(path, encoding="utf8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise PlanError(f"work-type plan not found at {path!r}") from exc
    except yaml.YAMLError as exc:
        raise PlanError(f"work-type plan {path!r} is not valid yaml: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise PlanError(f"work-type plan {path!r} must be a non-empty mapping")
    for work_type, spec in raw.items():
        _validate_work(work_type, spec, path)
    return raw


def _validate_work(work_type: str, spec: object, path: str) -> None:
    where = f"{path}: work type {work_type!r}"
    if not isinstance(spec, dict):
        raise PlanError(f"{where}: entry must be a mapping")
    missing = [key for key in REQUIRED_KEYS if key not in spec]
    if missing:
        raise PlanError(f"{where}: missing keys {', '.join(missing)}")
    # COMPUTERS is keyed by work type -- the same key the verifier recomputes
    # through (spec §5.3). The yaml's `computer` names the *module* that
    # answers that key, so the plan can never quietly point a work type at a
    # different computer than the code registry does: two sources, one truth,
    # checked here rather than at the twelve-millionth job.
    if work_type not in COMPUTERS:
        raise PlanError(
            f"{where}: no computer registered for it "
            f"(known: {', '.join(sorted(COMPUTERS))})"
        )
    module = getattr(COMPUTERS[work_type], "__module__", "")
    if not module.endswith("." + spec["computer"]):
        raise PlanError(
            f"{where}: plan names computer {spec['computer']!r} but the "
            f"registry resolves it to {module.split('.')[-1]!r}"
        )
    families = spec["families"]
    if not isinstance(families, dict) or not families:
        raise PlanError(f"{where}: families must be a non-empty mapping")
    for family, meta in families.items():
        if not isinstance(meta, dict) or not isinstance(meta.get("holdout"), bool):
            raise PlanError(f"{where}, family {family!r}: needs a boolean 'holdout'")
    record_types = spec["record_types"]
    variants = spec["variants_per_family"]
    if not isinstance(record_types, list) or not record_types:
        raise PlanError(f"{where}: record_types must be a non-empty list")
    if not isinstance(variants, dict):
        raise PlanError(f"{where}: variants_per_family must be a mapping")
    unknown = [rt for rt in record_types if rt not in variants]
    if unknown:
        raise PlanError(f"{where}: variants_per_family lacks {', '.join(unknown)}")
    for record_type, count in variants.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise PlanError(
                f"{where}: variants_per_family[{record_type!r}] must be an int >= 1"
            )
    registers = spec.get("registers")
    if registers is not None and not isinstance(registers, dict):
        raise PlanError(f"{where}: registers must be a mapping of family to list")
    for family in families:
        for record_type in record_types:
            problem = illegal_triple(spec, family, record_type)
            if problem:
                raise PlanError(f"{where}: {problem}")
    share = spec["max_share"]
    if (
        isinstance(share, bool)
        or not isinstance(share, (int, float))
        or not 0 < share <= 1
    ):
        raise PlanError(f"{where}: max_share must be in (0, 1]")


def train_family_count(plan: dict) -> int:
    """How many *train* families this plan describes.

    The denominator of the family cap (:func:`config.family_max_share`). Read
    off the plan rather than pinned in the control plane, because the cap's
    whole claim -- "no family runs more than a quarter ahead of an even share"
    -- is a statement about the plan that is loaded, and a constant could only
    ever be right for one of them. Holdout families are excluded for the same
    reason the cap measures the supervised pool: they do not compete for
    training rows.
    """
    return sum(
        1
        for spec in plan.values()
        for meta in spec["families"].values()
        if not meta["holdout"]
    )


def expand_jobs(plan: dict, *, smoke: bool = False) -> list[Job]:
    """The deterministic job list for *plan*, family-capped before anything runs.

    ``smoke=True`` (the CI inventory, spec §7) collapses every family to its
    variant ``0``: one draw per family tests the plumbing without spending a
    teacher budget, and the holdout families ride along -- the smoke output is
    *also* used for eval, and an eval slice must exercise the same code paths
    the train generator will take (see §7 of the analysis spec: the 18% story
    was precisely a train generator evaluated on data its generator had never
    produced, because holdout was a flag nobody checked and an inventory that
    did not expand them).

    The cap is per ``(work_type, family)`` measured as that family's share of
    the *planned supervised pool*: planned, because the gate must bite before
    any teacher call is paid for; supervised, because eval rows do not crowd
    the training distribution and must not be shrunk by a training-diversity
    rule. Two numbers therefore exist and the manifest keeps both: the
    planned share the cap enforces (spec §5.1's literal contract, "refuses to
    emit more than X% of planned SFT") and the realised share of what is
    actually emitted -- truncating the fat families shrinks the denominator,
    so realised shares sit above the cap whenever families are fewer than the
    cap invites (they sum to 1/100th each only for >= 1/max_share families;
    with ten train families they cannot sum under 3% no matter the variant
    counts). The cap's real work is anti-dominance: an easy family planned at
    half the pool is cut to the budget while the rest flow untouched. Which
    families a run has is a coverage decision for the lead, and axis 5 of
    verify_v3 measures the emitted shares, not this promise.
    """
    raw = [
        Job(work_type, family, record_type, variant, holdout=bool(meta["holdout"]))
        for work_type in sorted(plan)
        for family, meta in plan[work_type]["families"].items()
        for record_type in plan[work_type]["record_types"]
        for variant in range(
            1 if smoke else plan[work_type]["variants_per_family"][record_type]
        )
    ]
    if smoke:
        # The CI inventory is a plumbing check, not a training distribution:
        # one *successful* variant per family, every family (holdout included),
        # cap waived -- applying the diversity cap here would let it silently
        # delete the very cells -- the thin tails -- the smoke run exists to
        # prove can carry data. "Successful" matters: a computer may
        # legitimately reject a drawn parameter set (PackError), and a smoke
        # board showing "every family renders" must not be satisfied by a
        # family whose one reserved variant happened to be one of those.
        # Variant 0 is the default; if it is rejected, walk upward through
        # the budget the plan reserves for the family -- still a pure function
        # of (work_type, family, variant), so the chosen index is reproducible
        # and stays inside the reserved set the full run will use.
        out: list[Job] = []
        for work_type in sorted(plan):
            spec = plan[work_type]
            bound = max(spec["variants_per_family"][rt] for rt in spec["record_types"])
            for family, meta in spec["families"].items():
                variant = _first_renderable(work_type, family, bound)
                for record_type in spec["record_types"]:
                    out.append(
                        Job(
                            work_type,
                            family,
                            record_type,
                            variant,
                            holdout=bool(meta["holdout"]),
                        )
                    )
        return out
    planned_supervised = sum(not job.holdout for job in raw)
    kept = _family_cap(plan, raw, planned_supervised)
    return kept


def _family_cap(plan: dict, raw: list[Job], planned_supervised: int) -> list[Job]:
    """Enforce the per-family share, preserving each family's planned *shape*.

    Largest-remainder (Hamilton) apportionment, in integer arithmetic: the
    family's ``cap`` seats are divided across its record-type cells in
    proportion to the planned counts, floors first, leftover seats to the
    largest remainders, ties broken by the file's record-type order. Then a
    cell keeps the *prefix* of its reserved variants (``0..keep-1``), which
    is what ``pack_seed`` and the smoke walk already treat as the family's
    reserved budget.

    Proportional, and never head-truncation-by-enumeration-order, because
    head-truncation is a silent re-authoring of the corpus: a family planned
    at 80 exam / 120 analysis / 40 memo / ... cut at 111 in list order keeps
    all the exam, some analysis, and *zero* of every later type -- a mix the
    author never signed (exam alone at ~72% of the row population, precisely
    the v2 pathology §2.3/§12 exists to prevent), and one the manifest would
    print without ever distinguishing it from the plan. The cap's promise is
    about a family's *share of the pool*; this keeps that promise (cells sum
    to <= cap) and adds the one the mix table needs: the family's own shape
    survives the cut, deterministically, as a pure function of the yaml.
    """
    planned_by_cell: dict[tuple[str, str, str], int] = {}
    cell_order: dict[tuple[str, str], list[str]] = {}
    for job in raw:
        if job.holdout:
            continue
        cell = (job.work_type, job.family, job.record_type)
        planned_by_cell[cell] = planned_by_cell.get(cell, 0) + 1
        cells = cell_order.setdefault((job.work_type, job.family), [])
        if job.record_type not in cells:
            cells.append(job.record_type)

    derived_cap = config.family_max_share(train_family_count(plan))
    keep: dict[tuple[str, str, str], int] = {}
    for (work_type, family), cells in cell_order.items():
        counts = [planned_by_cell[(work_type, family, cell)] for cell in cells]
        total = sum(counts)
        share = min(plan[work_type]["max_share"], derived_cap)
        cap = min(total, _floor(share * planned_supervised))
        base = [cap * count // total for count in counts]
        remainders = [cap * count % total for count in counts]
        awarded = cap - sum(base)
        # largest remainder first; ties fall to the file's record-type order
        ranking = sorted(range(len(cells)), key=lambda i: (-remainders[i], i))
        for i in ranking[:awarded]:
            base[i] += 1
        for cell, kept_count in zip(cells, base):
            keep[(work_type, family, cell)] = min(
                planned_by_cell[(work_type, family, cell)], kept_count
            )
    kept: list[Job] = []
    for job in raw:
        if job.holdout:
            kept.append(job)
            continue
        if job.variant < keep[(job.work_type, job.family, job.record_type)]:
            kept.append(job)
    return kept


def _floor(share: float) -> int:
    """``int()`` on purpose: truncate the cap, never round it up.

    A family gets ``floor(share * N)`` slots, not the nearest integer -- the
    3% story is remembered as what the cap held *under*, and a generator
    whose caps are rounded upward holds a different distribution entirely.
    """
    return int(share)


def _first_renderable(work_type: str, family: str, bound: int) -> int:
    """Smallest variant in ``[0, bound)`` whose pack computes; ``0`` if none.

    Pure: ``compute_pack`` is the same function verification recomputes
    through, so this walk is reproducible to the byte and needs no memo of
    what happened to be on disk. When every reserved variant of a family is
    rejected by its computer the family is an authoring bug, not a smoke
    detail -- the stage records a skip and the inventory says so, but smoke
    may not silently report a family with no pack at all.
    """
    for variant in range(max(bound, 1)):
        try:
            COMPUTERS[work_type](family, variant)
        except PackError:
            continue
        return variant
    return 0


def plan_manifest(jobs: list[Job], plan: dict, *, smoke: bool) -> dict:
    """The JSON audit trail of one expansion: counts, shares, and the inputs.

    Written beside the plan file by ``cli inventory``; ``verify_v3`` axis 5
    re-measures the *emitted* corpus against these planned shares, so this is
    the manifest that lets the report claim the numbers it later prints.
    """
    supervised = [job for job in jobs if not job.holdout]
    total = len(supervised)
    n_train_families = train_family_count(plan)
    by_family: dict[str, dict[str, object]] = {}
    for (
        work_type,
        spec,
    ) in plan.items():  # every train family reports, capped to 0 or not
        for family, meta in spec["families"].items():
            if not meta["holdout"]:
                by_family[f"{work_type}/{family}"] = {
                    "rows": 0,
                    "share": 0.0,
                    "planned_share": 0.0,
                    "holdout": False,
                }
    for job in supervised:
        key = f"{job.work_type}/{job.family}"
        slot = by_family.setdefault(
            key, {"rows": 0, "share": 0.0, "planned_share": 0.0, "holdout": False}
        )
        slot["rows"] += 1
    demand: dict[str, int] = {}
    for work_type, spec in plan.items():
        for family, meta in spec["families"].items():
            if meta["holdout"]:
                continue
            name = f"{work_type}/{family}"
            if name in by_family:
                # planned_share is the family's *appetite* (pre-cap), not a
                # promise: with fewer than 1/cap families in the plan every
                # family wants more than the cap allows, and truncating the
                # fat ones is the point. The emitted truth is `rows`, checked
                # against `cap` by the verify board's coverage axis.
                demand[name] = sum(
                    spec["variants_per_family"][rt] for rt in spec["record_types"]
                )
    planned_total = sum(demand.values()) or 1
    for name, slot in by_family.items():
        slot["share"] = round(slot["rows"] / total, 6) if total else 0.0
        wanted = demand.get(name)
        if wanted is not None:
            slot["planned_share"] = round(wanted / planned_total, 6)
            spec = plan[name.split("/", 1)[0]]
            slot["cap"] = _floor(
                min(spec["max_share"], config.family_max_share(n_train_families))
                * planned_total
            )
    return {
        "schema_version": config.SCHEMA_VERSION,
        "smoke": smoke,
        "plan_path": config.taxonomy_path(),
        "jobs": len(jobs),
        "supervised_rows": total,
        "eval_rows": len(jobs) - total,
        "cap_basis": (
            "family cap measured against the planned (pre-truncation) supervised "
            "pool, per spec §5.1; realised shares are reported beside it"
        ),
        "max_planned_share": max(
            (slot["planned_share"] for slot in by_family.values()), default=0.0
        ),
        "max_realised_share": max(
            (slot["share"] for slot in by_family.values()), default=0.0
        ),
        "families": {name: by_family[name] for name in sorted(by_family)},
        "work_types": {
            work_type: {
                "max_share": plan[work_type]["max_share"],
                "computer": plan[work_type]["computer"],
                "families": sorted(plan[work_type]["families"]),
            }
            for work_type in sorted(plan)
        },
    }


def write_plan(out_path: str, manifest: dict, jobs: list[Job]) -> None:
    """Persist manifest + job list atomically (whole file, tmp+swap discipline)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "manifest": manifest,
        "jobs": [job.as_record() for job in jobs],
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out_path)


def read_plan(path: str) -> tuple[dict, list[Job]]:
    """Load a plan file :func:`write_plan` wrote; inverse of it, field-checked."""
    with open(path, encoding="utf8") as handle:
        payload = json.load(handle)
    jobs = [Job(**record) for record in payload.get("jobs", [])]
    return payload.get("manifest", {}), jobs
