"""Stage glue between the plan and the disk (spec §6.1 diagram, pack box).

Two functions, one each side of the ``fact_packs`` file:

* :func:`pack_record` -- the canonical ``{"id": pack_id, ...}`` line every
  fact pack occupies in ``<out>/fact_packs/<work_type>.jsonl``, including the
  ``verification.computed_by`` stamp that lets the audit trail answer "which
  code made this number" without re-deriving it;
* :func:`run_pack_stage` -- the idempotent writer the ``packs`` CLI calls,
  whose report (computed/skipped/existing) is what an operator reads at 3am
  to decide whether to re-run.

The idempotence is not a convenience: resume *is* re-running. The teacher
stages (PR2/PR3) sit downstream of these files, and the rule "ask the plan,
check the disk, ask the teacher only for the difference" -- the one rule that
keeps a 130k-completion budget bounded -- is only enforceable because this
stage can be replayed over a half-finished corpus a hundred times and
converge. The pack id is derived from the pack seed, so a replayed run
reproduces each id byte-for-byte and ``append_unique`` does the set
difference by itself.
"""

from __future__ import annotations

from . import write
from .packs import COMPUTERS, PackError, compute_pack
from .seed import pack_seed

#: The pack line's id namespace. ``cosimov3pack`` sits outside both supervised
#: prefixes on purpose (spec §4): a pack is provenance, never a training row,
#: and no verify axis that counts SFT ids may ever see one.
PACK_ID_PREFIX = "cosimov3pack"


def pack_id(work_type: str, family: str, variant: int) -> str:
    return f"{PACK_ID_PREFIX}_{pack_seed(work_type, family, variant):016x}"


def pack_record(work_type: str, family: str, variant: int) -> dict:
    """Compute one pack and wrap it in its shard line. Raises PackError."""
    pack = compute_pack(work_type, family, variant)
    return {
        "id": pack_id(work_type, family, variant),
        **pack.to_dict(),
        "verification": {
            "computed_by": getattr(COMPUTERS[work_type], "__module__", work_type),
            "computer": COMPUTERS[work_type].__name__,
            "pack_seed": f"{pack.seed:016x}",
            "verified": True,
        },
    }


def run_pack_stage(out_dir: str, jobs, *, types=None, limit=None) -> dict:
    """Write ``fact_packs/`` for *jobs*; returns the stage report. Never raises PackError.

    ``PackError`` is *data*, reported per family and never fatal (spec §5.3):
    a computer rejecting a drawn parameter set is the guard doing its job.
    What *is* fatal is a family the plan promised with zero packs across all
    its variants -- that is an authoring bug, not a skip, and the report says
    so with ``families_without_packs`` for the board to fail on.

    ``--types`` filters *before* ``--limit``, and both before any disk write:
    the spec's own bake-off line is ``render --types analysis,memo --limit
    200``, and a caller who caps the plan first would watch the first 200
    ``exam`` jobs exhaust the budget while asking for memos and getting
    none. What the flag promises the stage does, in that order, deterministically.
    """
    if types is not None:
        wanted = set(types)
        jobs = [job for job in jobs if job.record_type in wanted]
    if limit is not None:
        jobs = jobs[: max(0, limit)]
    by_work: dict[str, list] = {}
    attempted: set[tuple[str, str]] = set()
    produced: set[tuple[str, str]] = set()
    for job in jobs:
        key = (job.work_type, job.family, job.variant)
        if key in by_work.setdefault(job.work_type, {}):
            continue
        by_work[job.work_type][key] = job
        attempted.add((job.work_type, job.family))
    report = {
        "jobs_seen": len(jobs),
        "unique_packs": 0,
        "computed": 0,
        "existing": 0,
        "skipped": [],
        "families_without_packs": [],
    }
    for work_type in sorted(by_work):
        records = []
        wanted_keys = []
        for key in sorted(by_work[work_type]):
            wt, family, variant = key
            report["unique_packs"] += 1
            try:
                records.append(pack_record(wt, family, variant))
            except PackError as exc:
                report["skipped"].append(
                    {
                        "id": pack_id(wt, family, variant),
                        "work_type": wt,
                        "family": family,
                        "variant": variant,
                        "reason": str(exc),
                    }
                )
                continue
            wanted_keys.append(key)
            produced.add((wt, family))
        path = write.path_for("fact_packs", work_type, out_dir)
        added, already = write.append_unique(path, records)
        report["computed"] += added
        report["existing"] += already
    for work_type in sorted(by_work):
        for family in sorted({f for (w, f) in attempted if w == work_type}):
            if (work_type, family) not in produced:
                report["families_without_packs"].append(f"{work_type}/{family}")
    return report
