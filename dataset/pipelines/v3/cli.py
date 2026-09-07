"""``python -m pipelines.v3.cli <command>`` -- the Airflow task boundary (spec §7).

One command per DAG task so a mapped Airflow task carries a string, not a
closure, and so every stage is debuggable from a laptop with the same bytes
that run on the cluster. The contract Airflow leans on: **exit 0 means the
output on disk is good**, anything else means do not schedule the downstream.
Unknown-command is exit 2 (usage), stage failure is exit 1 (data), so an
operator skimming the task log sees which kind of bad happened before reading
a word of it.

PR1 ships ``inventory``, ``packs`` and ``smoke`` wired; ``render``,
``verify``, ``prefer`` and ``publish`` are recognised, parse the flags spec
§7 shows (so the DAG's command lines are written and stable today) and then
exit 2 naming the PR that fills them -- "not built yet", never "not found".

Run from the repo root (``make v3-smoke``) or from ``dataset/``:
both paths resolve through the same ``sys.path`` bootstrap every corpus
module uses, so ``pipelines.core`` keeps its one identity in-process.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.dirname(os.path.dirname(_HERE))
for _p in (_DATASET, os.path.dirname(_DATASET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import config, inventory, stage, write  # noqa: E402
from .verification.invented_numbers import invented_numbers  # noqa: E402

EXIT_OK, EXIT_DATA, EXIT_USAGE = 0, 1, 2


def _split_types(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def cmd_inventory(args) -> int:
    plan_path = os.path.abspath(args.plan or config.taxonomy_path())
    try:
        plan = inventory.load_plan(plan_path)
    except inventory.PlanError as exc:
        print(f"inventory: {exc}", file=sys.stderr)
        return EXIT_DATA
    jobs = inventory.expand_jobs(plan, smoke=args.smoke)
    manifest = inventory.plan_manifest(jobs, plan, smoke=args.smoke)
    out = os.path.abspath(args.out or os.path.join(config.out_dir(), "plan.json"))
    inventory.write_plan(out, manifest, jobs)
    print(
        f"inventory: {manifest['jobs']} jobs "
        f"({manifest['supervised_rows']} supervised + {manifest['eval_rows']} eval) "
        f"across {len(manifest['families'])} train families -> {out}"
    )
    print(
        f"  family cap: basis={manifest['cap_basis'].split(',')[0]}, "
        f"max planned share={manifest['max_planned_share']:.4f}, "
        f"max realised share={manifest['max_realised_share']:.4f}"
    )
    return EXIT_OK


def cmd_packs(args) -> int:
    plan_path = os.path.abspath(
        args.plan or os.path.join(config.out_dir(), "plan.json")
    )
    if not os.path.isfile(plan_path):
        print(
            f"packs: no plan file at {plan_path}; run `inventory` first "
            "(the stages consume the committed plan, never re-derive it)",
            file=sys.stderr,
        )
        return EXIT_DATA
    _, jobs = inventory.read_plan(plan_path)
    report = stage.run_pack_stage(
        os.path.abspath(args.out or config.out_dir()),
        jobs,
        types=_split_types(args.types),
        limit=args.limit,
    )
    print(
        f"packs: {report['unique_packs']} unique packs "
        f"(computed {report['computed']}, already there {report['existing']}, "
        f"skipped via PackError {len(report['skipped'])})"
    )
    if report["skipped"]:
        by_family: dict[str, int] = {}
        for skip in report["skipped"]:
            key = f"{skip['work_type']}/{skip['family']}"
            by_family[key] = by_family.get(key, 0) + 1
        for key in sorted(by_family):
            print(f"  skipped {by_family[key]:>4}  {key}")
    if report["families_without_packs"]:
        for key in report["families_without_packs"]:
            print(
                f"packs: family {key} produced no pack in any variant -- "
                "computer or plan bug, not a skip",
                file=sys.stderr,
            )
        return EXIT_DATA
    return EXIT_OK


def _smoke_self_check(out_dir: str) -> list[str]:
    """Re-read what the smoke run wrote and believe none of it unverified.

    The board prints a happy table; the table means nothing unless someone
    checks the files it came from, so ``smoke`` reads its own output back
    through the same two readers every later stage uses (``write.read_jsonl``,
    ``invented_numbers``) rather than a private fast path -- the CI gate
    exercises the real verification edge, not a picture of it. A pack whose
    question contains a number its own ``allowed_numbers`` does not authorise
    is a corrupt fact computer or a corrupt pack file, and either way smoke
    may not pass on it.
    """
    problems: list[str] = []
    packs_dir = os.path.join(out_dir, "fact_packs")
    if not os.path.isdir(packs_dir):
        return [f"no fact_packs directory at {packs_dir}"]
    for name in sorted(os.listdir(packs_dir)):
        if not name.endswith(".jsonl") or name.startswith("_"):
            continue
        path = os.path.join(packs_dir, name)
        try:
            packs = write.read_jsonl(path)
        except ValueError as exc:
            problems.append(f"{path}: {exc}")
            continue
        if not packs:
            problems.append(f"{path}: empty")
        for pack in packs:
            rid = pack.get("id", "?")
            allowed = [float(a) for a in (pack.get("allowed_numbers") or [])]
            if not allowed:
                problems.append(f"{rid}: empty allowed_numbers")
                continue
            if any(a != a or a in (float("inf"), float("-inf")) for a in allowed):
                problems.append(f"{rid}: allowed_numbers not finite")
                continue
            register = pack.get("register")
            if register not in config.VALID_REGISTERS:
                problems.append(f"{rid}: register {register!r} is not a known register")
            question = (pack.get("question") or "").strip()
            if not question:
                problems.append(f"{rid}: empty question")
                continue
            for token in invented_numbers(question, allowed):
                problems.append(
                    f"{rid}: question quotes {token!r}, which its "
                    "allowed_numbers does not authorise"
                )
    return problems


def cmd_smoke(args) -> int:
    out_dir = os.path.abspath(args.out or os.path.join(config.out_dir(), "_smoke"))
    try:
        plan = inventory.load_plan()
    except inventory.PlanError as exc:
        print(f"smoke: {exc}", file=sys.stderr)
        return EXIT_DATA
    jobs = inventory.expand_jobs(plan, smoke=True)
    report = stage.run_pack_stage(out_dir, jobs)
    families_by_work: dict[str, set[str]] = {}
    for job in jobs:
        families_by_work.setdefault(job.work_type, set()).add(job.family)
    skipped_by_work: dict[str, int] = {}
    for row in report["skipped"]:
        skipped_by_work[row["work_type"]] = skipped_by_work.get(row["work_type"], 0) + 1
    for work_type in sorted(families_by_work):
        path = os.path.join(out_dir, "fact_packs", f"{work_type}.jsonl")
        made = len(write.read_jsonl(path)) if os.path.isfile(path) else 0
        flag = "" if made else "  <-- NO PACKS"
        print(
            f"{work_type:<34} families {len(families_by_work[work_type]):>2}  "
            f"packs {made:>2}  skipped {skipped_by_work.get(work_type, 0):>2}{flag}"
        )
    for family_key in report["families_without_packs"]:
        print(
            f"smoke: family {family_key} produced no pack in any variant",
            file=sys.stderr,
        )
    problems = _smoke_self_check(out_dir)
    for problem in problems:
        print(f"smoke: {problem}", file=sys.stderr)
    if report["families_without_packs"] or problems:
        print("SMOKE FAIL", file=sys.stderr)
        return EXIT_DATA
    print(
        f"smoke: ok -- {report['computed'] + report['existing']} packs on disk, "
        f"{len(report['skipped'])} variants skipped by guard"
    )
    return EXIT_OK


def _not_built(pr: str):
    def run(args) -> int:
        print(
            f"{args.command}: arrives in {pr} -- recognised and parsed "
            "(spec §7 keeps the DAG command lines stable across PRs), but not wired yet",
            file=sys.stderr,
        )
        return EXIT_USAGE

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipelines.v3.cli",
        description="Cosimo v3 corpus control plane (spec §5, §7)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "inventory", help="expand work_types.yaml into the deterministic job plan"
    )
    p.add_argument(
        "--plan", help="plan yaml (default: dataset/taxonomy/work_types.yaml)"
    )
    p.add_argument("--out", help="plan json to write (default: <out>/plan.json)")
    p.add_argument(
        "--smoke", action="store_true", help="one successful variant per family"
    )
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("packs", help="compute fact packs for a plan (idempotent)")
    p.add_argument("--plan", help="plan json written by `inventory`")
    p.add_argument("--out", help="corpus root (default: config.out_dir())")
    p.add_argument("--types", help="comma list of record types to cover")
    p.add_argument(
        "--limit", type=int, help="truncate the job list (bake-off/dev only)"
    )
    p.set_defaults(func=cmd_packs)

    p = sub.add_parser("smoke", help="CI gate: smoke inventory + packs + self-check")
    p.add_argument("--out", help="corpus root (default: <out>/_smoke)")
    p.set_defaults(func=cmd_smoke)

    # PR2+ -- present in the parser so the DAG lines exist today; exit 2 until wired.
    p = sub.add_parser("render", help="teacher render stages (PR2)")
    p.add_argument("--types", help="comma list of record types")
    p.add_argument("--limit", type=int)
    p.add_argument("--live", action="store_true")
    p.set_defaults(func=_not_built("PR2"))

    p = sub.add_parser("verify", help="the v3 verification board (PR2-PR4)")
    p.add_argument("--quick", action="store_true", help="skip axes 10-12 (spec §6)")
    p.set_defaults(func=_not_built("PR2"))

    p = sub.add_parser("prefer", help="contrastive preference pairs (PR4)")
    p.add_argument("--types", help="limit pitfall families")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=_not_built("PR4"))

    p = sub.add_parser("publish", help="the v3 branch of push_to_hub (PR4)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--repo-id")
    p.set_defaults(func=_not_built("PR4"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
