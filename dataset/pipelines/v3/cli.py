"""``python -m pipelines.v3.cli <command>`` -- the Airflow task boundary (spec §7).

One command per DAG task so a mapped Airflow task carries a string, not a
closure, and so every stage is debuggable from a laptop with the same bytes
that run on the cluster. The contract Airflow leans on: **exit 0 means the
output on disk is good**, anything else means do not schedule the downstream.
Unknown-command is exit 2 (usage), stage failure is exit 1 (data), so an
operator skimming the task log sees which kind of bad happened before reading
a word of it.

PR2 wired ``inventory``, ``packs``, ``smoke``, ``render`` and ``verify``;
PR3 added the agentic lane to render and to the board; PR4 adds the exam
lane (composed, no teacher), the implementation lane (composed record,
authored limitations, sandboxed suite) and the corpus-measurement and
hidden-test axes. ``prefer`` and ``publish`` were the last to arrive and now
run for real, so every command the DAG's command lines name (spec §7) is
wired: each parses today and does the work it names, rather than exiting 2 to
mark a PR still ahead of it.

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
from .prefer import run_prefer_stage  # noqa: E402
from .publish import run_publish  # noqa: E402
from .render.agentic import KIND as AGENTIC_KIND  # noqa: E402
from .render.agentic import run_agentic_stage  # noqa: E402
from .render.exam import EXAM_KIND, run_exam_stage  # noqa: E402
from .render.implementation import IMPL_KIND, run_impl_stage  # noqa: E402
from .render.prose import run_render_stage  # noqa: E402
from .teacher.client import TeacherError, teacher_from_env  # noqa: E402
from .teacher.prompts import BRIEF_KINDS  # noqa: E402
from .verification.invented_numbers import invented_numbers  # noqa: E402
from .verify_v3 import AXES, verify_dir  # noqa: E402

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
    work_types = _split_types(args.work_type)
    if work_types:
        known = {job.work_type for job in jobs}
        unknown = sorted(set(work_types) - known)
        if unknown:
            print(
                f"packs: unknown --work-type {', '.join(unknown)}; this committed "
                f"plan covers {', '.join(sorted(known)) or 'no work types'} -- if "
                "the taxonomy changed, re-run `inventory` to refresh the plan",
                file=sys.stderr,
            )
            return EXIT_USAGE
        wanted = set(work_types)
        jobs = [job for job in jobs if job.work_type in wanted]
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


def cmd_render(args) -> int:
    out_dir = os.path.abspath(args.out or config.out_dir())
    plan_path = os.path.abspath(
        args.plan or os.path.join(config.out_dir(), "plan.json")
    )
    if not os.path.isfile(plan_path):
        print(
            f"render: no plan file at {plan_path}; run `inventory` first "
            "(the stages consume the committed plan, never re-derive it)",
            file=sys.stderr,
        )
        return EXIT_DATA
    types = _split_types(args.types)
    allowed = set(BRIEF_KINDS) | {AGENTIC_KIND, EXAM_KIND, IMPL_KIND}
    if types and not set(types) <= allowed:
        unknown = sorted(set(types) - allowed)
        print(
            f"render: the render stages cover {', '.join(sorted(allowed))}; "
            f"unknown --types {', '.join(unknown)} (the preference and eval "
            "slices arrive with PR4's later stages)",
            file=sys.stderr,
        )
        return EXIT_USAGE
    # One command line, up to four stages: the DAG's cell stays whole whether
    # it runs prose, agentic, exam, implementation, or all four, and each stage
    # keeps its own ledger -- the prose board says "planned prose jobs", the
    # agentic one adds the mix tallies the PR3 gate reads, the exam one the
    # liturgy tally the PR4 share axes measure, because a trajectory mix is a
    # property of the schedule and a paragraph mix is not.
    prose_types = None if types is None else tuple(t for t in types if t in BRIEF_KINDS)
    runs = []
    if prose_types is None or prose_types:
        runs.append(
            (
                "prose",
                lambda: run_render_stage(
                    out_dir,
                    jobs,
                    teacher,
                    types=prose_types,
                    limit=args.limit,
                    holdout=args.holdout,
                ),
            )
        )
    if types is None or AGENTIC_KIND in types:
        runs.append(
            (
                "agentic",
                lambda: run_agentic_stage(
                    out_dir, jobs, teacher, limit=args.limit, holdout=args.holdout
                ),
            )
        )
    if types is None or EXAM_KIND in types:
        runs.append(
            (
                "exam",
                lambda: run_exam_stage(
                    out_dir, jobs, limit=args.limit, holdout=args.holdout
                ),
            )
        )
    if types is None or IMPL_KIND in types:
        runs.append(
            (
                "implementation",
                lambda: run_impl_stage(
                    out_dir, jobs, teacher, limit=args.limit, holdout=args.holdout
                ),
            )
        )
    try:
        _, jobs = inventory.read_plan(plan_path)
        teacher = teacher_from_env(live=True if args.live else None)
        boards = [(label, run()) for label, run in runs]
    except (inventory.PlanError, TeacherError, ValueError, OSError) as exc:
        print(f"render: {exc}", file=sys.stderr)
        return EXIT_DATA
    missing_total = 0
    for label, report in boards:
        bucket = report.get("bucket", "sft")
        logs = report.get("teacher_logs", 0)
        print(
            f"render: {report['rendered']} rows rendered, "
            f"{report['existing']} already on disk, {report['dead_lettered']} dead-lettered "
            f"(of {report['jobs_seen']} planned {label} jobs; "
            f"{len(report['skipped_by_pack_gate'])} skipped by the pack gate) "
            f"-> {out_dir}/{bucket}" + (f"  [+{logs} teacher logs]" if logs else "")
        )
        for kind in sorted(report["by_kind"]):
            cell = report["by_kind"][kind]
            extra = ""
            if label == "agentic":
                extra = (
                    f"  no_call {report['no_call']:>3}  faulted {report['faulted']:>3}"
                    f"  calls {report['tool_calls']:>4}"
                )
            elif label == "exam":
                extra = f"  liturgy {report['liturgy']:>3}"
            print(
                f"  {kind:<12} rendered {cell['rendered']:>5}  "
                f"existing {cell['existing']:>5}{extra}"
            )
        missing_total += len(report["missing_packs"])
        for row in report["missing_packs"][:8]:
            print(
                f"  missing pack: {row['work_type']}/{row['family']}"
                f"/{row['record_type']}/{row['variant']} -- {row['reason']}",
                file=sys.stderr,
            )
        if len(report["missing_packs"]) > 8:
            print(
                f"  ... and {len(report['missing_packs']) - 8} more",
                file=sys.stderr,
            )
    if missing_total:
        print(
            "render: the packs stage has not covered this plan -- the DAG does not "
            "schedule render before packs",
            file=sys.stderr,
        )
        return EXIT_DATA
    return EXIT_OK


def cmd_verify(args) -> int:
    out_dir = os.path.abspath(args.out or config.out_dir())
    report = verify_dir(out_dir, quick=args.quick)
    mode = (
        "quick (--quick: the hidden-suite and near-duplicate axes are witheld; "
        "schema, replay, the share bands and teacher pinning are cheap and run)"
    )
    print(
        f"verify: {out_dir} -- {report['rows']} rows"
        + (f" [{mode}]" if args.quick else "")
    )
    for number, name in AXES:
        axis = report["axes"][name]
        if axis["failures"]:
            mark = f"FAIL ({len(axis['failures'])})"
        elif axis.get("skipped"):
            mark = "skipped"  # --quick withheld an expensive axis on purpose
        elif axis.get("note"):
            mark = "reported"  # measured, not certifiable at this support
        else:
            mark = "ok"
        note = f"  [{axis['note']}]" if axis.get("note") else ""
        print(f"  axis {number} {name:<32} {axis['checked']:>5} rows  {mark}{note}")
        for failure in axis["failures"][:8]:
            print(f"    {failure['id']}: {failure['problem']}", file=sys.stderr)
        if len(axis["failures"]) > 8:
            print(
                f"    ... and {len(axis['failures']) - 8} more",
                file=sys.stderr,
            )
    if not report["ok"]:
        print("VERIFY FAIL", file=sys.stderr)
        return EXIT_DATA
    print(f"verify: ok -- {report['rows']} rows clean across {len(AXES)} axes")
    return EXIT_OK


def cmd_prefer(args) -> int:
    """Second pass over the shipped rows: pairs the draw picks, not the dice."""
    out_dir = os.path.abspath(args.out or config.out_dir())
    types = _split_types(args.types)
    if types and not set(types) <= set(config.PREF_PROBABILITIES):
        unknown = sorted(set(types) - set(config.PREF_PROBABILITIES))
        print(
            f"prefer: pairs are drawn for "
            f"{', '.join(sorted(config.PREF_PROBABILITIES))}; unknown --types "
            f"{', '.join(unknown)} (exam and implementation rows are their own "
            "contracts and carry no pair)",
            file=sys.stderr,
        )
        return EXIT_USAGE
    try:
        teacher = teacher_from_env(live=True if args.live else None)
        report = run_prefer_stage(
            out_dir, teacher, types=types, limit=args.limit, holdout=args.holdout
        )
    except (TeacherError, ValueError, OSError) as exc:
        print(f"prefer: {exc}", file=sys.stderr)
        return EXIT_DATA
    print(
        f"prefer: {report['rendered']} pairs written, "
        f"{report['existing']} already on disk, {report['dead_lettered']} dead-lettered "
        f"(of {report['jobs_seen']} drawn from the shipped rows; "
        f"{len(report['skipped_by_pack_gate'])} skipped by the pack gate) -> {out_dir}"
    )
    for kind in sorted(report["by_kind"]):
        cell = report["by_kind"][kind]
        print(
            f"  {kind:<12} pairs {cell['rendered']:>5}  existing {cell['existing']:>5}"
        )
    for row in report["skipped_by_pack_gate"][:8]:
        print(
            f"  pair skipped: {row['work_type']}/{row['scenario_id']}"
            f"/{row.get('parent_kind')}/{row.get('pitfall')} -- {row['reason']}",
            file=sys.stderr,
        )
    if len(report["skipped_by_pack_gate"]) > 8:
        print(
            f"  ... and {len(report['skipped_by_pack_gate']) - 8} more",
            file=sys.stderr,
        )
    return EXIT_OK


def cmd_publish(args) -> int:
    """The last gate: certify the corpus whole, then write the card or refuse."""
    out_dir = os.path.abspath(args.out or config.out_dir())
    result = run_publish(
        out_dir, dry_run=args.dry_run, hub_id=args.repo_id or config.hub_repo_id()
    )
    for reason in result["refusals"]:
        print(f"publish: {reason}", file=sys.stderr)
    if not result["ok"]:
        print("PUBLISH REFUSED", file=sys.stderr)
        return EXIT_DATA
    tally = result["tally"]
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"publish: {verb} {result['card_path']!r} -- {tally['rows']} rows "
        f"({tally['supervised']} supervised, {tally['eval']} eval, "
        f"{tally['pairs']} pairs) over {tally['work_types']} work types / "
        f"{tally['families']} families, clean across {len(AXES)} axes"
    )
    if args.dry_run:
        print("publish: --dry-run wrote nothing", file=sys.stderr)
    return EXIT_OK


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
    p.add_argument(
        "--work-type",
        help="comma list of work types to cover (default: all); the DAG maps "
        "one task per work type so a broken computer cannot block the others",
    )
    p.add_argument("--types", help="comma list of record types to cover")
    p.add_argument(
        "--limit", type=int, help="truncate the job list (bake-off/dev only)"
    )
    p.set_defaults(func=cmd_packs)

    p = sub.add_parser("smoke", help="CI gate: smoke inventory + packs + self-check")
    p.add_argument("--out", help="corpus root (default: <out>/_smoke)")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("render", help="prose render + repair loop (PR2)")
    p.add_argument("--plan", help="plan json (default: <out>/plan.json)")
    p.add_argument("--out", help="corpus root (default: config.out_dir())")
    p.add_argument(
        "--types",
        help="comma list of record types (prose kinds, 'agentic', 'exam', "
        "'implementation')",
    )
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--live",
        action="store_true",
        help="force the live teacher (COSIMO_V3_LIVE=1 also does; default is the fixture)",
    )
    p.add_argument(
        "--holdout",
        action="store_true",
        help="render the HOLDOUT families into eval/ instead of the train "
        "families into sft/ (amendment §E: the eval slice is generated, not "
        "borrowed from shipped families)",
    )
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("verify", help="the v3 verification board (axes 1-5 in PR2)")
    p.add_argument("--out", help="corpus root (default: config.out_dir())")
    p.add_argument(
        "--quick",
        action="store_true",
        help="skip the costly dedup/gold-bar axes at publish time (spec §6)",
    )
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("prefer", help="contrastive preference pairs (PR4)")
    p.add_argument("--out", help="corpus root (default: config.out_dir())")
    p.add_argument(
        "--types",
        help="comma list of parent record types (default: every paired kind)",
    )
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--live",
        action="store_true",
        help="force the live teacher (COSIMO_V3_LIVE=1 also does; default is the fixture)",
    )
    p.add_argument(
        "--holdout",
        action="store_true",
        help="pair the eval/ rows into preference/eval_pairs.jsonl instead of "
        "the sft/ rows into preference/pairs.jsonl",
    )
    p.set_defaults(func=cmd_prefer)

    p = sub.add_parser("publish", help="the v3 branch of push_to_hub (PR4)")
    p.add_argument("--out", help="corpus root (default: config.out_dir())")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="run the gate and print the card without writing it",
    )
    p.add_argument("--repo-id", help="the Hub repository id to record on the card")
    p.set_defaults(func=cmd_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
