"""Publish: the last gate and the v3 dataset card (arch spec §6-§7; analysis spec §9).

``publish`` is not a formatter with a rubber stamp. It is the one place the
whole control plane has to say *no* and mean it, because past it the corpus
becomes public training data and a leak or an unpinned teacher is a fact about
the world's models, not a todo. It therefore refuses rather than warns:

* on **any red board axis** -- the same fifteen the CLI verify prints, run to
  completion (``--quick`` is a CI affordance, never a publish basis: an axis
  that was withheld has not been passed);
* for **want of the gold bar** -- the human, non-generated held-out set the
  near-duplicate fence (§5.11) leans on. The board *reports* its absence (a CI
  box has never held a human artefact); publish cannot ship without the fence,
  so absence here is a refusal, not a note;
* on an **empty corpus** -- publishing nothing is how a green CI goes on
  publishing nothing.

What it writes, when every one of those holds, is the dataset card: the public
face of the corpus, with its counts, its coverage, and the gate that cleared.
"""

from __future__ import annotations

import os

from . import config, write
from .teacher.prompts import BRIEF_KINDS  # the five prose lanes
from .verification.exam import EXAM_KIND
from .verification.implementation import IMPL_KIND
from .verification.preference import PREFER_KIND
from .verify_v3 import AXES, verify_dir

#: The agentic record type, named as the board names it; kept local so the
#: tally of kinds here reads identically to ``verify_dir``'s own sweep.
AGENTIC_KIND = "agentic"

#: Every shard the card tallies, in the order the card prints them.
SHARD_KINDS = (*BRIEF_KINDS, AGENTIC_KIND, EXAM_KIND, IMPL_KIND)


def _tally(out_dir: str) -> dict:
    """Rows per shard, the pair count, and the coverage the card reports.

    Read-only over the shards; a corrupt one is the board's to red (the gate
    below will refuse on it), so here it simply tallies zero rather than
    double-report the same break.
    """
    by_kind: dict[str, int] = {}
    families: set[tuple[str, str]] = set()
    work_types: set[str] = set()
    for kind in SHARD_KINDS:
        try:
            rows = write.read_jsonl(write.path_for("sft", kind, out_dir))
        except ValueError:
            by_kind[kind] = 0
            continue
        by_kind[kind] = len(rows)
        for row in rows:
            work_type = row.get("work_type")
            scenario = row.get("scenario_id")
            if isinstance(work_type, str):
                work_types.add(work_type)
            if isinstance(work_type, str) and isinstance(scenario, str):
                families.add((work_type, scenario[len(work_type) + 1 :]))
    try:
        pairs = len(write.read_jsonl(write.path_for("preference", "pairs", out_dir)))
    except ValueError:
        pairs = 0
    supervised = sum(by_kind[k] for k in SHARD_KINDS if k not in (EXAM_KIND, IMPL_KIND))
    return {
        "by_kind": by_kind,
        "rows": sum(by_kind.values()),
        "supervised": supervised,
        "eval": by_kind.get(EXAM_KIND, 0) + by_kind.get(IMPL_KIND, 0),
        "pairs": pairs,
        "families": len(families),
        "work_types": len(work_types),
    }


def _red_axes(report: dict) -> list[tuple[str, int]]:
    return [
        (name, len(report["axes"][name]["failures"]))
        for _, name in AXES
        if report["axes"][name]["failures"]
    ]


def _refusals(report: dict, gold_bar: str, tally: dict) -> list[str]:
    """Every reason this corpus may not go out, spelled out for the operator.

    Ordered so the most structural read first: an axis that did not run, then
    an axis that ran and failed, then the missing human fence, then emptiness.
    """
    reasons: list[str] = []
    for _, name in AXES:
        if report["axes"][name].get("skipped"):
            reasons.append(
                f"axis '{name}' did not run -- publish certifies on a full board, "
                "never a --quick one"
            )
    for name, count in _red_axes(report):
        reasons.append(f"axis '{name}' is red: {count} finding(s)")
    if not os.path.isfile(gold_bar):
        reasons.append(
            f"no gold bar at {gold_bar!r}: the near-duplicate fence (spec §5.11) "
            "has nothing on its far side; a real publish needs the human bar committed"
        )
    if tally["rows"] == 0:
        reasons.append(
            "nothing to publish: the corpus is empty (a green board over no rows "
            "is a green board over nothing)"
        )
    return reasons


def _card(
    out_dir: str, report: dict, tally: dict, hub_id: str | None, gold_bar: str
) -> str:
    """The dataset card: the public, auditable face of a cleared corpus."""
    lines = [
        "# Cosimo v3 — dataset card",
        "",
        f"- schema: `{config.SCHEMA_VERSION}`",
        f"- corpus root: `{out_dir}`",
        f"- hub repository: `{hub_id or 'unset'}`",
        f"- gold-bar fence: `{gold_bar}` (Jaccard ≥ {config.GOLDBAR_NEAR_DUP_THRESHOLD})",
        "",
        "## Counts",
        "",
        "| record type | rows |",
        "| --- | --: |",
    ]
    for kind in SHARD_KINDS:
        lines.append(f"| {kind} | {tally['by_kind'].get(kind, 0)} |")
    lines.append(f"| {PREFER_KIND} (pairs) | {tally['pairs']} |")
    lines += [
        "",
        f"- supervised rows: {tally['supervised']} · eval rows: {tally['eval']} "
        f"· pairs: {tally['pairs']} · total: {tally['rows']}",
        f"- coverage: {tally['work_types']} work types, {tally['families']} scenario families",
        "",
        "## Gates (verify_v3, all fifteen green)",
        "",
        "| # | axis | rows read |",
        "| --: | --- | --: |",
    ]
    for number, name in AXES:
        lines.append(f"| {number} | {name} | {report['axes'][name]['checked']} |")
    lines.append("")
    return "\n".join(lines)


def run_publish(
    out_dir: str, *, dry_run: bool = False, hub_id: str | None = None
) -> dict:
    """Run the publish gate; write the card only if the corpus may go out.

    Returns the verdict and everything the CLI needs to report -- ``refusals``
    spells each reason, ``wrote`` is the card path when one was written and
    ``None`` otherwise (a refusal or a dry run never writes).
    """
    gold_bar = config.gold_bar_v3_path()
    report = verify_dir(out_dir, quick=False, gold_bar_path=gold_bar)
    tally = _tally(out_dir)
    reasons = _refusals(report, gold_bar, tally)
    ok = not reasons
    hub = hub_id or config.hub_repo_id()
    card = _card(out_dir, report, tally, hub, gold_bar)
    card_path = os.path.join(config.publish_dir(), config.DATASET_CARD_FILENAME)
    wrote: str | None = None
    if ok and not dry_run:
        os.makedirs(os.path.dirname(card_path) or ".", exist_ok=True)
        with open(card_path, "w", encoding="utf8") as handle:
            handle.write(card)
        wrote = card_path
    return {
        "ok": ok,
        "refusals": reasons,
        "report": report,
        "tally": tally,
        "card": card,
        "card_path": card_path,
        "wrote": wrote,
        "dry_run": dry_run,
        "hub_id": hub,
    }
